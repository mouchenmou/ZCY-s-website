"""GALATEA adaptation of an alternating differentiable MBRL loop.

Not the authors' implementation. No expert, old WM, response loss or learned
normalization. Dynamics is conditioned on the executed PD command, not raw action.
"""
import hashlib
import torch
from torch import nn
from wm4_model import control


def mlp(dims):
    layers=[]
    for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(dims)-2: layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def state_scale(device):
    x=torch.ones(312,device=device)
    x[29:58]=10; x[65:68]=2; x[68:71]=10; x[155:239]=2
    x[243:258]=20; x[258:287]=10
    return x


def physics_scale(device):
    return torch.tensor([1000.]*7+[100.]*22+[100.]*7+[10.]*22+[1.]*4+[.1],device=device)


def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.net=mlp([323,1024,1024,512,512,29])
        nn.init.normal_(self.net[-1].weight,std=.001);nn.init.zeros_(self.net[-1].bias)
    def forward(self,obs):return self.net(obs).tanh()


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__();self.net=mlp([404,2048,2048,2048,312]);self.reward_failure=mlp([961,512,256,2])
        nn.init.normal_(self.net[-1].weight,std=.001);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('scale',state_scale('cpu'));self.register_buffer('pscale',physics_scale('cpu'))
        delta=self.scale.clone()*.05;delta[:29]=.05;delta[29:58]=2.;delta[243:258]=10;delta[258:287]=5
        self.register_buffer('delta',delta)
    def forward(self,s,target,action,last,history,ref,physics,age):
        y=self.net(torch.cat((s/self.scale,target,physics/self.pscale),-1))
        v=s[:,:287]+y[:,:287]*self.delta[:287]
        v=torch.cat((v[:,:61],nn.functional.normalize(v[:,61:65],dim=-1,eps=1e-6),v[:,65:239],
                     nn.functional.normalize(v[:,239:243],dim=-1,eps=1e-6),v[:,243:]),-1)
        sn=torch.cat((v,y[:,287:297].sigmoid(),s[:,297:312]+y[:,297:312]*self.delta[297:312]),-1)
        rf=self.reward_failure(torch.cat((s/self.scale,sn/self.scale,action,last,history,ref,physics/self.pscale,age/1000),-1))
        return sn,rf[:,0]*10,rf[:,1],y[:,287:297]


def advance_history(hist,sn):
    # Force-active probability, with the SAME physical definition as official history.
    # This is separate from distance-gated hand-object contact probability.
    return torch.cat((hist[:,5:],sn[:,292:297]),-1)


def critic_features(s,last,prev,mp,hist,ref,physics,age,remaining,joints):
    return torch.cat((s/state_scale(s.device),last,prev,mp,hist,ref,physics/physics_scale(s.device),
                      age/1000,remaining[:,None]/10,joints.float()),-1)


class Critic(nn.Module):
    def __init__(self):super().__init__();self.net=mlp([694,256,128,1])
    def forward(self,x):return self.net(x).squeeze(-1)*10


def sample_control(remaining,joints,max_duration,generator):
    """Only present context + fresh RNG. No recorded future delay/mask accepted."""
    n=len(remaining);dev=remaining.device
    delay=torch.randint(0,2,(n,),device=dev,generator=generator).bool()
    start=(remaining<=0)&(torch.rand(n,device=dev,generator=generator)<.15)
    sampled=torch.zeros_like(joints)
    sampled.scatter_(1,torch.rand(n,22,device=dev,generator=generator).topk(3,dim=-1).indices,True)
    joints=torch.where(start[:,None],sampled,joints)
    duration=torch.randint(1,max_duration+1,(n,),device=dev,generator=generator)
    remaining=torch.where(start,duration,remaining)
    mask=(remaining>0)[:,None]&joints
    return delay,mask,(remaining-1).clamp_min(0),joints


def gae(rewards,values,continuations,gamma=.99,lam=.95):
    """values has H+1 entries. A true terminal (continuation=0) never bootstraps."""
    running=torch.zeros_like(rewards[0]);targets=[]
    for t in range(len(rewards)-1,-1,-1):
        delta=rewards[t]+gamma*continuations[t]*values[t+1]-values[t]
        running=delta+gamma*lam*continuations[t]*running
        targets.append(running+values[t])
    return torch.stack(targets[::-1])

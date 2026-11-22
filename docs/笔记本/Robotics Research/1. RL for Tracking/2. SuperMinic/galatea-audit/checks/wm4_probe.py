"""Counterfactual tests rebuilt from seed and complete simulator history.

No tensor-only q/qd restoration. Each branch constructs a fresh PhysX process.
No reset occurs during a branch; terminal flags remain intact and are reported.
"""
import argparse, json, os, time, types
from pathlib import Path
import numpy as np
from wm4_collect import make, state, reference, scripted, guard
import torch
from wm4_model import control


def run(a):
    start=time.monotonic();env,cfg,cap,phys,c=make(a.ids.split(','),a.envs,a.seed)
    n=env.num_envs
    # Valid reset frames with sufficient reference remaining for a 32-step test.
    def sample(self,idx):
        valid=self._demo_value('reset_valid',idx).bool()
        f=torch.arange(valid.shape[1],device='cuda')[None,:]
        allowed=valid&(f<(self.demo_seq_len[idx,None]-48))
        if not (allowed.sum(-1)>0).all():raise RuntimeError('No valid probe reset frames')
        rank=(torch.rand(len(idx),device='cuda')*allowed.sum(-1)).long()
        return ((allowed.long().cumsum(-1)>rank[:,None])&allowed).long().argmax(-1)
    env._sample_reset_frames=types.MethodType(sample,env);env.reset()
    noise=torch.zeros(n,29,device='cuda');warm=[]
    for step in range(4):
        act,noise=scripted(env,noise);env.step(act);warm.append(state(env).cpu())
    initial_done=env.reset_buf.clone().bool().cpu()
    base=torch.load(a.base,weights_only=False) if a.base else None
    rows={};maxerr=0.
    for h in range(32):
        guard(start,180)
        s=state(env);prev=env.prev_targets.clone();last=env.last_actions.clone()
        mp=env.mask_previous_command.clone();hist=env.tip_contact_history.flatten(1).float().clone()
        ref=reference(env);age=env.running_progress_buf[:,None].float().clone()
        act,noise=scripted(env,noise)
        if base is not None:act=base['action'][h].to('cuda').clone()
        if h<8:
            joint=7+torch.arange(n,device='cuda')%22
            act[torch.arange(n,device='cuda'),joint]+=a.branch*.2
            act=act.clamp(-1,1)
        _,r,done,_=env.step(act)
        target=env.pos_targets.clone()
        err=float((control(s,act,prev,last,mp,cap['delay'],cap['mask'],c)-target).abs().max())
        maxerr=max(maxerr,err);assert err<2e-6
        row={'s':s,'sn':state(env),'action':act,'previous':prev,'last':last,'mask_previous':mp,
             'history':hist,'ref':ref,'age':age,'physics':phys.clone(),'target':target,
             'delay':cap['delay'].clone(),'mask':cap['mask'].clone(),'reward':r.clone(),'done':done.clone().bool()}
        for k,v in row.items():rows.setdefault(k,[]).append(v.cpu())
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    data={k:torch.stack(v) for k,v in rows.items()}
    data.update({'warmup':torch.stack(warm),'initial_done':initial_done,
                 'contract':c,'ids':a.ids.split(','),'seed':a.seed,'branch':a.branch})
    torch.save(data,str(out)+'.tmp');os.replace(str(out)+'.tmp',out)
    meta={'max_pd_error':maxerr,'seconds':time.monotonic()-start,
          'task_valid_cases':{str(h):int((~data['done'][:h-1].any(0)).sum()) for h in (1,8,32)}}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=9091);p.add_argument('--branch',type=int,default=0)
    p.add_argument('--envs',type=int,default=120)
    p.add_argument('--base');run(p.parse_args())

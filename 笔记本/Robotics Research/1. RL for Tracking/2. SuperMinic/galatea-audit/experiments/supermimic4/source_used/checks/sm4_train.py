"""Bounded four-group from-scratch MBRL smoke test. No formal-run CLI."""
import argparse,json,time,os,random,shutil,traceback
from pathlib import Path
from sm4_env import build_env,physical_state,References,ROOT
import numpy as np
import torch
from torch.nn import functional as F
from sm4_model import Actor,WorldModel,Critic,control,advance_history,critic_features,sample_control,gae,digest
from omegaconf import OmegaConf

OUT=ROOT/'experiments/supermimic4'


def save_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def atomic_save(x,p):torch.save(x,str(p)+'.tmp');os.replace(str(p)+'.tmp',p)


class Replay:
    def __init__(self,n,capacity=1000):self.n=n;self.capacity=capacity;self.total=0;self.data={}
    def add(self,row):
        for k,v in row.items():
            if k not in self.data:self.data[k]=torch.empty((self.capacity,*v.shape),device=v.device,dtype=v.dtype)
            self.data[k][self.total%self.capacity].copy_(v)
        self.total+=1
    def sample(self,batch,generator,horizon=1):
        # Smoke fits in ring; chronological indexing also works after wrap.
        oldest=max(0,self.total-self.capacity)
        times=torch.arange(oldest,self.total,device='cuda');indices=times%self.capacity
        starts=self.data['selected'][indices].nonzero()
        if not len(starts):raise RuntimeError('No physically valid non-reset replay states')
        pick=starts[torch.randint(len(starts),(batch,),device='cuda',generator=generator)]
        t=times[pick[:,0]];lane=pick[:,1];rows=[];alive=torch.ones(batch,dtype=torch.bool,device='cuda')
        for j in range(horizon):
            ix=(t+j).clamp_max(self.total-1)%self.capacity
            row={k:v[ix,lane] for k,v in self.data.items()}
            alive=alive&(t+j<self.total)&row['selected']
            if j:alive=alive&(row['episode']==rows[0]['episode'])
            row['valid']=alive.clone();rows.append(row);alive=alive&~row['done']
        return rows


def wm_update(wm,opt,rows,c):
    first=rows[0];s=first['s'];prev=first['previous'];last=first['last'];mp=first['mask_previous'];hist=first['history']
    total=s.new_zeros(());den=s.new_zeros(());parts={}
    for r in rows:
        target=control(s,r['action'],prev,last,mp,r['delay'],r['mask'],c)
        sn,reward,failure,logits=wm(s,target,r['action'],last,hist,r['ref'],r['physics'],r['age'])
        blocks=[]
        for a,b in ((0,29),(29,58),(58,71),(71,155),(155,239),(239,243),(243,258),(258,287),(297,312)):
            blocks.append(((sn[:,a:b]-r['sn'][:,a:b])/wm.delta[a:b]).square().mean(-1))
        phys=torch.stack(blocks).mean(0)
        contacts=F.binary_cross_entropy_with_logits(logits,r['sn'][:,287:297],reduction='none').mean(-1)
        rew=((reward-r['reward'])/10).square()
        fail=F.binary_cross_entropy_with_logits(failure,r['failure'].float(),reduction='none')
        valid=r['valid'].float();total+=((phys+contacts+rew+fail)*valid).sum();den+=valid.sum()
        if not parts:parts={'physical':float((phys*valid).sum()/valid.sum()),'contact':float((contacts*valid).sum()/valid.sum()),
                            'reward':float((rew*valid).sum()/valid.sum()),'failure':float((fail*valid).sum()/valid.sum())}
        s=sn;hist=advance_history(hist,s);prev=target;mp=target[:,7:];last=r['action']
    loss=total/den.clamp_min(1);opt.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
    return {'loss':float(loss),'grad':float(grad),'valid_supervised_steps':int(den),**parts}


def imagined(actor,wm,critic,r,refs,c,generator,max_duration,horizon=32):
    # Explicit whitelist of INITIAL replay context. No future trajectory passed.
    s=r['s'];prev=r['previous'];last=r['last'];mp=r['mask_previous'];hist=r['history'];physics=r['physics']
    age=r['age'];frame=r['frame'];motion=r['motion'];remaining=r['mask_remaining'];joints=r['mask_joints']
    weights=[];rewards=[];values=[];conts=[];features=[];survival=torch.ones(len(s),device=s.device);objective=torch.zeros_like(survival)
    for t in range(horizon):
        ref=refs.get(motion,frame)
        x=critic_features(s,last,prev,mp,hist,ref,physics,age,remaining,joints)
        v=critic(x);features.append(x.detach());values.append(v.detach());weights.append(survival.detach())
        a=actor(refs.obs(s,last,motion,frame,generator=generator))
        delay,mask,remaining,joints=sample_control(remaining,joints,max_duration,generator)
        target=control(s,a,prev,last,mp,delay,mask,c)
        sn,reward,failure,_=wm(s,target,a,last,hist,ref,physics,age)
        continuation=(1-failure.sigmoid())*(frame+4<refs.length[motion]).float()
        objective=objective+(.99**t)*survival*reward
        rewards.append(reward.detach());conts.append(continuation.detach())
        survival=survival*continuation
        s=sn;hist=advance_history(hist,s);prev=target;mp=target[:,7:];last=a;age=age+1;frame=frame+1
    boot=critic(critic_features(s,last,prev,mp,hist,refs.get(motion,frame),physics,age,remaining,joints))
    objective=objective+.99**horizon*survival*boot
    values.append(boot.detach())
    targets=gae(torch.stack(rewards),torch.stack(values),torch.stack(conts)).detach()
    return -objective.mean()/10,torch.stack(features),targets,torch.stack(weights)


def policy_update(actor,wm,critic,ao,co,row,refs,c,generator,max_duration):
    wm.requires_grad_(False);critic.requires_grad_(False)
    loss,features,targets,weights=imagined(actor,wm,critic,row,refs,c,generator,max_duration)
    ao.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
    if not float(grad)>0:raise RuntimeError('Actor has no differentiable WM gradient')
    assert all(p.grad is None for p in wm.parameters()), 'WM parameters received actor gradients'
    ao.step();ao.zero_grad(set_to_none=True);wm.requires_grad_(True);critic.requires_grad_(True)
    x=features.flatten(0,1);y=targets.flatten();w=weights.flatten();closs=0.;cgrad=0.
    for _ in range(4):
        ix=torch.randint(len(y),(1024,),device='cuda',generator=generator)
        pred=critic(x[ix]);cl=(((pred-y[ix])/10).square()*w[ix]).sum()/w[ix].sum().clamp_min(1e-8)
        co.zero_grad(set_to_none=True);cl.backward();cg=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True)
        co.step();co.zero_grad(set_to_none=True);closs+=float(cl)/4;cgrad+=float(cg)/4
    return {'actor_loss':float(loss),'actor_grad':float(grad),'critic_loss':closs,'critic_grad':cgrad,
            'imagined_effective_steps':float(weights.sum(0).mean()),'imagined_weight_step32':float(weights[-1].mean())}


def run(args):
    began=time.monotonic();out=OUT/('preflight/attempt_'+args.attempt if args.preflight else 'smoke')/('group_'+args.group)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists() or (out/'failure.json').exists():raise RuntimeError('Attempt closed; do not overwrite results')
    ids=json.loads((ROOT/'experiments/wm4_200k/nested_groups.json').read_text())['groups'][args.group]
    env,cfg,cap,physics,c=build_env(ids,120,10401)
    OmegaConf.save(cfg,out/'environment.yaml');save_json(out/'contract.json',c)
    refs=References(env,len(ids));motion=torch.arange(120,device='cuda')%len(ids)
    saved=torch.get_rng_state();savedcuda=torch.cuda.get_rng_state_all();torch.manual_seed(45);torch.cuda.manual_seed_all(45)
    actor=Actor().cuda();wm=WorldModel().cuda();critic=Critic().cuda()
    initial={k:digest(v) for k,v in [('actor',actor),('wm',wm),('critic',critic)]}
    torch.set_rng_state(saved);torch.cuda.set_rng_state_all(savedcuda)
    ao=torch.optim.Adam(actor.parameters(),lr=2e-4,betas=(.9,.95));wo=torch.optim.Adam(wm.parameters(),lr=2e-4);co=torch.optim.Adam(critic.parameters(),lr=2e-4)
    exploration=torch.Generator(device='cuda').manual_seed(10402);replay_rng=torch.Generator(device='cuda').manual_seed(10403)
    imagine_rng=torch.Generator(device='cuda').manual_seed(10404)
    buf=Replay(120,capacity=1000);episode=torch.arange(120,device='cuda');version=0;maxpd=0.;maxobs=0.
    collect_seconds=0.;update_seconds=0.;peak=0.;logs=[];cycles=1 if args.preflight else 28
    returns=torch.zeros(120,device='cuda');lengths=torch.zeros(120,dtype=torch.long,device='cuda');completed=[]
    updates={'actor':0,'wm':0,'critic':0};audit={};wm_hash_after_fit=None
    def checkpoint(name):
        atomic_save({'actor':actor.state_dict(),'wm':wm.state_dict(),'critic':critic.state_dict(),
          'actor_optimizer':ao.state_dict(),'wm_optimizer':wo.state_dict(),'critic_optimizer':co.state_dict(),
          'updates':updates.copy(),'raw_transitions':buf.total*120,'initial_hashes':initial,'ids':ids,'contract':c,
          'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate(),
                 'exploration':exploration.get_state(),'replay':replay_rng.get_state(),'imagination':imagine_rng.get_state()},
          'normalization':'fixed engineering scales; no test-data fitting','physical_resume':'restart legal frame0; hidden PhysX state not serialized'},out/name)
    def guard():
        nonlocal peak
        free,total=torch.cuda.mem_get_info();peak=max(peak,(total-free)/2**20)
        if time.monotonic()-began>args.seconds:raise RuntimeError('Per-job wall-time cap')
        if total-free>24*2**30:raise RuntimeError('GPU memory cap 24 GiB')
        if shutil.disk_usage(OUT).free<20*2**30:raise RuntimeError('Disk reserve below 20 GiB')
    def collect(steps):
        nonlocal maxpd,maxobs,collect_seconds
        t0=time.monotonic()
        for _ in range(steps):
            guard()
            # Explicit observation timing: current frame+1 goal in real and imagined decisions.
            env.compute_observations()
            with torch.no_grad():
                s=physical_state(env);obs=env.obs_dict['actor'].clone();mu=actor(obs)
                noise=torch.randn(mu.shape,device='cuda',generator=exploration)*.12;action=(mu+noise).clamp(-1,1)
                prev=env.prev_targets.clone();last=env.last_actions.clone();mp=env.mask_previous_command.clone()
                hist=env.tip_contact_history.flatten(1).float().clone();f=env.progress_buf.clone();age=env.running_progress_buf[:,None].float().clone()
                reset=env.just_reset.clone();ph=physics.clone();remaining=env.mask_remaining.clone();joints=env.mask_joints.clone()
                err_obs=refs.parity(env,s,motion);maxobs=max(maxobs,err_obs)
                assert err_obs<3e-5,'Differentiable actor observation mismatch'
                _,reward,done,_=env.step(action);sn=physical_state(env);reward=reward.clone();done=done.clone().bool()
                expected=control(s,action,prev,last,mp,cap['delay'],cap['mask'],c)
                err=float((expected-env.pos_targets).abs().max());maxpd=max(maxpd,err)
                assert err<2e-6 and torch.equal(env.progress_buf,f+1),'Control or label alignment failure'
                assert torch.isfinite(sn).all() and torch.isfinite(reward).all(),'Nonfinite real labels'
                assert torch.equal(advance_history(hist,sn),env.tip_contact_history.flatten(1).float()),'Force-history semantics mismatch'
                row={'s':s,'sn':sn,'previous':prev,'last':last,'mask_previous':mp,'history':hist,'frame':f,'age':age,
                     'physics':ph,'ref':refs.get(motion,f),'motion':motion,'episode':episode.clone(),
                     'mask_remaining':remaining,'mask_joints':joints,'action':action,'actor_mean':mu,'exploration':noise,
                     'policy_version':torch.full((120,),version,device='cuda',dtype=torch.long),'obs':obs,
                     'target':env.pos_targets.clone(),'delay':cap['delay'].clone(),'mask':cap['mask'].clone(),
                     'reward':reward,'done':done,'failure':env.failure_buf.clone().bool(),'success':env.success_buf.clone().bool(),
                     'selected':~reset,'done_reasons':env.done_reason_bits.clone()}
                assert torch.equal(action,(row['actor_mean']+row['exploration']).clamp(-1,1))
                buf.add(row)
                returns.add_(reward);lengths.add_(1)
                for lane in done.nonzero().flatten().tolist():
                    completed.append({'motion':ids[int(motion[lane])],'return':float(returns[lane]),'length':int(lengths[lane]),
                                      'success':bool(env.success_buf[lane]),'reason_bits':int(env.done_reason_bits[lane]),'policy_version':version})
                returns[done]=0;lengths[done]=0
                _,idx=env.reset_done();assert len(idx)==int(done.sum());episode[idx]+=120
        torch.cuda.synchronize();collect_seconds+=time.monotonic()-t0
    try:
        checkpoint('initial.pt');collect(32)
        # Terminal bootstrap and explicit history tests are not implementation-mirroring smoke assertions.
        rr=torch.tensor([[2.]],device='cuda');vv=torch.tensor([[3.],[100.]],device='cuda')
        assert float(gae(rr,vv,torch.zeros_like(rr)))==2.
        assert abs(float(gae(rr,vv,torch.ones_like(rr)))-101.)<1e-5
        for cycle in range(cycles):
            guard();t0=time.monotonic();wmstats=[]
            for _ in range(8):wmstats.append(wm_update(wm,wo,buf.sample(128,replay_rng,8),c));updates['wm']+=1
            row=buf.sample(128,replay_rng)[0]
            if cycle==0:
                wm_hash_after_fit=digest(wm)
                # Same initial state/RNG, poison logged executed future-draw fields: imaginary objective must be identical.
                gstate=imagine_rng.get_state()
                with torch.no_grad():v1=imagined(actor,wm,critic,row,refs,c,imagine_rng,env._mask_max_duration(),4)[0]
                poison=dict(row);poison['delay']=~row['delay'];poison['mask']=~row['mask'];poison['sn']=row['sn']*100
                imagine_rng.set_state(gstate)
                with torch.no_grad():v2=imagined(actor,wm,critic,poison,refs,c,imagine_rng,env._mask_max_duration(),4)[0]
                assert torch.equal(v1,v2),'Future-label/control leakage into Actor rollout'
                imagine_rng.set_state(gstate);audit['future_draw_poison_test']='passed'
            ps=policy_update(actor,wm,critic,ao,co,row,refs,c,imagine_rng,env._mask_max_duration())
            updates['actor']+=1;updates['critic']+=4;version+=1
            if cycle==0:assert digest(wm)==wm_hash_after_fit,'Actor changed frozen WM parameters'
            torch.cuda.synchronize();update_seconds+=time.monotonic()-t0
            collect(8)
            log={'cycle':cycle+1,'raw_transitions':buf.total*120,'seconds':time.monotonic()-began,
                 'wm_loss':sum(x['loss'] for x in wmstats)/8,'wm_grad':wmstats[-1]['grad'],**ps,
                 'actor_hash':digest(actor),'wm_hash':digest(wm),'critic_hash':digest(critic)}
            logs.append(log)
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(json.dumps(log),flush=True)
            if (cycle+1)%7==0:checkpoint('latest.pt')
        final={k:digest(v) for k,v in [('actor',actor),('wm',wm),('critic',critic)]}
        assert all(final[k]!=initial[k] for k in initial),'A component did not update'
        d={k:v[:buf.total].cpu() for k,v in buf.data.items()};selected=d['selected'];phase=d['frame'].float()/refs.length.cpu()[d['motion']]
        prevcontact=d['s'][...,287:292].bool().any(-1);contact=d['sn'][...,287:292].bool().any(-1)
        result={'group':args.group,'ids':ids,'status':'short_loop_complete','raw_transitions':buf.total*120,'usable_transitions':int(selected.sum()),
          'updates':updates,'initial_hashes':initial,'final_hashes':final,'control_error_max_rad':maxpd,'actor_obs_error_max':maxobs,
          'seconds':time.monotonic()-began,'collection_seconds':collect_seconds,'update_seconds':update_seconds,'gpu_peak_mib':peak,
          'phase_quartile_counts':torch.bincount((phase[selected]*4).long().clamp(0,3),minlength=4).tolist(),
          'contact_transitions':int((selected&contact).sum()),'sustained_contact_transitions':int((selected&contact&prevcontact).sum()),
          'independent_contact_onsets':int((selected&contact&~prevcontact).sum()),
          'policy_version_counts':torch.bincount(d['policy_version'].flatten()).tolist(),
          'per_motion_raw':{x:int((d['motion']==i).sum()) for i,x in enumerate(ids)},
          'episode_count':len(completed),'successes':sum(x['success'] for x in completed),
          'mean_episode_length':sum(x['length'] for x in completed)/max(1,len(completed)),
          'mean_episode_return':sum(x['return'] for x in completed)/max(1,len(completed)),
          'audit':audit,'random_model_seed':45,'environment_seed':10401,'no_expert_no_old_weights':True,
          'wm_effective_supervision':'8-step truncated at observed Done; terminal transition included; reset-first excluded',
          'observations_refreshed_before_action':True}
        atomic_save(d,out/'replay.pt');checkpoint('final.pt');save_json(out/'episodes.json',completed);save_json(out/'result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as e:
        checkpoint('failure.pt');save_json(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'updates':updates,
                  'raw_transitions':buf.total*120,'seconds':time.monotonic()-began})
        raise
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['3','12','20','30'],required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--attempt',default='1');p.add_argument('--seconds',type=int,default=300);run(p.parse_args())

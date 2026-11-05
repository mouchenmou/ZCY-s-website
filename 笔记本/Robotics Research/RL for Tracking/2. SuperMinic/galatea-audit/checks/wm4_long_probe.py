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
    env.record_done_reasons=True
    expert=None
    if a.policy=="expert":
        from wm4_coverage_collect import Expert
        expert=Expert(env,cfg)
        c=dict(c);c['collector']='frozen official power_drill Expert mean; baseline action replay for paired diagnostic'
    # Preserve the original reset distribution for exact 32-step prefixes; longer suffixes are censored at Done.
    def sample(self,idx):
        valid=self._demo_value('reset_valid',idx).bool()
        if a.policy=='expert':
            assert valid[:,0].all()
            return torch.zeros_like(idx)
        f=torch.arange(valid.shape[1],device='cuda')[None,:]
        allowed=valid&(f<(self.demo_seq_len[idx,None]-48))
        if not (allowed.sum(-1)>0).all():raise RuntimeError('No valid probe reset frames')
        rank=(torch.rand(len(idx),device='cuda')*allowed.sum(-1)).long()
        return ((allowed.long().cumsum(-1)>rank[:,None])&allowed).long().argmax(-1)
    env._sample_reset_frames=types.MethodType(sample,env);env.reset()
    noise=torch.zeros(n,29,device='cuda');warm=[]
    for step in range(a.warmup):
        if expert is None:act,noise=scripted(env,noise)
        else:act=expert.action(env)
        env.step(act);warm.append(state(env).detach().cpu())
    initial_done=env.reset_buf.clone().bool().cpu()
    base=torch.load(a.base,weights_only=False) if a.base else None
    rows={};maxerr=0.
    for h in range(a.horizon):
        # No reset: prevent out-of-bank indexing only for already terminal lanes.
        dead=env.reset_buf.bool()
        env.progress_buf[dead]=torch.minimum(env.progress_buf[dead],env.demo_seq_len[dead]-5)
        guard(start,300)
        frame=env.progress_buf.clone();s=state(env);prev=env.prev_targets.clone();last=env.last_actions.clone()
        mp=env.mask_previous_command.clone();hist=env.tip_contact_history.flatten(1).float().clone()
        ref=reference(env);age=env.running_progress_buf[:,None].float().clone()
        if expert is None:act,noise=scripted(env,noise)
        else:act=expert.action(env)
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
             'frame':frame,'seq_len':env.demo_seq_len.clone(),'done_reasons':env.done_reason_bits.clone(),
             'done_metrics':env.done_metrics.clone(),'delay':cap['delay'].clone(),'mask':cap['mask'].clone(),'reward':r.clone(),'done':done.clone().bool()}
        for k,v in row.items():rows.setdefault(k,[]).append(v.detach().cpu())
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    data={k:torch.stack(v) for k,v in rows.items()}
    data.update({'warmup':torch.stack(warm),'initial_done':initial_done,
                 'contract':c,'ids':a.ids.split(','),'seed':a.seed,'branch':a.branch,'horizon':a.horizon,'done_reason_names':env.done_reason_schema()[0],'policy':a.policy,'warmup_steps':a.warmup})
    torch.save(data,str(out)+'.tmp');os.replace(str(out)+'.tmp',out)
    meta={'max_pd_error':maxerr,'seconds':time.monotonic()-start,
          'task_valid_cases':{str(h):int((~(data['initial_done']|data['done'][:h].any(0))).sum()) for h in (1,8,32,64,128,a.horizon)}}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=9091);p.add_argument('--branch',type=int,default=0)
    p.add_argument('--envs',type=int,default=120)
    p.add_argument('--horizon',type=int,default=256)
    p.add_argument('--policy',choices=['script','expert'],default='script')
    p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--base');run(p.parse_args())

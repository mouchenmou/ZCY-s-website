"""Pilot collector repair: frozen official Expert and safe frame-0 starts.

No policy optimization, environment reward/done changes, reset teleport to grasps,
or reference-state substitution. Expert provenance precludes strict unseen-policy claims.
"""
import argparse,json,os,time,types
from pathlib import Path
from wm4_collect import make,state,reference,scripted,guard,ROOT
import torch
from omegaconf import OmegaConf
from wm4_model import control

class Expert:
    def __init__(self,env,cfg):
        from armhand.rl.network import AsymmetricMLPBuilder
        from armhand.rl.model import ContinuousPolicyModel
        params=OmegaConf.to_container(cfg.train.params,resolve=True)
        from gym import spaces
        import numpy as np
        shapes=spaces.Dict({**{k:spaces.Box(-np.inf,np.inf,shape=tuple(v.shape[1:]),dtype=np.float32) for k,v in env.obs_dict.items()},'block_coef':spaces.Box(-np.inf,np.inf,shape=(1,),dtype=np.float32)})
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        builder=AsymmetricMLPBuilder();builder.load(params['network'])
        self.model=ContinuousPolicyModel(builder).build({'actions_num':29,'input_shape':shapes,
            'num_seqs':env.num_envs,'value_size':1,'normalize_input_excluded_keys':['block_coef'],
            'block_config':params['config']['blocks']}).cuda().eval()
        ck=torch.load(ROOT/'downloads/power_drill.pth',map_location='cuda',weights_only=False)
        self.model.load_state_dict(ck['model'],strict=True);self.model.requires_grad_(False)
        # Loading fixed weights must not change reset / physics / exploration RNG.
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
    @torch.no_grad()
    def action(self,env):
        obs={k:v.clone() for k,v in env.obs_dict.items()};obs['block_coef']=torch.zeros(env.num_envs,1,device='cuda')
        # Official deterministic mean, without Normal.sample() consuming random draws.
        obs=self.model.norm_obs(obs)
        mu,_,_,_=self.model.a2c_network({'obs':obs})
        return mu.clamp(-1,1)

def run(a):
    start=time.monotonic();env,cfg,cap,phys,c=make(a.ids.split(','),a.envs,a.seed)
    env.record_done_reasons=True
    if a.strict_hand_gate:env.hand_velocity_gate_unlocked=True
    expert=Expert(env,cfg) if a.mode!='script' else None
    c=dict(c)
    if expert is not None:c['collector']='frozen official power_drill Expert mean; optional recorded exploration; no policy training'
    if a.mode=='expert_early':
        def sample(self,idx):
            assert self._demo_value('reset_valid',idx,torch.zeros_like(idx)).all()
            return torch.zeros_like(idx)
        env._sample_reset_frames=types.MethodType(sample,env)
        env.reset()
    out=Path(a.output);OmegaConf.save(cfg,out.with_suffix('.yaml'))
    n=env.num_envs;noise=torch.zeros(n,29,device='cuda');episodes=torch.arange(n,device='cuda')
    records={};maxerr=0.;peak=0
    for step in range(a.steps):
        if step%32==0:
            guard(start,280);free,total=torch.cuda.mem_get_info();peak=max(peak,(total-free)/2**20)
        s=state(env);previous=env.prev_targets.clone();last=env.last_actions.clone();mp=env.mask_previous_command.clone()
        hist=env.tip_contact_history.flatten(1).float().clone();frame=env.progress_buf.clone();age=env.running_progress_buf[:,None].float().clone()
        ref=reference(env);physics=phys.clone();reset=env.just_reset.clone();ep=episodes.clone()
        if expert is None:action,noise=scripted(env,noise)
        else:
            action=expert.action(env)
            if a.exploration>0:
                noise=.9*noise+(1-.9**2)**.5*torch.randn_like(noise)
                scale=torch.full((29,),a.exploration,device="cuda");scale[:7]*=.2
                action=(action+noise*scale).clamp(-1,1)
        _,r,done,_=env.step(action);sn=state(env);r=r.clone();done=done.clone().bool()
        err=float((control(s,action,previous,last,mp,cap['delay'],cap['mask'],c)-env.pos_targets).abs().max());maxerr=max(maxerr,err)
        assert err<2e-6 and torch.equal(env.progress_buf,frame+1) and torch.isfinite(sn).all()
        row={'s':s,'sn':sn,'action':action,'target':env.pos_targets.clone(),'previous':previous,'last':last,
             'mask_previous':mp,'history':hist,'ref':ref,'physics':physics,'age':age,'reward':r,'done':done,
             'delay':cap['delay'],'mask':cap['mask'],'episode':ep,'frame':frame,'selected':~reset,
             'reset_input':reset,'delayed':cap['delayed'],'done_reasons':env.done_reason_bits.clone(),
             'done_metrics':env.done_metrics.clone(),'success':env.success_buf.clone()}
        for k,v in row.items():records.setdefault(k,[]).append(v.detach().cpu())
        _,reset_ids=env.reset_done();assert len(reset_ids)==int(done.sum())
        episodes[reset_ids]+=n;noise[reset_ids]=0
        if step%200==0:print(json.dumps({'step':step,'seconds':time.monotonic()-start}),flush=True)
    data={k:torch.stack(v) for k,v in records.items()}
    data.update({'ids':a.ids.split(','),'contract':c,'seed':a.seed,'mode':a.mode,'seq_len':env.demo_seq_len.cpu(),
                 'done_reason_names':env.done_reason_schema()[0],'done_metric_names':env.done_reason_schema()[1]})
    torch.save(data,str(out)+'.tmp');os.replace(str(out)+'.tmp',out)
    meta={'seconds':time.monotonic()-start,'steps':a.steps,'raw_transitions':a.steps*n,'selected':int(data['selected'].sum()),
          'max_pd_error_rad':maxerr,'gpu_peak_mib':peak,'mode':a.mode,'hand_velocity_gate_unlocked':env.hand_velocity_gate_unlocked,
          'exploration_std':a.exploration,'action_mask_step':env.action_mask_step,'expert_normalization_frozen':True if expert else None,
          'reward_termination_modified':False,'official_hand_gate_explicitly_enabled':a.strict_hand_gate,'training_updates':0}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
    env.gym.destroy_sim(env.sim)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--mode',choices=['script','expert','expert_early'],required=True)
    p.add_argument('--envs',type=int,default=60);p.add_argument('--steps',type=int,default=1400);p.add_argument('--seed',type=int,default=9401)
    p.add_argument('--strict-hand-gate',action='store_true')
    p.add_argument('--exploration',type=float,default=0.)
    p.add_argument('--output',required=True);run(p.parse_args())

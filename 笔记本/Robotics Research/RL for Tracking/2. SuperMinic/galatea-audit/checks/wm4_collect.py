"""Real PhysX collection with audited control and pre-reset labels.

All environment changes are wrappers for collection/indexing; official code is
unmodified. The scripted exploration controller has no learned parameters.
"""
import os, sys, json, time, types, argparse, random, hashlib, shutil
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from wm4_model import control, SDIM

ROOT=Path(__file__).resolve().parents[1]
RL=ROOT/'sources/GALATEA/rl'
EXP=ROOT/'experiments/wm4_200k'
sys.path.insert(0,str(RL))
from armhand.envs import make_env
from armhand.envs.robot import PAD_BODIES, HAND_BODIES, FINGERTIP_BODIES


def make(ids,n,seed):
    os.chdir(RL)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    with initialize_config_dir(version_base='1.1',config_dir=str(RL/'cfg')):
        cfg=compose(config_name='config',overrides=[f'num_envs={n}',f'seed={seed}',
            'object.data_indices=['+','.join(ids)+']',
            'task.env.gravityInitialScale=1.0','task.env.tightenSchedule=const'])
    env=make_env(sim_device='cuda:0',rl_device='cuda:0',multi_gpu=False,cfg=cfg.task)
    # Each lane keeps the geometry/reference it was constructed with; quotas are balanced.
    fixed=torch.arange(n,device='cuda')%len(ids)
    env._sample_demo_ids=types.MethodType(lambda self,idx: fixed[idx],env)
    assert env.arm_dofs.tolist()==list(range(7)) and env.hand_dofs.tolist()==list(range(7,29))
    capture={}
    original_sample=env.randomizer.sample
    physics=torch.zeros(n,63,device='cuda')
    def sample(rows):
        draw=original_sample(rows)
        capture['draw']=draw
        return draw
    original_resample=env.randomizer.resample
    env.randomizer.sample=sample
    def resample(idx):
        original_resample(idx)
        if not len(idx):return
        d=capture['draw']
        values=np.concatenate([d[k].reshape(len(idx),-1) for k in
             ('arm_kp','hand_kp','arm_kd','hand_kd','surface','object_friction','table_friction')],axis=1)
        physics[idx,:62]=torch.from_numpy(values).to('cuda')
    env.randomizer.resample=resample
    for i in range(n):
        physics[i,-1]=env.gym.get_actor_rigid_body_properties(env.envs[i],env.object_actors[i])[0].mass
    delay_orig=env._delay_actions;mask_orig=env._mask_hand_targets
    def delay(self,a):
        old=self.action_history[self.action_history_ptr].clone()
        reset=self.action_history_reset.clone()
        randint=torch.randint;draws=[]
        def capture_randint(*args,**kwargs):
            value=randint(*args,**kwargs);draws.append(value.clone());return value
        torch.randint=capture_randint
        try:out=delay_orig(a)
        finally:torch.randint=randint
        cur=(out-a).abs().amax(-1)<1e-7
        prev=(out-old).abs().amax(-1)<1e-7
        assert (cur|prev).all()
        capture['delay']=(draws[-1].bool() & ~reset).clone()
        capture['delayed']=out.clone()
        return out
    def mask(self,t):
        remaining=self.mask_remaining.clone();rand=torch.rand;draws=[]
        def capture_rand(*args,**kwargs):
            value=rand(*args,**kwargs);draws.append(value.clone());return value
        torch.rand=capture_rand
        try:out=mask_orig(t)
        finally:torch.rand=rand
        active=(remaining>0)|((remaining<=0)&(draws[0]<self.action_mask_probability))
        capture['mask']=(active[:,None]&self.mask_joints).clone()
        capture['mask_input']=t.clone()
        return out
    env._delay_actions=types.MethodType(delay,env)
    env._mask_hand_targets=types.MethodType(mask,env)
    env.reset()
    contract={k:float(getattr(env,k)) for k in ('arm_action_scale','arm_smoothing','hand_smoothing','hand_target_deadband')}
    for k in ('arm_max_step','arm_lower','arm_upper','hand_lower','hand_upper'):
        contract[k]=getattr(env,k).tolist()
    contract.update({'hand_dof_indices':env.randomizer.hand_slots.tolist(),
        'tip_body_indices':[HAND_BODIES.index(x) for x in FINGERTIP_BODIES],
        'state_dim':SDIM,'control_dt':1/30,'reward_frame':'pre-increment progress; reward after physics',
        'collector':'untrained script: arm reference tracking plus correlated hand exploration'})
    return env,cfg,capture,physics,contract


def state(env):
    force=torch.stack([env._net_contact_force[:,env.body_handles[k]] for k in PAD_BODIES],1)
    contact,_=env._pad_contacts(force)
    body=env.states['body_state']
    return torch.cat((env._q,env._qd,env._object_state,body[:,:,:3].flatten(1),
        body[:,:,7:10].flatten(1),env.states['wrist_state'][:,3:7],force.flatten(1),
        env._dof_force,contact.float()),-1).clone()


def reference(env):
    idx=torch.arange(env.num_envs,device='cuda')
    frame=env.progress_buf.clamp_max(env.demo_bank['object_pose'].shape[1]-1)
    lift,vel=env._floor_lift_at(idx,frame)
    values=[]
    for key in ('wrist_pos','wrist_rot','wrist_vel','wrist_ang_vel','joints_pos','joints_vel',
                'object_pose','object_vel','object_ang_vel','tips_distance','object_pos_tolerance'):
        a=env._demo_value(key,idx,frame)
        if key=='object_pose':a=a[:,:3,:]
        values.append(a.reshape(env.num_envs,-1))
    values.extend((lift[:,None],vel[:,None]))
    return torch.cat(values,-1).clone()


def scripted(env,noise):
    n=env.num_envs;idx=torch.arange(n,device='cuda')
    f=torch.minimum(env.progress_buf+1,env.demo_seq_len.long()-1)
    arm_ref=env._demo_value('arm_reset_qpos',idx,f)
    arm=((arm_ref-env._q[:,:7])/env.arm_action_scale).clamp(-.8,.8)
    # Same stationary exploration distribution, independent of motion identity.
    noise=.9*noise+(1-.9**2)**.5*torch.randn_like(noise)
    open_action=(2*(torch.zeros_like(env.hand_lower)-env.hand_lower)/(env.hand_upper-env.hand_lower)-1)
    hand=(open_action[None,:]*.45 + noise[:,7:]*.45).clamp(-1,1)
    return torch.cat(((arm+noise[:,:7]*.12).clamp(-1,1),hand),-1),noise


def guard(start,budget=900):
    if time.monotonic()-start>budget:raise RuntimeError('Collection wall time budget exceeded')
    if shutil.disk_usage(EXP).free<20*2**30:raise RuntimeError('Disk reserve below 20 GiB')
    free,total=torch.cuda.mem_get_info()
    if total-free>40*2**30:raise RuntimeError('Total GPU use exceeds 40 GiB')


def collect(args):
    ids=args.ids.split(',');start=time.monotonic()
    env,cfg,cap,phys,c=make(ids,args.envs,args.seed)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,out.with_suffix('.yaml'))
    n=env.num_envs;noise=torch.zeros(n,29,device='cuda')
    episodes=torch.arange(n,device='cuda');counts=np.zeros(len(ids),dtype=np.int64)
    quotas=np.full(len(ids),args.count//len(ids),dtype=np.int64);quotas[:args.count%len(ids)]+=1
    motion=(torch.arange(n)%len(ids)).numpy()
    records={};maxerr=0.;steps=0;warmup=0;done_count=0;gpu_peak=0
    # Same process seed, fixed lane assignment, no pretrained normalization.
    while counts.sum()<args.count:
        if steps%32==0:
            guard(start,args.seconds)
            free,total=torch.cuda.mem_get_info();gpu_peak=max(gpu_peak,(total-free)/2**20)
        s=state(env);previous=env.prev_targets.clone();last=env.last_actions.clone()
        mp=env.mask_previous_command.clone();history=env.tip_contact_history.flatten(1).float().clone()
        frame=env.progress_buf.clone();age=env.running_progress_buf[:,None].float().clone()
        ref=reference(env);physics=phys.clone();reset=env.just_reset.clone()
        ep=episodes.clone();action,noise=scripted(env,noise)
        _,r,done,_=env.step(action)
        sn=state(env);r=r.clone();done=done.clone().bool()
        expected=control(s,action,previous,last,mp,cap['delay'],cap['mask'],c)
        err=float((expected-env.pos_targets).abs().max());maxerr=max(maxerr,err)
        if err>2e-6:raise RuntimeError('PD target contract mismatch')
        if not torch.equal(env.progress_buf,frame+1):raise RuntimeError('Incorrect frame alignment')
        if not torch.isfinite(sn).all() or not torch.isfinite(r).all():raise RuntimeError('Nonfinite PhysX labels')
        keep=np.zeros(n,dtype=bool);eligible=(~reset).cpu().numpy()
        for m in range(len(ids)):
            lanes=np.flatnonzero((motion==m)&eligible)[:max(0,int(quotas[m]-counts[m]))]
            keep[lanes]=True;counts[m]+=len(lanes)
        row={'s':s,'sn':sn,'action':action,'target':env.pos_targets.clone(),'previous':previous,
             'last':last,'mask_previous':mp,'history':history,'ref':ref,'physics':physics,'age':age,
             'reward':r,'done':done,'delay':cap['delay'],'mask':cap['mask'],
             'episode':ep,'frame':frame,'selected':torch.from_numpy(keep),
             'reset_input':reset,'delayed':cap['delayed']}
        for k,v in row.items():records.setdefault(k,[]).append(v.detach().cpu())
        warmup+=int(reset.sum());done_count+=int(done.sum());steps+=1
        _,reset_ids=env.reset_done()
        if len(reset_ids)!=int(done.sum()):raise RuntimeError('Reset boundary mismatch')
        episodes[reset_ids]+=n;noise[reset_ids]=0
        if steps%128==0:print(json.dumps({'event':'collection','selected':int(counts.sum()),'steps':steps,'seconds':time.monotonic()-start}),flush=True)
    data={k:torch.stack(v) for k,v in records.items()}
    data.update({'contract':c,'ids':ids,'seed':args.seed})
    assert int(data['selected'].sum())==args.count
    selected=data['selected'];cont=data['sn'][...,287:].bool().any(-1)
    prev=data['s'][...,287:].bool().any(-1)
    episodes_contact=torch.unique(data['episode'][selected&cont]).numel()
    phases={'no_contact':int((selected&~prev&~cont).sum()),'first_or_recontact':int((selected&~prev&cont).sum()),
            'sustained':int((selected&prev&cont).sum()),'release':int((selected&prev&~cont).sum())}
    fractions=data['frame'].float()/env.demo_bank['seq_len'].cpu()[torch.from_numpy(motion)][None,:]
    phase_hist=torch.bincount((fractions[selected]*4).long().clamp(0,3),minlength=4).tolist()
    act=data['action'][selected];cov=torch.cov(act.T)
    eig=torch.linalg.eigvalsh(cov).clamp_min(0);p=eig/eig.sum().clamp_min(1e-12)
    effective_rank=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum()))
    # Strictly contiguous 8/32 windows, never across reset or outside selected quota.
    windows={}
    for h in (8,32):
        ok=selected.clone()
        for j in range(1,h):
            ok[:-j]&=selected[j:]&(data['episode'][:-j]==data['episode'][j:])
            ok[-j:]=False
        windows[str(h)]=int(ok.sum())
    meta={'motion_ids':ids,'transitions':args.count,'per_motion':dict(zip(ids,counts.tolist())),
          'raw_simulated_transitions':steps*n,'excluded_reset_first_transitions':warmup,'done_count':done_count,
          'seconds':time.monotonic()-start,'selected_transitions_per_second':args.count/(time.monotonic()-start),
          'max_pd_error':maxerr,'contact_episodes':episodes_contact,'contact_phases':phases,'valid_windows':windows,
          'reference_phase_quartile_counts':phase_hist,'action_covariance_effective_rank':effective_rank,
          'action_std':data['action'][selected].std(0).tolist(),'pd_target_std':data['target'][selected].std(0).tolist(),
          'action_saturation_fraction':float((data['action'][selected].abs()>.99).float().mean()),
          'torch_peak_mib':torch.cuda.max_memory_allocated()/2**20,'gpu_total_peak_sampled_mib':gpu_peak}
    torch.save(data,str(out)+'.tmp');os.replace(str(out)+'.tmp',out)
    meta['sha256']=hashlib.sha256(out.read_bytes()).hexdigest();meta['bytes']=out.stat().st_size
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True);env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=200000);p.add_argument('--envs',type=int,default=120)
    p.add_argument('--seed',type=int,default=20261105);p.add_argument('--seconds',type=int,default=900)
    collect(p.parse_args())

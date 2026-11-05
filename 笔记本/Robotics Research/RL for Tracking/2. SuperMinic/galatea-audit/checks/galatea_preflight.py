"""Short, training-candidate-only audit of the unmodified GALATEA environment."""
import os
import sys
import json
import time
import types
import threading
import subprocess
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
RL=ROOT/'sources/GALATEA/rl'
sys.path.insert(0,str(RL));os.chdir(RL)
from armhand.envs import make_env
from armhand.envs.robot import ARM_JOINTS,PAD_BODIES
torch.manual_seed(20261104);np.random.seed(20261104)
torch.set_num_threads(4)
samples=[];stop=threading.Event()
def monitor():
    while not stop.is_set():
        try:
            val=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
            samples.append(float(val.strip().splitlines()[0]))
        except (subprocess.SubprocessError,ValueError):pass
        stop.wait(.2)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
with initialize_config_dir(version_base='1.1',config_dir=str(RL/'cfg')):
    cfg=compose(config_name='config',overrides=['num_envs=16','object.data_indices=[g16]',
        'task.env.gravityInitialScale=1.0','task.env.tightenSchedule=const'])
(ROOT/'evidence/preflight_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
start=time.perf_counter()
env=make_env(sim_device='cuda:0',rl_device='cuda:0',multi_gpu=False,cfg=cfg.task)
init_seconds=time.perf_counter()-start
log={};delay_original=env._delay_actions;mask_original=env._mask_hand_targets
def delay(self,actions):
    out=delay_original(actions);log['delayed']=out.clone();return out
def mask(self,targets):
    log['mask_input']=targets.clone();log['mask_previous']=self.mask_previous_command.clone()
    out=mask_original(targets);log['mask_output']=out.clone();return out
env._delay_actions=types.MethodType(delay,env)
env._mask_hand_targets=types.MethodType(mask,env)
env.reset()
counts={'transitions':0,'done':0,'actual_pad_contact_frames':0,'masked_target_entries':0,'delayed_rows':0}
max_error=0.;peak_torch=0;torch.cuda.reset_peak_memory_stats()
episode_ids=torch.arange(env.num_envs,device='cuda',dtype=torch.long)
start=time.perf_counter()
for step in range(128):
    q=env._q.clone();prev=env.prev_targets.clone();old_progress=env.progress_buf.clone()
    previous_command=env.last_actions.clone()
    actions=torch.rand(env.num_envs,env.num_actions,device='cuda')*.6-.3
    if step%19==0:actions[:,0]=1.5  # Verify public step clamps the submitted command.
    command=actions.clamp(-env.clip_actions,env.clip_actions)
    obs,reward,done,extra=env.step(actions)
    # Save terminal labels BEFORE reset_done; returned tensors alias environment buffers.
    transition_reward=reward.clone();transition_done=done.clone()
    next_q=env._q.clone();next_qd=env._qd.clone();next_object=env._object_state.clone()
    assert torch.equal(env.last_actions,command)
    assert torch.equal(env.progress_buf,old_progress+1)
    assert torch.isfinite(next_q).all() and torch.isfinite(next_qd).all()
    assert torch.isfinite(next_object).all() and torch.isfinite(transition_reward).all()
    delayed=log['delayed']
    same_current=torch.isclose(delayed,command).all(1)
    same_old=torch.isclose(delayed,previous_command).all(1)
    assert (same_current|same_old).all()
    counts['delayed_rows']+=int((same_old&~same_current).sum())
    a=delayed[:,:len(ARM_JOINTS)]
    arm=q[:,env.arm_dofs]+a*env.arm_action_scale
    arm=env.arm_smoothing*arm+(1-env.arm_smoothing)*prev[:,env.arm_dofs]
    arm=prev[:,env.arm_dofs]+(arm-prev[:,env.arm_dofs]).clamp(-env.arm_max_step,env.arm_max_step)
    arm=arm.clamp(env.arm_lower,env.arm_upper)
    hand=.5*(delayed[:,len(ARM_JOINTS):].clamp(-1,1)+1)*(env.hand_upper-env.hand_lower)+env.hand_lower
    hand=env.hand_smoothing*hand+(1-env.hand_smoothing)*prev[:,env.hand_dofs]
    delta=hand-prev[:,env.hand_dofs]
    hand=prev[:,env.hand_dofs]+delta.sign()*(delta.abs()-env.hand_target_deadband).clamp(min=0)
    hand=hand.clamp(env.hand_lower,env.hand_upper)
    err=max(float((arm-env.pos_targets[:,env.arm_dofs]).abs().max()),float((hand-log['mask_input']).abs().max()))
    max_error=max(max_error,err);assert err<2e-6
    masked=~torch.isclose(log['mask_output'],log['mask_input'],atol=1e-7,rtol=0)
    assert torch.isclose(log['mask_output'][masked],log['mask_previous'][masked],atol=1e-7,rtol=0).all()
    assert torch.allclose(env.pos_targets[:,env.hand_dofs],log['mask_output'])
    assert torch.equal(env.prev_targets,env.pos_targets)
    counts['masked_target_entries']+=int(masked.sum())
    forces=torch.stack([env._net_contact_force[:,env.body_handles[x],:] for x in PAD_BODIES],dim=1)
    contact,_=env._pad_contacts(forces)
    counts['actual_pad_contact_frames']+=int(contact.any(1).sum())
    counts['done']+=int(transition_done.sum());counts['transitions']+=env.num_envs
    _,reset_ids=env.reset_done()
    assert len(reset_ids)==int(transition_done.sum())
    episode_ids[reset_ids]+=env.num_envs
    # Terminal tensors are unchanged by resets because they were cloned above.
    if step==15:
        torch.cuda.synchronize();start=time.perf_counter()
torch.cuda.synchronize();elapsed=time.perf_counter()-start
result={'scope':'g16 only; random-action preflight, NOT formal collection or a trained Actor',
    'initialization_seconds':init_seconds,'timed_steps':112,'num_envs':16,'timed_seconds':elapsed,
    'env_transitions_per_second':112*16/elapsed,'gpu_total_peak_mib':max(samples) if samples else None,
    'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
    'max_pd_target_recompute_error':max_error,'counts':counts,
    'passed':['CUDA PhysX and official task','MANO reference and meshes','command clipping',
        'delay outputs current or previous command','PD smoothing/deadband/limits',
        'mask output preserves previous target','frame advances once per control step',
        'finite terminal state/reward labels captured before explicit reset'],
    'not_tested':['paired same-history branch replay','WM prediction','Actor success','external backup']}
stop.set();thread.join(timeout=6)
(ROOT/'evidence/galatea_preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
env.gym.destroy_sim(env.sim)

"""Bounded GPU-PhysX test with official robot asset, independent of MANO.

This checks infrastructure only, not GALATEA task success or WM throughput.
"""
import json
import math
import runpy
import time
from pathlib import Path
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
robot = runpy.run_path(str(ROOT/'sources/GALATEA/rl/armhand/envs/robot.py'))
gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.dt = 1/60
params.substeps = 2
params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline = True
params.physx.use_gpu = True
params.physx.solver_type = 1
params.physx.num_position_iterations = 8
params.physx.num_velocity_iterations = 0
params.physx.num_threads = 4
sim = gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
assert sim is not None
plane = gymapi.PlaneParams()
plane.normal = gymapi.Vec3(0,0,1)
gym.add_ground(sim,plane)
opt = gymapi.AssetOptions()
opt.fix_base_link = True
opt.disable_gravity = True
opt.use_physx_armature = True
opt.collapse_fixed_joints = False
opt.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset = gym.load_asset(sim,str(ROOT/'sources/GALATEA/rl/assets/robot'),'fr3_sharpa_right.urdf',opt)
assert asset is not None
names = list(gym.get_asset_dof_names(asset))
props = gym.get_asset_dof_properties(asset)
for j,name in enumerate(names):
    props['driveMode'][j] = gymapi.DOF_MODE_POS
    if name in robot['ARM_JOINTS']:
        k = robot['ARM_JOINTS'].index(name)
        props['stiffness'][j] = robot['ARM_STIFFNESS'][k]
        props['damping'][j] = robot['ARM_DAMPING'][k]
        props['effort'][j] = robot['ARM_EFFORT'][k]
    else:
        for key,value in zip(('stiffness','damping','armature','friction'),robot['HAND_JOINT_DRIVE'][name]):
            props[key][j] = value
        if name in robot['HAND_JOINT_LIMITS']:
            props['lower'][j],props['upper'][j] = robot['HAND_JOINT_LIMITS'][name]
box = gym.create_box(sim,.05,.05,.05,gymapi.AssetOptions())
n_env = 16
arm0 = np.load(ROOT/'assets/power_drill/ik/g16_fr3_retarget.npz')['arm_qpos_reset'][0]
q0 = np.zeros(len(names),np.float32)
for j,name in enumerate(names):
    if name in robot['ARM_JOINTS']:q0[j]=arm0[robot['ARM_JOINTS'].index(name)]
for i in range(n_env):
    env = gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),4)
    pose = gymapi.Transform();pose.p.z=1.0
    actor = gym.create_actor(env,asset,pose,'robot',i,1)
    gym.set_actor_dof_properties(env,actor,props)
    state = np.zeros(len(names),dtype=gymapi.DofState.dtype);state['pos']=q0
    gym.set_actor_dof_states(env,actor,state,gymapi.STATE_ALL)
    pose = gymapi.Transform();pose.p= gymapi.Vec3(.8,0,.5)
    gym.create_actor(env,box,pose,'box',i,0)
gym.prepare_sim(sim)
q = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(n_env,len(names),2)
roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(n_env,2,13)
targets = torch.tensor(q0,device='cuda').repeat(n_env,1)
j = names.index('right_index_MCP_FE')
before=None
torch.cuda.reset_peak_memory_stats()
started=time.perf_counter()
for t in range(270):
    targets[:,j]=.15*math.sin(t*.04)
    gym.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(targets))
    for _ in range(2):gym.simulate(sim)
    gym.fetch_results(sim,True)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    assert q.is_cuda and torch.isfinite(q).all() and torch.isfinite(roots).all()
    if t==29:
        torch.cuda.synchronize();started=time.perf_counter();before=q[:,:,0].clone()
torch.cuda.synchronize()
elapsed=time.perf_counter()-started
result=dict(test='infrastructure_only',num_envs=n_env,measured_control_steps=240,
    control_hz=30,physics_steps=int(gym.get_frame_count(sim)),
    torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
    seconds=elapsed,env_transitions_per_second=n_env*240/elapsed,
    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    hand_motion_rad=float((q[:,j,0]-before[:,j]).abs().mean()),
    box_final_z_mean=float(roots[:,1,2].mean()),finite=True)
assert result['hand_motion_rad']>1e-4
assert .01<result['box_final_z_mean']<.08
assert result['physics_steps']==540
(ROOT/'evidence/physx_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
gym.destroy_sim(sim)

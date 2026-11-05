"""Real PhysX boundary and differentiable reconstruction of official actor obs."""
import types
from wm4_collect import make,state,ROOT,HAND_BODIES,PAD_BODIES
import torch
from armhand.envs.frames import _multiply as mul,_conjugate as conj,_rotate as rotate
from armhand.envs.frames import position_to_base as posbase,quaternion_to_base as quatbase,vector_to_base as vecbase
from armhand.data.rotations import aa_to_quat,rotmat_to_quat


def physical_state(env):
    s=state(env)
    active=(s[:,243:258].reshape(-1,5,3).norm(dim=-1)>0).float()
    return torch.cat((s,active,env._body_positions(PAD_BODIES).flatten(1)),-1)


def build_env(ids,n,seed,evaluation=False):
    env,cfg,cap,phys,c=make(ids,n,seed)
    env.record_done_reasons=True;env.hand_velocity_gate_unlocked=True
    def frame0(self,idx):
        z=torch.zeros_like(idx)
        assert self._demo_value('reset_valid',idx,z).all(), 'Illegal frame-zero reset'
        return z
    env._sample_reset_frames=types.MethodType(frame0,env)
    if evaluation:
        env.eval_mode=True
        def mask(self,t):
            self.mask_previous_command.copy_(t);cap['mask']=torch.zeros_like(t,dtype=torch.bool);return t
        env._mask_hand_targets=types.MethodType(mask,env)
    env.reset()
    c['legacy_physical_state_dim']=292
    c['state_dim']=312
    c['collector']='current random-initialized learned Actor + iid Gaussian exploration; no expert/script'
    return env,cfg,cap,phys,c


class References:
    """Only official references and object geometry, never future simulated states."""
    def __init__(self,env,m):
        self.base_pos=env.base_pos[0];self.base_quat=env.base_quat[0]
        self.shape=env.shape_code[:m].clone();self.length=env.demo_seq_len[:m].long().clone()
        ids=torch.arange(m,device='cuda');t=int(self.length.max())
        frame=torch.arange(t,device='cuda')[None,:].expand(m,-1)
        frame=torch.minimum(frame,self.length[:,None]-1)
        lift,vel=env._floor_lift_at(ids,frame);values=[]
        for key in ('wrist_pos','wrist_rot','wrist_vel','wrist_ang_vel','joints_pos','joints_vel','object_pose',
                    'object_vel','object_ang_vel','tips_distance','object_pos_tolerance'):
            a=env._demo_value(key,ids,frame)
            if key=='object_pose':a=a[...,:3,:]
            values.append(a.reshape(m,t,-1))
        self.bank=torch.cat(values+[lift[...,None],vel[...,None]],-1)
        self.pos_noise=env.object_pos_noise;self.rot_noise=float(env.object_rot_noise)*3.141592653589793/360
    def get(self,m,f):return self.bank[m,torch.minimum(f,self.length[m]-1).clamp_min(0)]
    def obs(self,s,last,m,frame,generator=None,noisy=None):
        r=self.get(m,frame+1);n=len(s);body=s[:,71:155].reshape(n,28,3);wp=body[:,0];wq=s[:,239:243]
        if noisy is None:
            op=s[:,58:61]+torch.randn(n,3,device=s.device,generator=generator)*self.pos_noise
            oq=torch.nn.functional.normalize(s[:,61:65]+torch.randn(n,4,device=s.device,generator=generator)*self.rot_noise,dim=-1)
        else:op,oq=noisy
        lift=torch.cat((torch.zeros_like(r[:,198:199]).expand(-1,2),r[:,198:199]),-1)
        twp=r[:,:3]+lift;twq=aa_to_quat(r[:,3:6])[:,[1,2,3,0]]
        tj=r[:,12:93].reshape(n,27,3)+lift[:,None]
        mat=r[:,174:186].reshape(n,3,4);top=mat[:,:,3]+lift;toq=rotmat_to_quat(mat[:,:,:3])[:,[1,2,3,0]]
        bq=self.base_quat;bp=self.base_pos
        wbase=posbase(wp,bp,bq);wqb=quatbase(wq,bq);twqb=quatbase(twq,bq);oqb=quatbase(oq,bq);toqb=quatbase(toq,bq)
        pad=rotate(conj(wq)[:,None].expand(-1,5,-1),s[:,297:312].reshape(n,5,3)-wp[:,None]).flatten(1)
        return torch.cat((s[:,:29],last,wbase,wqb,pad,rotate(conj(wq),op-wp),mul(conj(wq),oq),
                          posbase(twp,bp,bq)-wbase,twqb,mul(wqb,conj(twqb)),vecbase(tj-body[:,1:],bq).flatten(1),
                          posbase(top,bp,bq)-posbase(op,bp,bq),toqb,mul(oqb,conj(toqb)),r[:,192:197],self.shape[m]),-1)
    def parity(self,env,s,m):
        # Called immediately after explicit official compute_observations; ignore patched reset-body cache.
        o=env.obs_dict['actor'];wp=s[:,71:74];wq=s[:,239:243]
        noisy=(rotate(wq,o[:,80:83])+wp,mul(wq,o[:,83:87]))
        ours=self.obs(s,env.last_actions,m,env.progress_buf,noisy=noisy)
        valid=~env.just_reset
        return float((ours[valid]-o[valid]).abs().max()) if valid.any() else 0.

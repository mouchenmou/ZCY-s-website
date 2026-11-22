"""Frozen, new WM baseline for the motion-count experiment; not author SuperMimic code.

No response loss, Actor, Critic or termination head. Deterministic control is
explicit; the model predicts residual physical state, contact and official reward.
"""
import torch
from torch import nn

STATE = {'q': (0,29), 'qd': (29,58), 'object': (58,71),
         'body_pos': (71,155), 'body_vel': (155,239),
         'wrist_quat': (239,243), 'pad_force': (243,258),
         'torque': (258,287), 'contact': (287,292)}
SDIM = 292


def control(s, action, previous, last_action, mask_previous, delay, mask, c):
    """Known exogenous delay/mask draws; no true future PD targets in rollouts."""
    effective = torch.where(delay.bool().unsqueeze(-1), last_action, action)
    arm = s[..., :7] + effective[..., :7] * c['arm_action_scale']
    arm = c['arm_smoothing'] * arm + (1-c['arm_smoothing']) * previous[..., :7]
    max_step = s.new_tensor(c['arm_max_step'])
    arm = previous[..., :7] + (arm-previous[..., :7]).clamp(-max_step,max_step)
    arm = arm.clamp(s.new_tensor(c['arm_lower']),s.new_tensor(c['arm_upper']))
    lo, hi = s.new_tensor(c['hand_lower']), s.new_tensor(c['hand_upper'])
    hand = (effective[..., 7:].clamp(-1,1)+1)*.5*(hi-lo)+lo
    hand = c['hand_smoothing']*hand+(1-c['hand_smoothing'])*previous[...,7:]
    delta = hand-previous[...,7:]
    hand = previous[...,7:]+delta.sign()*(delta.abs()-c['hand_target_deadband']).clamp_min(0)
    hand = hand.clamp(lo,hi)
    hand = torch.where(mask.bool(),mask_previous,hand)
    return torch.cat((arm,hand),-1)


class WorldModel(nn.Module):
    def __init__(self, input_dim, hidden=2048):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),
                               nn.Linear(hidden,SDIM+1))
        # Same seed and initialization for every group, no pretrained weights.
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight,std=.001)

    def forward(self,x):
        return self.net(x)


def features(s,action,target,previous,last,mask_previous,history,ref,physics,age):
    return torch.cat((s,action,target,previous,last,mask_previous,history,ref,physics,age),-1)


def predict(model,norm,s,action,target,previous,last,mask_previous,history,ref,physics,age):
    x=features(s,action,target,previous,last,mask_previous,history,ref,physics,age)
    y=model((x-norm['x_mean'])/norm['x_std'])
    continuous=s[...,:287]+y[...,:287]*norm['delta_std']+norm['delta_mean']
    # Quaternion normalization is representation maintenance, not a physics prior.
    parts=[];offset=0
    for a,b in ((61,65),(239,243)):
        parts.extend((continuous[...,offset:a],nn.functional.normalize(continuous[...,a:b],dim=-1,eps=1e-8)))
        offset=b
    parts.append(continuous[...,offset:])
    continuous=torch.cat(parts,-1)
    state=torch.cat((continuous,y[...,287:292].sigmoid()),-1)
    reward=y[...,292]*norm['reward_std']+norm['reward_mean']
    return state,reward,y[...,287:292]


def advance_history(history,state):
    # Official reward uses force > 0, independent of distance-gated pad contacts.
    contact=(state[...,243:258].reshape(*state.shape[:-1],5,3).norm(dim=-1)>0).float()
    return torch.cat((history[...,5:],contact),-1)

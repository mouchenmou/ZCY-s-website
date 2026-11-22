"""Read-only common-seed true PhysX evaluation; never used to select updates."""
import argparse,json,time,os
from pathlib import Path
from sm4_env import build_env,physical_state,References,ROOT
import torch
from sm4_model import Actor,WorldModel,control,advance_history,digest

OUT=ROOT/'experiments/supermimic4'

@torch.no_grad()
def run(a):
    for g in ['3','12','20','30']:
        assert (OUT/'smoke'/('group_'+g)/'result.json').exists(),'Evaluate only after all four short runs finish'
    start=time.monotonic();frozen=json.loads((ROOT/'experiments/wm4_200k/nested_groups.json').read_text())
    ids=frozen['groups']['3']+frozen['heldout_ids'];env,cfg,cap,phys,c=build_env(ids,120,a.seed,True)
    refs=References(env,len(ids));motion=torch.arange(120,device='cuda')%8
    out=OUT/'evaluation'/f'{a.policy}_{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('Evaluation already completed')
    rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all()
    actor=Actor().cuda()
    path=OUT/'smoke'/('group_3' if a.policy=='initial' else 'group_'+a.policy)/('initial.pt' if a.policy=='initial' else 'final.pt')
    ck=torch.load(path,map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor']);del ck
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng);actor.eval()
    alive=torch.ones(120,dtype=torch.bool,device='cuda');returns=torch.zeros(120,device='cuda');lengths=torch.zeros(120,dtype=torch.long,device='cuda')
    sustained=torch.zeros_like(lengths);contacts=torch.zeros_like(lengths);onsets=torch.zeros_like(lengths);progress=torch.zeros_like(returns)
    success=torch.zeros_like(alive);reasons=torch.zeros_like(lengths);records={};maxpd=0.;maxobs=0.;peak=0.
    for step in range(int(refs.length.max())):
        if time.monotonic()-start>160:raise RuntimeError('Evaluation time budget')
        free,total=torch.cuda.mem_get_info();peak=max(peak,(total-free)/2**20)
        assert total-free<24*2**30
        # Ended lanes are ignored, never counted as reset-spanning trajectories.
        env.progress_buf[~alive]=0
        env.compute_observations();s=physical_state(env);last=env.last_actions.clone();prev=env.prev_targets.clone();mp=env.mask_previous_command.clone()
        hist=env.tip_contact_history.flatten(1).float().clone();age=env.running_progress_buf[:,None].float().clone();frame=env.progress_buf.clone()
        selected=alive&~env.just_reset;action=actor(env.obs_dict['actor']);ph=phys.clone()
        err=refs.parity(env,s,motion);maxobs=max(maxobs,err);assert err<3e-5
        _,reward,done,_=env.step(action);sn=physical_state(env);reward=reward.clone();done=done.clone().bool()
        err=float((control(s,action,prev,last,mp,cap['delay'],cap['mask'],c)-env.pos_targets).abs().max());maxpd=max(maxpd,err);assert err<2e-6
        contact=sn[:,287:292].bool().any(-1);prevcontact=s[:,287:292].bool().any(-1)
        returns+=reward*alive;lengths+=alive.long();contacts+=(alive&contact).long()
        sustained+=(selected&contact&prevcontact).long();onsets+=(selected&contact&~prevcontact).long()
        progress=torch.maximum(progress,torch.where(alive,(frame+1)/refs.length[motion],torch.zeros_like(progress)))
        ended=alive&done;success[ended]=env.success_buf[ended].bool();reasons[ended]=env.done_reason_bits[ended]
        if a.policy=='initial':
            row={'s':s,'sn':sn,'action':action,'last':last,'previous':prev,'mask_previous':mp,'history':hist,'age':age,'frame':frame,
                 'motion':motion,'physics':ph,'ref':refs.get(motion,frame),'target':env.pos_targets.clone(),
                 'delay':cap['delay'].clone(),'mask':cap['mask'].clone(),'reward':reward,'failure':env.failure_buf.clone().bool(),
                 'done':done,'selected':selected,'alive':alive.clone()}
            for k,v in row.items():records.setdefault(k,[]).append(v.cpu())
        alive&=~done
        if not alive.any():break
    episodes=[]
    for i in range(120):
        episodes.append({'motion':ids[int(motion[i])],'split':'core_seen_all' if int(motion[i])<3 else 'heldout_unseen_all',
                         'return':float(returns[i]),'length':int(lengths[i]),'success':bool(success[i]),'truncated':bool(alive[i]),
                         'reference_max_fraction':float(progress[i]),'contact_steps':int(contacts[i]),'sustained_steps':int(sustained[i]),
                         'contact_onsets':int(onsets[i]),'reason_bits':int(reasons[i])})
    if records:torch.save({k:torch.stack(v) for k,v in records.items()},out/'common_physx.pt')
    result={'policy':a.policy,'seed':a.seed,'seconds':time.monotonic()-start,'physical_steps':step+1,'raw_physical_transitions':(step+1)*120,
            'counted_episode_transitions':int(lengths.sum()),'gpu_peak_mib':peak,'pd_error':maxpd,'obs_error':maxobs,
            'actor_hash':digest(actor),'episodes':episodes,'done_reason_names':env.done_reason_schema()[0]}
    (out/'result.json').write_text(json.dumps(result,indent=2));env.gym.destroy_sim(env.sim)
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','done_reason_names']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['initial','3','12','20','30'],required=True);p.add_argument('--seed',type=int,required=True);run(p.parse_args())

"""Audit all recorded old episodes and reconstruct official failure conditions."""
import sys,json,csv,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/wm4_200k';OUT=EXP/'followup'
spec=importlib.util.spec_from_file_location('robot',ROOT/'sources/GALATEA/rl/armhand/envs/robot.py');robot=importlib.util.module_from_spec(spec);spec.loader.exec_module(robot)
from wm4_time_response import write_csv,stats

def reasons(d,lengths):
    sn=d['sn'];ref=d['ref'];age=d['age'][...,0];f=d['frame']
    jp=ref[...,12:93].reshape(*ref.shape[:2],27,3).clone();jp[...,2]+=ref[...,198,None]
    errors=(jp-sn[...,71:155].reshape(*sn.shape[:2],28,3)[...,1:,:]).norm(dim=-1)
    checks={}
    for key,threshold in [('thumb_tip',.08),('index_tip',.09),('middle_tip',.10),('ring_tip',.12),('pinky_tip',.12),('level_1_joints',.14),('level_2_joints',.16)]:
        checks[key]=(errors[...,[k-1 for k in robot.KEYPOINT_GROUPS[key]]].mean(-1)>threshold)&(age>=8)
    obj=ref[...,174:186].reshape(*ref.shape[:2],3,4);pos=obj[...,:3,3].clone();pos[...,2]+=ref[...,198]
    checks['object_position']=((pos-sn[...,58:61]).norm(dim=-1)>(.04+ref[...,197]))&(age>=8)
    q=sn[...,61:65];x,y,z,w=q.unbind(-1)
    mat=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)
    cos=((mat*obj[...,:3,:3]).sum((-1,-2))-1)/2
    checks['object_rotation']=(cos<np.cos(np.pi/6))&(age>=8)
    checks['arm_velocity_limit']=(sn[...,29:36].abs()/torch.tensor(robot.ARM_VELOCITY_LIMITS)).amax(-1)>.15
    checks['arm_velocity_limit']&=(age>=2)
    fail=torch.stack(list(checks.values())).any(0)
    checks['timeline_success']=(f+4>=lengths[None,:])&~fail
    pred=fail|checks['timeline_success']
    return checks,{'predicted_done_mismatch':int((pred!=d['done']).sum()),'total_done':int(d['done'].sum())}

def analyze(path,label):
    d=torch.load(path,map_location='cpu',weights_only=False);T,N=d['frame'].shape
    lengths=torch.tensor([len(np.load(ROOT/f'sources/GALATEA/rl/data/power_drill/ik/{d["ids"][i%len(d["ids"])]}_fr3_retarget.npz')['arm_qpos']) for i in range(N)])
    if 'done_reasons' in d:
        checks={name:(d['done_reasons']&(1<<j))!=0 for j,name in enumerate(d['done_reason_names'])};audit={'recorded_official_reasons':True}
    else:checks,audit=reasons(d,lengths)
    rows=[];selected=d['selected'];contact=d['sn'][...,287:].bool().any(-1);before=d['s'][...,287:].bool().any(-1)
    phase=(d['frame'].float()/lengths[None,:]*4).long().clamp(0,3)
    for lane in range(N):
        es=d['episode'][:,lane].numpy();starts=np.r_[0,np.flatnonzero(np.diff(es))+1];ends=np.r_[starts[1:]-1,T-1]
        for st,en in zip(starts,ends):
            terminal=bool(d['done'][en,lane]);cont=contact[st:en+1,lane].numpy();longest=run=0
            for value in cont:run=run+1 if value else 0;longest=max(longest,run)
            rows.append({'label':label,'motion':d['ids'][lane%len(d['ids'])],'lane':lane,'episode':int(es[st]),
                 'start_frame':int(d['frame'][st,lane]),'end_frame_after_action':int(d['frame'][en,lane])+1,
                 'reference_length':int(lengths[lane]),'steps':int(en-st+1),'terminated':terminal,
                 'right_censored_by_collection_budget':not terminal,'reasons':';'.join(k for k,v in checks.items() if v[en,lane]) if terminal else 'collection_budget',
                 'contact_steps':int(cont.sum()),'longest_contact_run':longest,
                 'selected_steps':int(selected[st:en+1,lane].sum())})
    write_csv(OUT/f'{label}_episodes.csv',rows)
    per_motion=[]
    for m,mid in enumerate(d['ids']):
        mask=selected & (torch.arange(N)[None,:]%len(d['ids'])==m)
        per_motion.append({'motion':mid,'transitions':int(mask.sum()),'quartile_counts':torch.bincount(phase[mask],minlength=4).tolist(),
            'completed_episodes':sum(r['terminated'] for r in rows if r['motion']==mid),
            'successes':sum('timeline_success' in r['reasons'] for r in rows if r['motion']==mid),
            'max_reached_frame':max(r['end_frame_after_action'] for r in rows if r['motion']==mid)})
    completed=[r for r in rows if r['terminated']]
    events=(selected&contact&~before);sust=selected&contact&before
    result={'label':label,'raw_transitions':T*N,'selected_transitions':int(selected.sum()),'episodes':len(rows),
        'completed_episodes':len(completed),'censored_episodes':len(rows)-len(completed),
        'completed_episode_length':stats([r['steps'] for r in completed]),
        'successes':sum('timeline_success' in r['reasons'] for r in rows),
        'termination_reasons_nonexclusive':{k:sum(k in r['reasons'].split(';') for r in completed) for k in checks},
        'reference_quartile_counts':torch.bincount(phase[selected],minlength=4).tolist(),
        'reset_quartile_counts':np.bincount([min(3,int(r['start_frame']/r['reference_length']*4)) for r in rows],minlength=4).tolist(),
        'contact_episode_count':sum(r['contact_steps']>0 for r in rows),'contact_onsets':int(events.sum()),
        'sustained_transitions':int(sust.sum()),'contact_transitions':int((selected&contact).sum()),
        'episodes_contact_run_at_least_8':sum(r['longest_contact_run']>=8 for r in rows),
        'selected_reward_mean':float(d['reward'][selected].mean()),'motion_details':per_motion,'audit':audit}
    if 'done_metrics' in d:
        result['object_error_mm_mean']=float(d['done_metrics'][selected][:,3].mean()*1000)
    (OUT/f'{label}_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='motion_details'}),flush=True)
    return result

if __name__=='__main__':
    torch.set_num_threads(4)
    if len(sys.argv)>1 and sys.argv[1]=='pilot':
        results={m:analyze(OUT/f'coverage_{m}.pt','pilot_'+m) for m in ['script','expert','expert_early']}
    else:results={g:analyze(EXP/f'local_run/train_{g}.pt','old_'+g) for g in ['3','12','20','30']}
    (OUT/('pilot_summary.json' if 'script' in results else 'old_summary.json')).write_text(json.dumps(results,indent=2))

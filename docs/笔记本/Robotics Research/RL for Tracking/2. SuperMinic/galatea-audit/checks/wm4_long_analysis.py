"""Variable-length, first-terminal-censored PhysX/WM response diagnosis."""
import json,sys,time,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from wm4_model import WorldModel,control,predict,advance_history
from wm4_time_response import write_csv,sha,stats
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/wm4_200k';OUT=EXP/'followup'
DEG=180/math.pi;GROUPS=['3','12','20','30'];NAMES=['PhysX']+['WM-'+g for g in GROUPS]

def longest(x):
    run=best=0
    for b in x:run=run+1 if b else 0;best=max(best,run)
    return best

def recovery(q,threshold,start=8):
    ok=np.abs(q)<=threshold
    for t in range(start,len(ok)-7):
        if ok[t:t+8].all():return t+1
    return None

@torch.no_grad()
def rollout(model,norm,d,ends):
    keys=['s','previous','last','mask_previous','history','action','delay','mask','ref','physics','age']
    x={k:d[k].cuda() for k in keys};s=x['s'][0];prev=x['previous'][0];last=x['last'][0];mp=x['mask_previous'][0];hist=x['history'][0]
    state0=s.clone();ends=ends.cuda();result=[];targets=[];bad=torch.zeros(len(ends),dtype=torch.bool,device='cuda');badstep=torch.zeros_like(ends)
    for t in range(len(d['action'])):
        target=control(s,x['action'][t],prev,last,mp,x['delay'][t],x['mask'][t],d['contract'])
        sn,_,_=predict(model,norm,s,x['action'][t],target,prev,last,mp,hist,x['ref'][t],x['physics'][t],x['age'][t])
        invalid=~torch.isfinite(sn).all(-1);new=invalid&~bad&(ends>t);badstep[new]=t+1;bad|=invalid
        visible=(ends>t)&~bad
        result.append(torch.where(visible[:,None],sn,torch.full_like(sn,float('nan'))).cpu())
        targets.append(torch.where(visible[:,None],target,torch.full_like(target,float('nan'))).cpu())
        s=torch.where(visible[:,None],sn,state0);prev=torch.where(visible[:,None],target,x['previous'][0]);last=x['action'][t]
        mp=prev[:,7:];hist=advance_history(hist,s)
    return {'sn':torch.stack(result),'target':torch.stack(targets),'nonfinite_first_step':badstep.cpu()}

def equal_physical(a,b):
    if torch.equal(a,b):return True
    if a.shape[-1]!=292:return False
    return (torch.equal(a[...,:243],b[...,:243]) and torch.equal(a[...,258:],b[...,258:])
            and float((a[...,243:258]-b[...,243:258]).abs().max())<=5e-6)

def run(include_expert=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    start=time.monotonic();ckhash={g:sha(EXP/f'local_run/wm_{g}/final.pt') for g in GROUPS}
    joint_names=json.loads((EXP/'time_response/dof_names.json').read_text())
    joints={j.get('name'):j for j in ET.parse(ROOT/'sources/GALATEA/rl/assets/robot/fr3_sharpa_right.urdf').getroot().findall('joint')}
    hand_limits=torch.tensor([float(joints[n].find('limit').get('velocity')) for n in joint_names[7:]])
    all_cases=[];metrics=[];curve_sets={};audits={};time_rows=[];controls=[];returns=[]
    sets=['core','unseen','core2','unseen2']+(['expert128','expert512'] if include_expert else [])
    for label in sets:
        d={k:torch.load(OUT/f'{label}_{k}.pt',weights_only=False) for k in ['base','repeat','plus']};b,r,p=[d[k] for k in ['base','repeat','plus']]
        bitwise_mismatches=[k for k,v in b.items() if isinstance(v,torch.Tensor) and not torch.equal(v,r[k])]
        mismatches=[k for k in bitwise_mismatches if k not in ['s','sn','warmup'] or not equal_physical(b[k],r[k])]
        assert not mismatches,(label,mismatches)
        assert equal_physical(b['warmup'],p['warmup']),label
        force_prefix_error=float((b['warmup'][...,243:258]-p['warmup'][...,243:258]).abs().max())
        for k in ['s','previous','last','mask_previous','history','ref','physics','age']:
            assert (equal_physical(b[k][0],p[k][0]) if k=='s' else torch.equal(b[k][0],p[k][0])),(label,k)
        # Identical model initial input; physical force reduction noise is separately audited.
        p['s'][0]=b['s'][0].clone()
        for k in ['delay','mask','physics']:assert torch.equal(b[k],p[k]),(label,k)
        T,N=b['action'].shape[:2];lanes=torch.arange(N);joint=7+lanes%22
        ends=torch.full((N,),T,dtype=torch.long)
        initial=b['initial_done']|p['initial_done']
        for i in range(N):
            failed=(b['done'][:,i]|p['done'][:,i]).nonzero().flatten();ends[i]=0 if initial[i] else (int(failed[0]) if len(failed) else T)
        strict_censored={}
        for branch in [b,p]:
            assert not ((branch['warmup'][2:,:,36:58].abs()/hand_limits).amax(-1)>.35).any(), 'Strict hand gate in warmup'
            violations=((branch['sn'][...,36:58].abs()/hand_limits).amax(-1)>.35)&(branch['age'][...,0]>=2)
            for i in range(N):
                hits=violations[:int(ends[i]),i].nonzero().flatten()
                if len(hits):ends[i]=int(hits[0]);strict_censored[str(i)]=int(ends[i])+1
        expected=b['action'].clone();expected[:8,lanes,joint]=(expected[:8,lanes,joint]+.2).clamp(-1,1);assert torch.equal(expected,p['action'])
        for t in range(T):
            active=ends>t
            assert torch.equal(b['frame'][t,active],p['frame'][t,active])
            assert (b['frame'][t,active]<b['seq_len'][t,active]-4).all()
            for branch in [b,p]:
                if t>0:assert torch.equal(branch['s'][t,active],branch['sn'][t-1,active])
        maxerr=max(json.loads((OUT/f'{label}_{branch}.json').read_text())['max_pd_error'] for branch in ['base','repeat','plus']);assert maxerr<2e-6
        prefix=None
        if label in ['core','unseen']:
            old=torch.load(EXP/f'local_run/probe_{label}_base.pt',weights_only=False)
            prefix=torch.equal(b['sn'][:32],old['sn']);assert prefix
        audits[label]={'baseline_repeat_all_tensors_exact':not bitwise_mismatches,'baseline_repeat_state_except_force_exact':True,
                       'baseline_repeat_force_max_error_N':float((b['sn'][...,243:258]-r['sn'][...,243:258]).abs().max()),
                       'branch_complete_warmup_exact':force_prefix_error==0,'branch_prefix_force_max_error_N':force_prefix_error,'shared_model_initial_state_exact':True,'old32_baseline_exact':prefix,
                       'max_pd_error_rad':maxerr,'horizon_survivors':{str(h):int((ends>=h).sum()) for h in [8,32,64,128,256]},
                       'strict_hand_gate_censored':strict_censored,'initial_done_n':int(initial.sum()),'candidates':N,'new_physics_steps':3*N*(T+b['warmup'].shape[0])}
        truthq=(p['sn'][:,:,:29]-b['sn'][:,:,:29])[:,lanes,joint].numpy()*DEG
        truthv=(p['sn'][:,:,29:58]-b['sn'][:,:,29:58])[:,lanes,joint].numpy()*DEG
        trutht=(p['target']-b['target'])[:,lanes,joint].numpy()*DEG
        models={'PhysX':{'q':truthq,'qd':truthv,'target':trutht,'nonfinite_first_step':np.zeros(N,int)}}
        for g in GROUPS:
            ck=torch.load(EXP/f'local_run/wm_{g}/final.pt',map_location='cuda',weights_only=False);m=WorldModel(ck['input_dim'],ck['hidden']).cuda();m.load_state_dict(ck['model']);m.eval();m.requires_grad_(False)
            pred={k:rollout(m,ck['norm'],d[k],ends) for k in ['base','plus']}
            out={}
            for key,offset in [('q',0),('qd',29)]:out[key]=((pred['plus']['sn']-pred['base']['sn'])[:,lanes,joint+offset]*DEG).numpy()
            out['target']=((pred['plus']['target']-pred['base']['target'])[:,lanes,joint]*DEG).numpy()
            out['nonfinite_first_step']=torch.maximum(pred['plus']['nonfinite_first_step'],pred['base']['nonfinite_first_step']).numpy()
            models['WM-'+g]=out
            del m,ck,pred
        cases=[]
        for i in range(N):
            end=int(ends[i]);j=int(joint[i]);finger=['thumb','index','middle','ring','pinky'].index(joint_names[j].split('_')[1])
            cc=(b['sn'][:end,i,287:].bool()|p['sn'][:end,i,287:].bool()).numpy();contact=cc.any(-1)
            initialcontact=bool((b['s'][0,i,287:].bool()|p['s'][0,i,287:].bool()).any());own=cc[:,finger]
            cat='sustained_contact' if longest(contact)>=8 else ('first_or_brief_contact' if contact.any() or initialcontact else 'no_contact')
            peak=float(np.abs(truthq[:min(end,32),i]).max()) if end else 0
            threshold=min(.5,.1*peak);clear=end>=32 and peak>=1 and np.abs(trutht[:8,i]).max()>=.1
            reason='initial_done' if initial[i] else ('observation_limit' if end==T else ';'.join(k for bit,k in enumerate(b['done_reason_names']) if (int(b['done_reasons'][end,i])|int(p['done_reasons'][end,i]))&(1<<bit)))
            if str(i) in strict_censored:reason='strict_hand_velocity_limit'
            item={'set':label,'population':'expert_core' if label.startswith('expert') else ('script_unseen' if label.startswith('unseen') else 'script_core'),
                'lane':i,'motion':b['ids'][i%len(b['ids'])],'joint':joint_names[j],'valid_steps':end,'excluded_reason':reason,
                'initial_reference_frame':int(b['frame'][0,i]),'reference_length':int(b['seq_len'][0,i]),'contact_category':cat,
                'initial_contact':initialcontact,'own_contact_steps':int(own.sum()),'any_contact_steps':int(contact.sum()),
                'max_contact_run':longest(contact),'first_contact_step':int(np.flatnonzero(contact)[0])+1 if contact.any() else None,
                'clear32':clear,'true_peak_first32_deg':peak,'near_threshold_deg':threshold,
                'q_return_start':recovery(truthq[:end,i],threshold) if clear else None,
                'pd_return_start':recovery(trutht[:end,i],.5) if clear else None}
            cases.append(item);all_cases.append(item)
            for model in NAMES:
                q=models[model]['q'][:end,i];v=models[model]['qd'][:end,i];target=models[model]['target'][:end,i]
                if clear:
                    ret=recovery(q,threshold);returns.append({**item,'model':model,'return_start':ret,
                        'stays_near_until_observed_end':bool(np.all(np.abs(q[ret-1:])<=threshold)) if ret else False,
                        'q_at_true_return_deg':float(q[item['q_return_start']-1]) if item['q_return_start'] else None,
                        'pd_at_true_return_deg':float(trutht[item['q_return_start']-1,i]) if item['q_return_start'] else None,
                        'near_on_true_return_window':bool(np.all(np.abs(q[item['q_return_start']-1:item['q_return_start']+7])<=threshold)) if item['q_return_start'] else None})
                for h in [8,32,64,128,256]:
                    if end<h:continue
                    err=np.abs(q[:h]-truthq[:h,i]);vel=np.abs(v[:h]-truthv[:h,i])
                    cont=contact[:h];phase='sustained_contact' if longest(cont)>=8 else ('first_or_brief_contact' if cont.any() or initialcontact else 'no_contact')
                    metrics.append({'set':label,'population':item['population'],'lane':i,'motion':item['motion'],'joint':item['joint'],'horizon':h,'model':model,
                        'clear32':clear,'contact_category':phase,'q_mae_deg':float(err.mean()),'qd_mae_deg_s':float(vel.mean()),
                        'start_mae_deg':float(err[:8].mean()),'withdrawal_mae_deg':float(err[8:].mean()) if h>8 else None,
                        'endpoint_delta_deg':float(q[h-1]),'true_endpoint_deg':float(truthq[h-1,i]),'pd_endpoint_deg':float(trutht[h-1,i]),
                        'peak_amplitude_error_deg':float(abs(np.abs(q[:h]).max()-np.abs(truthq[:h,i]).max())),
                        'prediction_nonfinite':bool(not np.isfinite(q[:h]).all())})
                for t in range(end):
                    time_rows.append({'set':label,'lane':i,'motion':item['motion'],'joint':item['joint'],'step':t+1,'model':model,
                        'q_delta_deg':float(q[t]),'qd_delta_deg_s':float(v[t]),'pd_delta_deg':float(target[t]),
                        'reference_frame_before':int(b['frame'][t,i]),'reference_fraction':float(b['frame'][t,i]/b['seq_len'][t,i]),
                        'done_base':False,'done_plus':False})
            # All steps, including terminal boundary, retain a validity flag; post-terminal states excluded from results.
            for t in range(min(T,end+1)):
                controls.append({'set':label,'lane':i,'step':t+1,'valid':t<end,'frame_base':int(b['frame'][t,i]),'frame_plus':int(p['frame'][t,i]),
                    'done_base':bool(b['done'][t,i]),'done_plus':bool(p['done'][t,i]),'delay':bool(b['delay'][t,i]),
                    'mask':bool(b['mask'][t,i,j-7]),'action_base':float(b['action'][t,i,j]),'action_plus':float(p['action'][t,i,j]),
                    'pd_base_deg':float(b['target'][t,i,j]*DEG),'pd_plus_deg':float(p['target'][t,i,j]*DEG)})
        curve_sets[label]={'cases':cases,'models':models,'any_contact':(b['sn'][...,287:].bool()|p['sn'][...,287:].bool()).any(-1).numpy()}
        print(label,audits[label],flush=True)
    write_csv(OUT/'long_case_inventory.csv',all_cases);write_csv(OUT/'long_metrics.csv',metrics);write_csv(OUT/'long_time_series.csv',time_rows)
    write_csv(OUT/'long_controls_and_done.csv',controls);write_csv(OUT/'long_recovery.csv',returns)
    summary={}
    for population in ['script_core','script_unseen','expert_core']:
        summary[population]={}
        for h in [32,64,128,256]:
            summary[population][str(h)]={}
            for phase in ['all','no_contact','first_or_brief_contact','sustained_contact']:
                selected=[x for x in metrics if x['population']==population and x['horizon']==h and x['clear32'] and (phase=='all' or x['contact_category']==phase)]
                summary[population][str(h)][phase]={m:{k:stats([x[k] for x in selected if x['model']==m and np.isfinite(x[k])]) for k in ['q_mae_deg','qd_mae_deg_s','start_mae_deg','withdrawal_mae_deg','peak_amplitude_error_deg','endpoint_delta_deg']} for m in NAMES}
    recovery_summary={}
    for pop in ['script_core','script_unseen','expert_core']:
        cs=[x for x in all_cases if x['population']==pop and x['clear32']]
        rr=[x for x in cs if x['q_return_start']]
        recovery_summary[pop]={'clear32_cases':len(cs),'observed_returns':len(rr),'return_step':stats([x['q_return_start'] for x in rr]),
            'no_observed_return_valid_length':stats([x['valid_steps'] for x in cs if not x['q_return_start']]),
            'models':{m:{'predicted_returns':sum(x['return_start'] is not None for x in returns if x['population']==pop and x['model']==m),
            'near_during_true_return':sum(x['near_on_true_return_window'] is True for x in returns if x['population']==pop and x['model']==m),
            'return_step_absolute_error':stats([abs(x['return_start']-x['q_return_start']) for x in returns if x['population']==pop and x['model']==m and x['return_start'] and x['q_return_start']])} for m in NAMES}}
    assert all(sha(EXP/f'local_run/wm_{g}/final.pt')==h for g,h in ckhash.items())
    (OUT/'long_summary.json').write_text(json.dumps({'audit':audits,'metrics':summary,'recovery':recovery_summary,'model_hashes':ckhash,'seconds':time.monotonic()-start,'nonfinite_rows':sum(x['prediction_nonfinite'] for x in metrics)},indent=2))
    torch.save(curve_sets,OUT/'long_curves.pt');plot(curve_sets)

def plot(curves):
    sys.path.insert(0,str(ROOT/'audit-lib/plots'));import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors=['black','#2378B5','#E08A18','#23925C','#9954B0']
    def draw(ax,label,i,key='q'):
        d=curves[label];c=d['cases'][i];end=c['valid_steps'];x=np.arange(1,end+1)
        for name,col in zip(NAMES,colors):ax.plot(x,d['models'][name][key][:end,i],label=name,color=col,lw=2 if name=='PhysX' else 1)
        ax.axvspan(1,8,color='orange',alpha=.12);ax.axhline(0,color='grey',lw=.5);ax.grid(alpha=.15)
        ax.set_title(f"{label} / {c['motion']} / lane {i} / {c['joint']}\n{c['contact_category']} / valid {end} steps",fontsize=8)
        ax.set_ylabel({'q':'Extra angle (deg)','qd':'Extra velocity (deg/s)','target':'Extra PD target (deg)'}[key]);ax.set_xlabel('Control step')
    # All >=8-step valid cases, including weak responses, are shown. No error-based selection.
    eligible=[(label,i,c) for label,d in curves.items() for i,c in enumerate(d['cases']) if c['valid_steps']>=8]
    with PdfPages(OUT/'long_all_valid_cases.pdf') as pdf:
        for offset in range(0,len(eligible),12):
            fig,axes=plt.subplots(4,3,figsize=(15,14))
            for ax,(label,i,c) in zip(axes.flat,eligible[offset:offset+12]):draw(ax,label,i)
            for ax in list(axes.flat)[len(eligible[offset:offset+12]):]:ax.set_visible(False)
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='upper center');fig.tight_layout(rect=(0,0,1,.97));pdf.savefig(fig);plt.close(fig)
    examples=[]
    for pop in ['script_core','script_unseen','expert_core']:
        for phase in ['no_contact','first_or_brief_contact','sustained_contact']:
            pool=[(l,i,c) for l,i,c in eligible if c['population']==pop and c['contact_category']==phase and c['clear32'] and c['valid_steps']>=64]
            if not pool:continue
            # Prefer >=128 observation if available, then median real peak; no WM error criterion.
            longpool=[x for x in pool if x[2]['valid_steps']>=128];pool=longpool or pool
            med=np.median([c['true_peak_first32_deg'] for l,i,c in pool]);label,i,c=min(pool,key=lambda x:(abs(x[2]['true_peak_first32_deg']-med),x[0],x[1]));examples.append(c)
            fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
            for ax,key in zip(axes[:3],['q','qd','target']):draw(ax,label,i,key)
            end=c['valid_steps'];axes[3].step(np.arange(1,end+1),curves[label]['any_contact'][:end,i].astype(int),where='mid');axes[3].set_ylabel('Any hand contact');axes[3].set_xlabel('Control step')
            axes[0].legend(ncol=5);axes[0].axhspan(-c['near_threshold_deg'],c['near_threshold_deg'],color='green',alpha=.1)
            fig.tight_layout();fig.savefig(OUT/f'long_{label}_{i:03d}.png',dpi=150);plt.close(fig)
    (OUT/'long_examples.json').write_text(json.dumps(examples,indent=2,default=lambda x:x.item()))
if __name__=='__main__':run('--expert' in sys.argv)

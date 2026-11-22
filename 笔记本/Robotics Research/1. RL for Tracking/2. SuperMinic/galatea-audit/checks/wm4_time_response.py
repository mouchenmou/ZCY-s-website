"""Read-only model diagnosis: positive 8-step pulse minus baseline, 32 steps.

Reuses validated real GPU PhysX trajectories; never trains or saves model weights.
All selected examples are chosen from PhysX/control metadata before WM inference.
"""
import csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from wm4_model import WorldModel, control, predict, advance_history

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/wm4_200k'
RAW=EXP/'local_run'
OUT=EXP/'time_response'
DEG=180/math.pi
GROUPS=('3','12','20','30')
STEPS=(1,4,8,16,24,32)
FINGERS=('thumb','index','middle','ring','pinky')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def stats(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return {'n':0}
    return {'n':len(x),'median':float(np.median(x)),
            'q25':float(np.quantile(x,.25)),'q75':float(np.quantile(x,.75)),
            'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean())}


@torch.no_grad()
def rollout(model,norm,data,lanes):
    d={k:v[:,lanes].to('cuda') for k,v in data.items()
       if isinstance(v,torch.Tensor) and k not in ('warmup','initial_done')}
    c=data['contract'];s=d['s'][0];prev=d['previous'][0];last=d['last'][0]
    mp=d['mask_previous'][0];hist=d['history'][0];states=[];targets=[]
    for t in range(32):
        action=d['action'][t]
        target=control(s,action,prev,last,mp,d['delay'][t],d['mask'][t],c)
        s,_,_=predict(model,norm,s,action,target,prev,last,mp,hist,d['ref'][t],d['physics'][t],d['age'][t])
        assert torch.isfinite(s).all()
        states.append(s.cpu());targets.append(target.cpu())
        prev=target;last=action;mp=target[:,7:];hist=advance_history(hist,s)
    return {'sn':torch.stack(states),'target':torch.stack(targets)}


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    names=json.loads((OUT/'dof_names.json').read_text())
    assert names[:7]==[f'fr3_joint{i}' for i in range(1,8)]
    ckhash={g:sha(RAW/f'wm_{g}/final.pt') for g in GROUPS}
    data={};inventory=[];audit={};inputs={};selected=[]
    for split in ('core','unseen'):
        d={b:torch.load(RAW/f'probe_{split}_{b}.pt',map_location='cpu',weights_only=False)
           for b in ('base','repeat','plus')}
        b,r,p=(d[k] for k in ('base','repeat','plus'))
        for branch in d:inputs[f'probe_{split}_{branch}.pt']=sha(RAW/f'probe_{split}_{branch}.pt')
        # Exact simulator replay evidence, not just restoring visible q/qd tensors.
        assert b['seed']==r['seed']==p['seed']
        assert torch.equal(b['warmup'],r['warmup']) and torch.equal(b['warmup'],p['warmup'])
        for key,value in b.items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,r[key]),'Repeat mismatch: '+key
        for key in ('s','previous','last','mask_previous','history','physics','ref','age'):
            assert torch.equal(b[key][0],p[key][0]),'Initial context mismatch: '+key
        for key in ('delay','mask','ref','physics','age'):
            assert torch.equal(b[key],p[key]),'External context mismatch: '+key
        for branch in (b,p):
            assert torch.equal(branch['s'][1:],branch['sn'][:-1]),'Reset/state discontinuity'
            assert torch.equal(branch['age'][1:],branch['age'][:-1]+1)
        n=b['s'].shape[1];lanes=torch.arange(n);joint=7+lanes%22
        expected=b['action'].clone()
        expected[:8,lanes,joint]=(expected[:8,lanes,joint]+.2).clamp(-1,1)
        assert torch.equal(expected,p['action'])
        assert torch.equal(p['action'][8:],b['action'][8:])
        valid=~(b['initial_done']|p['initial_done']|b['done'].any(0)|p['done'].any(0))
        delta=(p['sn'][:,:,:29]-b['sn'][:,:,:29])[:,lanes,joint]*DEG
        target_delta=(p['target']-b['target'])[:,lanes,joint]*DEG
        max_control_error=0.
        for branch in (b,p):
            for t in range(32):
                reconstructed=control(branch['s'][t],branch['action'][t],branch['previous'][t],
                    branch['last'][t],branch['mask_previous'][t],branch['delay'][t],branch['mask'][t],branch['contract'])
                max_control_error=max(max_control_error,float((reconstructed-branch['target'][t]).abs().max()))
        assert max_control_error<2e-6
        for i in range(n):
            j=int(joint[i]);finger=FINGERS.index(names[j].split('_')[1])
            contacts=torch.cat((b['s'][0:1,i,287:].bool()|p['s'][0:1,i,287:].bool(),
                                b['sn'][:,i,287:].bool()|p['sn'][:,i,287:].bool()))
            peak=float(delta[:,i].abs().max());actual_pulse=p['action'][:8,i,j]-b['action'][:8,i,j]
            done=(b['done'][:,i]|p['done'][:,i]).nonzero().flatten()
            item={'split':split,'lane':i,'motion':b['ids'][i%len(b['ids'])],
                'joint_index':j,'joint_name':names[j],'finger_contact_index':finger,
                'valid_32':bool(valid[i]),'first_done_step':int(done[0])+1 if len(done) else None,
                'initial_done':bool(b['initial_done'][i]|p['initial_done'][i]),
                'contact_phase':'perturbed_finger_contact' if contacts[:,finger].any() else
                    ('other_finger_contact' if contacts.any() else 'no_contact'),
                'own_contact_steps':contacts[:,finger].nonzero().flatten().tolist(),
                'any_contact_steps':contacts.any(-1).nonzero().flatten().tolist(),
                'pulse_unclipped':bool(torch.allclose(actual_pulse,torch.full_like(actual_pulse,.2),atol=1e-6,rtol=0)),
                'pulse_mask_count':int(b['mask'][:8,i,j-7].sum()),
                'peak_deg':peak,'peak_step':int(delta[:,i].abs().argmax())+1,
                'angle32_deg':float(delta[31,i]),'target32_deg':float(target_delta[31,i]),
                'residual32_fraction_of_peak':float(delta[31,i].abs()/peak) if peak>1e-8 else None,
                'clear_response':peak>=1. and float(target_delta[:8,i].abs().max())>=.1,
                'near_baseline_at32':peak>=1 and abs(float(delta[31,i]))<=.5 and abs(float(delta[31,i]))<=.1*peak}
            inventory.append(item)
        data[split]=d
        audit[split]={'complete_saved_prefix_equal':True,'all_repeat_tensors_bitwise_equal':True,
                      'shared_exogenous_control_equal':True,'continuity_and_age_checks_passed':True,
                      'max_recomputed_pd_target_error_rad':max_control_error,
                      'valid32':int(valid.sum()),'excluded32':int((~valid).sum())}
    # Example choice uses only true PhysX response and controls, never WM error.
    for split in ('core','unseen'):
        candidates=[x for x in inventory if x['split']==split and x['valid_32'] and x['clear_response']
                    and x['pulse_unclipped'] and x['pulse_mask_count']==0]
        for category in ('no_contact','perturbed_finger_contact'):
            pool=[x for x in candidates if x['contact_phase']==category]
            if pool:
                median=float(np.median([x['peak_deg'] for x in pool]))
                chosen=min(pool,key=lambda x:(abs(x['peak_deg']-median),x['lane']))
                selected.append({**chosen,'selection_pool_n':len(pool),'selection_median_peak_deg':median})
    specification={'contrast':'plus minus baseline, NOT plus minus minus',
        'pulse':'add 0.2 to same finger action for steps 1..8, remove at step 9',
        'strict_validity':'No initial done and no done in either branch at any step 1..32; no reset',
        'example_rule':'per split and no-contact/own-finger-contact strata, clear peak>=1 deg, unclipped pulse, unmasked pulse; choose closest to median true peak; numeric-lane tie break',
        'near_baseline_rule':'abs(delta_q32)<=0.5 degree AND <=10% true peak, for clear cases',
        'contact_rule':'official distance-gated pad-object contact, in either branch at initial state or steps 1..32; own-finger vs other-finger distinguished',
        'checkpoint_sha256':ckhash,'input_sha256':inputs,'examples':selected,'audit':audit,
        'new_physx_transitions':0,'training_updates':0}
    (OUT/'selection_and_audit.json').write_text(json.dumps(specification,indent=2))
    write_csv(OUT/'case_inventory.csv',inventory)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    curves={};per_case=[];time_rows=[];raw_rows=[]
    for split,d in data.items():
        cases=[x for x in inventory if x['split']==split and x['valid_32']]
        lanes=torch.tensor([x['lane'] for x in cases]);idx=torch.arange(len(cases))
        joint=torch.tensor([x['joint_index'] for x in cases])
        curves[split]={'cases':cases,'models':{}}
        pred={'PhysX':{b:{'sn':d[b]['sn'][:,lanes], 'target':d[b]['target'][:,lanes]} for b in ('base','plus')}}
        for g in GROUPS:
            ck=torch.load(RAW/f'wm_{g}/final.pt',map_location='cuda',weights_only=False)
            model=WorldModel(ck['input_dim'],ck['hidden']).cuda()
            model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
            pred['WM-'+g]={b:rollout(model,ck['norm'],d[b],lanes) for b in ('base','plus')}
        for name,branches in pred.items():
            q0=branches['base']['sn'][:,idx,joint]*DEG;q1=branches['plus']['sn'][:,idx,joint]*DEG
            v0=branches['base']['sn'][:,idx,joint+29]*DEG;v1=branches['plus']['sn'][:,idx,joint+29]*DEG
            p0=branches['base']['target'][:,idx,joint]*DEG;p1=branches['plus']['target'][:,idx,joint]*DEG
            curves[split]['models'][name]={'q':(q1-q0).numpy(),'qd':(v1-v0).numpy(),'target':(p1-p0).numpy()}
            truth=curves[split]['models']['PhysX'];current=curves[split]['models'][name]
            for k,case in enumerate(cases):
                i=case['lane'];j=case['joint_index'];finger=case['finger_contact_index']
                err=current['q'][:,k]-truth['q'][:,k]
                verr=current['qd'][:,k]-truth['qd'][:,k]
                per_case.append({'split':split,'lane':i,'motion':case['motion'],'joint':case['joint_name'],
                    'contact_phase':case['contact_phase'],'clear_response':case['clear_response'],'model':name,
                    'trajectory_q_mae_deg':float(np.abs(err).mean()),'pulse_q_mae_deg':float(np.abs(err[:8]).mean()),
                    'withdrawal_q_mae_deg':float(np.abs(err[8:]).mean()),'velocity_mae_deg_s':float(np.abs(verr).mean()),
                    'error32_abs_deg':float(abs(err[-1])),'angle32_deg':float(current['q'][-1,k]),
                    'peak_abs_deg':float(np.abs(current['q'][:,k]).max()),
                    'peak_step':int(np.abs(current['q'][:,k]).argmax())+1,
                    'pd_target_delta_error_max_deg':float(np.abs(current['target'][:,k]-truth['target'][:,k]).max())})
                for t in range(32):
                    time_rows.append({'split':split,'lane':i,'motion':case['motion'],'joint':case['joint_name'],
                        'contact_phase':case['contact_phase'],'model':name,'step':t+1,
                        'q_delta_deg':float(current['q'][t,k]),'qd_delta_deg_s':float(current['qd'][t,k]),
                        'pd_target_delta_deg':float(current['target'][t,k]),
                        'q_base_deg':float(q0[t,k]),'q_plus_deg':float(q1[t,k]),
                        'qd_base_deg_s':float(v0[t,k]),'qd_plus_deg_s':float(v1[t,k]),
                        'pd_target_base_deg':float(p0[t,k]),'pd_target_plus_deg':float(p1[t,k])})
                    if name=='PhysX':
                        delay=bool(d['base']['delay'][t,i]);mask=bool(d['base']['mask'][t,i,j-7])
                        effective={b:float(d[b]['last'][t,i,j] if delay else d[b]['action'][t,i,j]) for b in ('base','plus')}
                        raw_rows.append({'split':split,'lane':i,'step':t+1,'joint':case['joint_name'],
                            'action_base':float(d['base']['action'][t,i,j]),'action_plus':float(d['plus']['action'][t,i,j]),
                            'effective_action_base':effective['base'],'effective_action_plus':effective['plus'],
                            'delay_base':int(delay),'delay_plus':int(d['plus']['delay'][t,i]),
                            'mask_base':int(mask),'mask_plus':int(d['plus']['mask'][t,i,j-7]),
                            'own_contact_base':int(d['base']['sn'][t,i,287+finger]),
                            'own_contact_plus':int(d['plus']['sn'][t,i,287+finger]),
                            'any_contact_base':int(d['base']['sn'][t,i,287:].bool().any()),
                            'any_contact_plus':int(d['plus']['sn'][t,i,287:].bool().any()),
                            'done_base':int(d['base']['done'][t,i]),'done_plus':int(d['plus']['done'][t,i])})
    write_csv(OUT/'all_valid_time_series.csv',time_rows)
    write_csv(OUT/'all_valid_controls.csv',raw_rows)
    write_csv(OUT/'per_case_metrics.csv',per_case)
    chosen_keys={(x['split'],x['lane']) for x in selected}
    write_csv(OUT/'selected_case_timepoints.csv',[x for x in time_rows if (x['split'],x['lane']) in chosen_keys and x['step'] in STEPS])
    summary={}
    for split in ('core','unseen','pooled'):
        summary[split]={}
        for phase in ('all','no_contact','any_contact','perturbed_finger_contact'):
            def included(x):
                return ((split=='pooled' or x['split']==split) and
                        (phase=='all' or x['contact_phase']==phase or (phase=='any_contact' and x['contact_phase']!='no_contact')))
            true_cases=[x for x in inventory if x['valid_32'] and x['clear_response'] and included(x)]
            summary[split][phase]={'clear_case_n':len(true_cases),
                'true_peak_deg':stats([x['peak_deg'] for x in true_cases]),
                'true_abs_angle32_deg':stats([abs(x['angle32_deg']) for x in true_cases]),
                'true_residual32_fraction_of_peak':stats([x['residual32_fraction_of_peak'] for x in true_cases]),
                'near_baseline_at32_n':sum(x['near_baseline_at32'] for x in true_cases),'models':{}}
            for model in ('PhysX',*['WM-'+g for g in GROUPS]):
                rows=[x for x in per_case if x['clear_response'] and included(x) and x['model']==model]
                summary[split][phase]['models'][model]={key:stats([x[key] for x in rows]) for key in
                    ('trajectory_q_mae_deg','pulse_q_mae_deg','withdrawal_q_mae_deg','velocity_mae_deg_s','error32_abs_deg','angle32_deg','peak_abs_deg')}
    assert all(sha(RAW/f'wm_{g}/final.pt')==ckhash[g] for g in GROUPS)
    assert all(sha(RAW/name)==h for name,h in inputs.items())
    summary['audit']={'original_models_and_physx_files_unchanged':True,'inference_seconds':time.monotonic()-start,
                      'valid_cases':sum(x['valid_32'] for x in inventory),'clear_valid_cases':sum(x['valid_32'] and x['clear_response'] for x in inventory)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Only diagnostic outputs, never model parameters.
    torch.save(curves,OUT/'curves.pt')
    print(json.dumps({'selected_cases':[(x['split'],x['lane'],x['joint_name']) for x in selected],**summary['audit']}),flush=True)
    plot(curves,selected,raw_rows)


def plot(curves,selected,raw_rows):
    sys.path.insert(0,str(ROOT/'audit-lib/plots'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors={'PhysX':'#111111','WM-3':'#2378B5','WM-12':'#E08A18','WM-20':'#23925C','WM-30':'#9954B0'}
    x=np.arange(1,33)
    def angle(ax,split,k,title):
        for name,values in curves[split]['models'].items():
            ax.plot(x,values['q'][:,k],label=name,color=colors[name],lw=2.5 if name=='PhysX' else 1.4,
                    linestyle='-' if name=='PhysX' else '--')
        ax.axvspan(.5,8.5,color='#f6dd91',alpha=.35);ax.axvline(8.5,color='#8c762b',lw=.8)
        own_contact=curves[split]['cases'][k]['own_contact_steps']
        for step in own_contact:
            if step>=1:ax.axvspan(step-.5,step+.5,color='#5cab73',alpha=.16)
        if own_contact:
            ax.text(.02,.02,'Green bands: perturbed finger contact',transform=ax.transAxes,fontsize=7,color='#27653c')
        ax.axhline(0,color='#666666',lw=.6);ax.set_xlim(1,32);ax.set_xticks(STEPS)
        ax.set_title(title,fontsize=10);ax.set_ylabel('Extra joint angle (deg)');ax.grid(alpha=.2)
    for case in selected:
        split=case['split'];lane=case['lane'];k=next(i for i,c in enumerate(curves[split]['cases']) if c['lane']==lane)
        controls=[row for row in raw_rows if row['split']==split and row['lane']==lane]
        fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True,gridspec_kw={'height_ratios':[2,1.5,1.5,1,1.4]})
        title=f"{split} / {case['motion']} / lane {lane} / {case['joint_name']}\n{case['contact_phase']}; both branches valid through step 32"
        angle(axes[0],split,k,title);axes[0].legend(ncol=5,loc='upper right',fontsize=9)
        for ax,key,label in [(axes[1],'qd','Extra joint velocity (deg/s)'),(axes[2],'target','Extra PD target (deg)')]:
            for ni,(name,values) in enumerate(curves[split]['models'].items()):
                ax.plot(x,values[key][:,k],color=colors[name],lw=2.5 if name=='PhysX' else 1.2,
                    linestyle='-' if name=='PhysX' else '--',marker=None if key=='qd' or name=='PhysX' else 'o',
                    markevery=(ni,7),ms=4,fillstyle='none')
            ax.axhline(0,color='#666666',lw=.6);ax.set_ylabel(label);ax.grid(alpha=.2)
            ax.axvspan(.5,8.5,color='#f6dd91',alpha=.35);ax.axvline(8.5,color='#8c762b',lw=.8)
        axes[2].text(.98,.95,'WM finger PD targets overlap PhysX (known control mapping)',
                     transform=axes[2].transAxes,ha='right',va='top',fontsize=8)
        command=[r['action_plus']-r['action_base'] for r in controls]
        effective=[r['effective_action_plus']-r['effective_action_base'] for r in controls]
        axes[3].step(x,command,where='mid',label='Submitted action difference',color='#1e55a5')
        axes[3].step(x,effective,where='mid',label='After delay, before mask',color='#bb5816',ls='--')
        axes[3].set_ylabel('Action difference');axes[3].legend(fontsize=8,loc='upper right');axes[3].grid(alpha=.2)
        fields=['delay_base','mask_base','own_contact_base','own_contact_plus','any_contact_base','any_contact_plus']
        labels=['One-step delay','Perturbed joint masked','Own finger contact: base','Own finger contact: plus','Any finger contact: base','Any finger contact: plus']
        for row,(key,label) in enumerate(zip(fields,labels)):
            values=np.array([c[key] for c in controls],dtype=bool)
            axes[4].scatter(x[values],np.full(values.sum(),row),marker='s',s=28,
                            color='#538c65' if 'contact' in key else '#8b5274')
        axes[4].set_yticks(range(len(labels)),labels,fontsize=8);axes[4].set_ylim(5.7,-.7)
        axes[4].set_xlim(1,32);axes[4].set_xticks(STEPS);axes[4].grid(axis='x',alpha=.2)
        axes[4].set_xlabel('Control step (30 Hz); extra command removed at step 9')
        fig.suptitle('Positive 8-step action pulse minus the same-history baseline',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.98))
        stem=OUT/f'case_{split}_{lane:03d}'
        fig.savefig(str(stem)+'.png',dpi=170);fig.savefig(str(stem)+'.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4.8));angle(ax,split,k,title)
        ax.legend(ncol=5);ax.set_xlabel('Control step');fig.tight_layout()
        fig.savefig(str(stem)+'_angle.png',dpi=180);fig.savefig(str(stem)+'_angle.svg');plt.close(fig)
        svg=Path(str(stem)+'_angle.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    # Every fully valid case is visible; not just selected examples or averages.
    cases=[(s,k,c) for s in ('core','unseen') for k,c in enumerate(curves[s]['cases'])]
    with PdfPages(OUT/'all_48_valid_cases.pdf') as pdf:
        for offset in range(0,len(cases),12):
            fig,axes=plt.subplots(4,3,figsize=(14,14))
            for ax,entry in zip(axes.flat,cases[offset:offset+12]):
                s,k,c=entry;angle(ax,s,k,f"{s} / {c['motion']} / lane {c['lane']}\n{c['joint_name']} / {c['contact_phase']}")
            for ax in list(axes.flat)[len(cases[offset:offset+12]):]:ax.set_visible(False)
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='upper center')
            fig.tight_layout(rect=(0,0,1,.97));pdf.savefig(fig);plt.close(fig)
    # Show per-case errors, not a signed mean that could cancel opposite responses.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    rng=np.random.default_rng(5001)
    for ax,split in zip(axes,('core','unseen')):
        cs=curves[split]['cases'];truth=curves[split]['models']['PhysX']['q']
        for gi,g in enumerate(GROUPS):
            err=np.abs(curves[split]['models']['WM-'+g]['q']-truth).mean(0)
            use=np.array([c['clear_response'] for c in cs]);contact=np.array([c['contact_phase']!='no_contact' for c in cs])
            jitter=rng.uniform(-.12,.12,len(cs))
            for mask,marker,color in [(use&~contact,'o','#346c9c'),(use&contact,'^','#b15d36')]:
                ax.scatter(gi+jitter[mask],err[mask],marker=marker,c=color,alpha=.8,s=35)
            ax.plot([gi-.18,gi+.18],[np.median(err[use])]*2,color='black',lw=2)
        ax.set_xticks(range(4),['WM-'+g for g in GROUPS]);ax.set_title(split+' / clear true responses')
        ax.set_ylabel('Per-case 32-step response MAE (deg)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Each point is one case; triangles: hand contact, circles: no contact; black: median')
    fig.tight_layout();fig.savefig(OUT/'case_error_distribution.png',dpi=170);plt.close(fig)
    print('Plots and all-case PDF completed.',flush=True)


if __name__=='__main__':main()

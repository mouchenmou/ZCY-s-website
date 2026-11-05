"""Supplement diagnosis: phase at pulse, hysteresis, contact episodes and quality."""
import sys,json,csv,math
from pathlib import Path
import numpy as np
import torch
from wm4_time_response import write_csv,stats
from wm4_long_analysis import longest,NAMES,DEG
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/wm4_200k';OUT=EXP/'followup';torch.set_num_threads(4)
curves=torch.load(OUT/'long_curves.pt',weights_only=False);raw={};rows=[];comparison=[]
for label,cs in curves.items():
    b=torch.load(OUT/f'{label}_base.pt',weights_only=False);p=torch.load(OUT/f'{label}_plus.pt',weights_only=False);raw[label]={'base':b,'plus':p}
    for i,c in enumerate(cs['cases']):
        if c['valid_steps']<8:continue
        n=c['valid_steps'];finger=['thumb','index','middle','ring','pinky'].index(c['joint'].split('_')[1]);j=7+i%22
        own=(b['sn'][:n,i,287+finger].bool()|p['sn'][:n,i,287+finger].bool()).numpy()
        contact=cs['any_contact'][:n,i];phase='initial_contact' if c['initial_contact'] else ('first_contact_later' if contact.any() else 'no_contact')
        q=cs['models']['PhysX']['q'][:n,i];target=cs['models']['PhysX']['target'][:n,i]
        r={**c,'pulse_phase':phase,'own_contact_pulse_steps':int(own[:8].sum()),'own_longest_contact':longest(own),
           'true_q_last_deg':float(q[-1]),'true_target_last_deg':float(target[-1]),'true_q_last8_max_deg':float(abs(q[-8:]).max()),
           'true_target_last8_max_deg':float(abs(target[-8:]).max()),
           'true_target_last64_range_deg':float(target[-64:].max()-target[-64:].min()),
           'object_position_delta_last_mm':float((p['sn'][n-1,i,58:61]-b['sn'][n-1,i,58:61]).norm()*1000),
           'arm_target_delta_last_max_deg':float((p['target'][n-1,i,:7]-b['target'][n-1,i,:7]).abs().max()*DEG)}
        rows.append(r)
        if not c['clear32']:continue
        for h in [32,64,128,256]:
            if n<h:continue
            for m in NAMES[1:]:
                pred=cs['models'][m]['q'][:h,i]
                comparison.append({'population':c['population'],'set':label,'lane':i,'model':m,'horizon':h,'pulse_phase':phase,
                   'own_contact_pulse_steps':r['own_contact_pulse_steps'],'mae':float(abs(pred-q[:h]).mean()),
                   'peak_error':float(abs(abs(pred[:32]).max()-abs(q[:32]).max())),
                   'peak_timing_error':abs(int(abs(pred[:32]).argmax())-int(abs(q[:32]).argmax()))})
write_csv(OUT/'long_phase_and_endpoint_audit.csv',rows);write_csv(OUT/'long_phase_metrics.csv',comparison)
# All-case summaries (including weak or clipped responses), distinct from clear-case primary table.
metrics=list(csv.DictReader((OUT/'long_metrics.csv').open()));allstats={}
for pop in ['script_core','script_unseen','expert_core']:
    allstats[pop]={}
    for h in [32,64,128,256]:
        allstats[pop][str(h)]={m:stats([float(x['q_mae_deg']) for x in metrics if x['population']==pop and int(x['horizon'])==h and x['model']==m]) for m in NAMES[1:]}
(OUT/'long_all_response_statistics.json').write_text(json.dumps(allstats,indent=2))
# Diagnostic case rules use only real trajectory/control/contact; never WM errors.
extra=[]
for name,pool in [
 ('own_contact_at_pulse',[x for x in rows if x['set']=='expert512' and x['clear32'] and x['valid_steps']>=128 and x['own_contact_pulse_steps']>=4]),
 ('persistent_target',[x for x in rows if x['population']=='expert_core' and x['clear32'] and x['valid_steps']==256 and not x['q_return_start'] and x['true_target_last8_max_deg']>.5]),
 ('equal_target_unsettled',[x for x in rows if x['population']=='expert_core' and x['clear32'] and x['valid_steps']>=128 and not x['q_return_start'] and x['true_target_last8_max_deg']<.1])]:
    if pool:
        med=np.median([x['true_peak_first32_deg'] for x in pool]);chosen=min(pool,key=lambda x:(abs(x['true_peak_first32_deg']-med),x['set'],x['lane']));extra.append({**chosen,'diagnostic_type':name,'selection_pool':len(pool)})
(OUT/'long_extra_examples.json').write_text(json.dumps(extra,indent=2,default=lambda x:x.item()))
# Plot focused examples with explicit contact timing and all five absolute response curves.
sys.path.insert(0,str(ROOT/'audit-lib/plots'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
colors=['black','#2378B5','#E08A18','#23925C','#9954B0'];table=[]
examples=json.loads((OUT/'long_examples.json').read_text())+extra
for c in examples:
    label=c['set'];i=c['lane'];n=c['valid_steps'];x=np.arange(1,n+1);d=curves[label]
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for ax,key,ylabel in zip(axes,['q','qd','target'],['Extra angle (degrees)','Extra velocity (degrees/s)','Extra PD target (degrees)']):
        for m,col in zip(NAMES,colors):ax.plot(x,d['models'][m][key][:n,i],label=m,color=col,lw=2 if m=='PhysX' else 1.1)
        ax.axvspan(1,8,color='orange',alpha=.18);ax.axhline(0,color='grey',lw=.5);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.set_xlim(1,n)
    axes[0].axhspan(-c['near_threshold_deg'],c['near_threshold_deg'],color='green',alpha=.1);axes[0].legend(ncol=5)
    axes[0].set_title(f"{label} / {c['motion']} / {c['joint']} / lane {i}\n{c.get('diagnostic_type',c['contact_category'])}; valid through {n}")
    axes[2].set_xlabel('Control step (30 Hz); extra command withdrawn at step 9');fig.tight_layout()
    fig.savefig(OUT/f'focus_{label}_{i:03d}.png',dpi=150);plt.close(fig)
    for t in [1,4,8,16,24,32,64,128,256]:
        if t>n:continue
        table.append({'set':label,'lane':i,'joint':c['joint'],'step':t,**{m:float(d['models'][m]['q'][t-1,i]) for m in NAMES},'PhysX_PD_target_delta_deg':float(d['models']['PhysX']['target'][t-1,i])})
write_csv(OUT/'long_example_timepoints.csv',table)
# Detailed pilot statistics: true first-terminal prefixes, action excitation and independent contact episodes.
pilot={}
for mode in ['script','expert','expert_early','expert_early_strict','expert_early_explore']:
    path=OUT/f'coverage_{mode}.pt'
    if not path.exists():continue
    d=torch.load(path,weights_only=False);selected=d['selected'];valid=selected&~d['done'];N=d['frame'].shape[1]
    phase=(d['frame'].float()/d['seq_len'][None,:]*4).long().clamp(0,3);act=d['action'][selected]
    ev=torch.linalg.eigvalsh(torch.cov(act.T)).clamp_min(0);probs=ev/ev.sum();rank=float(torch.exp(-(probs*probs.clamp_min(1e-15).log()).sum()))
    cont=d['sn'][...,287:].bool().any(-1).numpy();ep=d['episode'].numpy();sel=selected.numpy();events=0;eps=set();runs8=0
    for lane in range(N):
        gap=999;run=0;prev_ep=-1
        for t in range(len(ep)):
            if ep[t,lane]!=prev_ep:gap=999;run=0;prev_ep=ep[t,lane]
            if not sel[t,lane]:continue
            if cont[t,lane]:
                if gap>=4:events+=1;eps.add((lane,int(ep[t,lane])))
                gap=0;run+=1
                if run==8:runs8+=1
            else:gap+=1;run=0
    rows_ep=list(csv.DictReader((OUT/f'pilot_{mode}_episodes.csv').open()))
    q0err=None
    if mode.startswith('expert_early'):
        errs=[]
        for lane in range(N):
            ik=np.load(ROOT/f'sources/GALATEA/rl/data/power_drill/ik/{d["ids"][lane%30]}_fr3_retarget.npz')
            errs.append(float(np.max(np.abs(d['s'][0,lane,:7].numpy()-ik['arm_qpos_reset'][0]))))
        q0err=max(errs)
    pilot[mode]={'nonterminal_transitions':int(valid.sum()),'nonterminal_quartile_counts':torch.bincount(phase[valid],minlength=4).tolist(),
          'action_cov_effective_rank':rank,'mean_hand_action_std':float(act[:,7:].std(0).mean()),
          'action_saturation_fraction':float((act.abs()>.99).float().mean()),
          'contact_events_gap_at_least4':events,'contact_episodes_with_gap4_events':len(eps),'contact_runs8':runs8,
          'frame0_successes':sum(r['start_frame']=='0' and 'timeline_success' in r['reasons'] for r in rows_ep),
          'frame0_arm_clamp_max_change_rad':q0err,
          'stale_reset_first_step_excluded':int(d['reset_input'].sum()),'no_nonfinite':bool(torch.isfinite(d['sn']).all())}
(OUT/'pilot_quality_supplement.json').write_text(json.dumps(pilot,indent=2))
print(json.dumps({'additional_examples':[(x['diagnostic_type'],x['set'],x['lane']) for x in extra],'pilot_quality':pilot},indent=2),flush=True)

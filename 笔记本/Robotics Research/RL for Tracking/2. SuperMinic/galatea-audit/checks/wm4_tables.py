"""Export the completed fixed experiment; never trains or selects a checkpoint."""
import csv,json
from pathlib import Path
import torch
from wm4_eval import cluster_ci

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/wm4_200k';OUT=EXP/'local_run'


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    result=json.loads((OUT/'results.json').read_text());ordinary=[];response=[];coverage=[]
    for group in ('3','12','20','30'):
        data=json.loads((OUT/f'train_{group}.json').read_text())
        train=json.loads((OUT/f'wm_{group}/summary.json').read_text())
        coverage.append({'motions':group,'train_transitions':data['transitions'],
            'raw_physx_transitions':data['raw_simulated_transitions'],'transitions_per_motion_mean':200000/int(group),
            'contact_episodes':data['contact_episodes'],**data['contact_phases'],
            'phase_quartiles':json.dumps(data['reference_phase_quartile_counts']),
            'action_covariance_effective_rank':data['action_covariance_effective_rank'],
            'collection_seconds':data['seconds'],'wm_seconds':train['training_seconds'],
            'updates':train['updates'],'last_training_loss':train['last_loss']['loss'],
            'collection_gpu_peak_mib':data['gpu_total_peak_sampled_mib'],
            'training_gpu_peak_mib':train['gpu_total_peak_sampled_mib']})
        for split in ('core','unseen'):
            for h,metrics in result[group][split]['ordinary'].items():
                ordinary.append({'motions':group,'split':split,'horizon':h,
                    **{k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}})
            for scope in ('before_task_termination','all_physical_rollouts'):
                for h,metrics in result[group][split]['response'][scope].items():
                    response.append({'motions':group,'split':split,'horizon':h,'scope':scope,
                        **{k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}})
    write_csv(EXP/'prediction_metrics.csv',ordinary);write_csv(EXP/'action_response_metrics.csv',response)
    write_csv(EXP/'data_coverage_resources.csv',coverage)
    cache=torch.load(OUT/'evaluation_predictions.pt',weights_only=False)
    per_motion=[];contact_subset=[]
    for split in ('core','unseen'):
        raw=torch.load(OUT/f'test_{split}.pt',weights_only=False);n=raw['s'].shape[1]
        for group in ('3','12','20','30'):
            for h in ('1','8','32'):
                cases=cache[group][split]['ordinary'][h]
                m=(cases['indices']%n)%len(raw['ids'])
                for i,name in enumerate(raw['ids']):
                    mask=m==i
                    per_motion.append({'motions':group,'split':split,'horizon':h,'test_motion':name,'n':int(mask.sum()),
                        **{k:float(v[mask].mean()) if mask.any() else None for k,v in cases['case_errors'].items()}})
                for contact in (False,True):
                    mask=cases['truth'][:,287:].bool().any(-1)==contact
                    contact_subset.append({'motions':group,'split':split,'horizon':h,'true_contact':contact,'n':int(mask.sum()),
                        **{k:float(v[mask].mean()) if mask.any() else None for k,v in cases['case_errors'].items()}})
    write_csv(EXP/'per_motion_metrics.csv',per_motion)
    write_csv(EXP/'contact_subset_metrics.csv',contact_subset)
    comparisons={}
    for split in ('core','unseen'):
        comparisons[split]={}
        for h in ('1','8','32'):
            base=cache['3'][split]['ordinary'][h]
            comparisons[split][h]={}
            for group in ('12','20','30'):
                other=cache[group][split]['ordinary'][h]
                assert torch.equal(base['indices'],other['indices'])
                delta={k:other['case_errors'][k]-v for k,v in base['case_errors'].items()}
                comparisons[split][h][group]={'mean_error_difference_vs_3':{k:float(v.mean()) for k,v in delta.items()},
                    'episode_bootstrap_95ci':cluster_ci(delta,base['episode'])}
    (EXP/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    try:
        import matplotlib
    except ImportError:
        print('CSV tables and paired comparisons complete. Optional plot skipped: matplotlib unavailable.')
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    specs=[('hand_q_mae_mrad','Finger joint MAE (mrad)'),('tip_position_mean_mm','Fingertip error (mm)'),
           ('object_position_mm','Object position error (mm)')]
    groups=['3','12','20','30'];x=list(range(4))
    for row,split in enumerate(('core','unseen')):
        for col,(key,title) in enumerate(specs):
            ax=axes[row,col]
            for h,style in [('8','o-'),('32','s--')]:
                ax.plot(x,[result[g][split]['ordinary'][h][key] for g in groups],style,label=h+' steps')
                ax.axhline(result['3'][split]['ordinary'][h]['persistence'][key],linestyle=':',alpha=.4)
            ax.set_xticks(x,groups);ax.set_xlabel('Training reference motions')
            ax.set_title(('Core 3: ' if row==0 else 'Unseen 5: ')+title)
            ax.grid(alpha=.2);ax.legend()
    fig.suptitle('200,000 training transitions per group; one training seed\nDotted lines: persistence baselines; only windows before reset')
    fig.tight_layout();fig.savefig(EXP/'wm4_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({'coverage':coverage,'ordinary':ordinary,'response':response}))


if __name__=='__main__':main()

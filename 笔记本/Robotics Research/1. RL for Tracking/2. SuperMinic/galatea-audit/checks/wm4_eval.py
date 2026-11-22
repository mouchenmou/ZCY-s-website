"""One final locked test, shared episodes and counterfactuals across four WMs."""
import json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from wm4_model import WorldModel, control, predict, advance_history

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/wm4_200k/local_run'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def mean(x):return float(x.mean()) if x.numel() else None


def quat_error(a,b):
    a=torch.nn.functional.normalize(a,dim=-1);b=torch.nn.functional.normalize(b,dim=-1)
    return 2*torch.acos((a*b).sum(-1).abs().clamp(0,1))*180/np.pi


def metrics(pred,true,rew=None,true_rew=None,prior_rew=None,tips=None):
    tip=pred[:,71:155].reshape(-1,28,3)[:,tips]
    ttip=true[:,71:155].reshape(-1,28,3)[:,tips]
    row={'hand_q_mae_mrad':(pred[:,7:29]-true[:,7:29]).abs().mean(-1)*1000,
         'arm_q_mae_mrad':(pred[:,:7]-true[:,:7]).abs().mean(-1)*1000,
         'hand_qd_mae_rad_s':(pred[:,36:58]-true[:,36:58]).abs().mean(-1),
         'tip_position_mean_mm':(tip-ttip).norm(dim=-1).mean(-1)*1000,
         'object_position_mm':(pred[:,58:61]-true[:,58:61]).norm(dim=-1)*1000,
         'object_rotation_deg':quat_error(pred[:,61:65],true[:,61:65]),
         'object_velocity_m_s':(pred[:,65:68]-true[:,65:68]).norm(dim=-1),
         'contact_brier':(pred[:,287:]-true[:,287:]).square().mean(-1)}
    answer={k:mean(v) for k,v in row.items()}
    pc=pred[:,287:]>.5;tc=true[:,287:]>.5
    tp=int((pc&tc).sum());fp=int((pc&~tc).sum());fn=int((~pc&tc).sum())
    answer.update({'n':len(pred),'nonfinite_rows':int((~torch.isfinite(pred).all(-1)).sum()),
                   'contact_f1':2*tp/max(1,2*tp+fp+fn),'contact_precision':tp/max(1,tp+fp),
                   'contact_recall':tp/max(1,tp+fn),'contact_prevalence':float(tc.float().mean()),
                   'contact_tp_fp_fn':[tp,fp,fn]})
    if rew is not None:
        row['reward_mae']=(rew-true_rew).abs();answer['reward_mae']=mean(row['reward_mae'])
        delta=true_rew-prior_rew;mask=delta.abs()>1e-3
        answer['reward_temporal_direction_n']=int(mask.sum())
        answer['reward_temporal_direction_accuracy']=mean(((rew-prior_rew)[mask].sign()==delta[mask].sign()).float())
    return answer,row


def cluster_ci(rows,episode):
    """Episode bootstrap: overlapping windows are not treated as independent."""
    ep=episode.cpu().numpy();unique,inv=np.unique(ep,return_inverse=True)
    if len(unique)<2:return {}
    counts=np.bincount(inv);rng=np.random.default_rng(1901)
    draw=rng.integers(0,len(unique),(400,len(unique)))
    ans={}
    for k,v in rows.items():
        vals=v.detach().cpu().numpy()
        sums=np.bincount(inv,weights=vals)
        boot=sums[draw].sum(1)/counts[draw].sum(1)
        ans[k]=[float(x) for x in np.percentile(boot,[2.5,97.5])]
    return ans


@torch.no_grad()
def rollout(model,norm,d,idx,n,h,c):
    s=d['s'][idx];prev=d['previous'][idx];last=d['last'][idx]
    mp=d['mask_previous'][idx];hist=d['history'][idx]
    states=[];rewards=[]
    for k in range(h):
        j=idx+k*n;act=d['action'][j]
        target=control(s,act,prev,last,mp,d['delay'][j],d['mask'][j],c)
        s,r,_=predict(model,norm,s,act,target,prev,last,mp,hist,d['ref'][j],d['physics'][j],d['age'][j])
        states.append(s);rewards.append(r)
        prev=target;last=act;mp=target[:,7:];hist=advance_history(hist,s)
    return torch.stack(states),torch.stack(rewards)


def flatten(raw):return {k:v.flatten(0,1).to('cuda') for k,v in raw.items() if isinstance(v,torch.Tensor) and k not in ('warmup','initial_done')}


def test_windows(raw,h):
    selected=raw['selected'];ok=selected.clone()
    for j in range(1,h):
        ok[:-j]&=selected[j:]&(raw['episode'][:-j]==raw['episode'][j:]);ok[-j:]=False
    idx=ok.flatten().nonzero().flatten()
    g=torch.Generator().manual_seed(1900+h)
    return idx[torch.randperm(len(idx),generator=g)[:2048]].to('cuda'),int(ok.sum())


def ordinary(model,norm,raw):
    d=flatten(raw);n=raw['s'].shape[1];c=raw['contract'];result={};cache={}
    for h in (1,8,32):
        idx,available=test_windows(raw,h)
        if not len(idx):result[str(h)]={'n':0,'available_windows':available};continue
        predictions=[];rewards=[]
        for start in range(0,len(idx),256):
            s,r=rollout(model,norm,d,idx[start:start+256],n,h,c)
            predictions.append(s[-1]);rewards.append(r[-1])
        pred=torch.cat(predictions);r=torch.cat(rewards);j=idx+(h-1)*n
        true=d['sn'][j];prior=d['reward'][idx-n];truth_reward=d['reward'][j]
        values,rows=metrics(pred,true,r,truth_reward,prior,c['tip_body_indices'])
        values['available_windows']=available;values['episode_bootstrap_95ci']=cluster_ci(rows,d['episode'][idx])
        values['test_episodes']=int(d['episode'][idx].unique().numel())
        persist=d['s'][idx].clone()
        cv=persist.clone();dt=h/30
        cv[:,:29]+=persist[:,29:58]*dt;cv[:,58:61]+=persist[:,65:68]*dt
        cv[:,71:155]+=persist[:,155:239]*dt
        values['persistence']=metrics(persist,true,prior,truth_reward,prior,c['tip_body_indices'])[0]
        values['constant_velocity']=metrics(cv,true,prior,truth_reward,prior,c['tip_body_indices'])[0]
        result[str(h)]=values
        cache[str(h)]={'indices':idx.cpu(),'prediction':pred.cpu(),'truth':true.cpu(),
                        'reward_prediction':r.cpu(),'reward_truth':truth_reward.cpu(),
                        'episode':d['episode'][idx].cpu(),'case_errors':{k:v.cpu() for k,v in rows.items()}}
    return result,cache


def response_stats(ps,ms,pr,mr,plus,minus,base,repeat,h,case_mask):
    n=ps.shape[1];lane=torch.arange(n);joint=7+lane%22
    truth=(plus['sn'][h-1,lane,joint]-minus['sn'][h-1,lane,joint])
    prediction=ps[h-1,lane,joint]-ms[h-1,lane,joint]
    noise=(base['sn'][h-1,lane,joint]-repeat['sn'][h-1,lane,joint]).abs()
    live=case_mask&(truth.abs()>torch.maximum(noise*10,torch.full_like(noise,1e-5)))
    ratio=prediction[live].abs()/truth[live].abs()
    rtruth=plus['reward'][h-1]-minus['reward'][h-1]
    rpred=pr[h-1]-mr[h-1]
    rnoise=(base['reward'][h-1]-repeat['reward'][h-1]).abs()
    rmask=case_mask&(rtruth.abs()>torch.maximum(rnoise*10,torch.full_like(rnoise,1e-4)))
    objtrue=plus['sn'][h-1,:,58:61]-minus['sn'][h-1,:,58:61]
    objpred=ps[h-1,:,58:61]-ms[h-1,:,58:61]
    objmask=case_mask&(objtrue.norm(dim=-1)>1e-4)
    ans={'cases':int(case_mask.sum()),'resolvable_joint_response_cases':int(live.sum()),
         'joint_direction_accuracy':mean((prediction[live].sign()==truth[live].sign()).float()),
         'joint_amplitude_ratio_median':float(ratio.median()) if len(ratio) else None,
         'joint_amplitude_ratio_q25_q75':torch.quantile(ratio,torch.tensor([.25,.75])).tolist() if len(ratio) else None,
         'joint_response_mae_mrad':mean((prediction[live]-truth[live]).abs()*1000),
         'true_joint_response_median_mrad':float(truth[live].abs().median()*1000) if live.any() else None,
         'reward_direction_n':int(rmask.sum()),
         'reward_direction_accuracy':mean((rpred[rmask].sign()==rtruth[rmask].sign()).float()),
         'reward_difference_mae':mean((rpred[rmask]-rtruth[rmask]).abs()),
         'object_response_n':int(objmask.sum()),
         'object_response_direction_accuracy':mean(((objpred[objmask]*objtrue[objmask]).sum(-1)>0).float()),
         'object_response_amplitude_ratio_median':float((objpred[objmask].norm(dim=-1)/objtrue[objmask].norm(dim=-1)).median()) if objmask.any() else None}
    return ans


def probes(model,norm,data):
    base,repeat,plus,minus=(data[k] for k in ('base','repeat','plus','minus'))
    # Independent fresh simulator construction with identical complete prefix.
    for x in (repeat,plus,minus):
        if not torch.equal(base['warmup'],x['warmup']):raise RuntimeError('Counterfactual prefix differs; cannot claim matched history')
        if not torch.equal(base['delay'],x['delay']) or not torch.equal(base['mask'],x['mask']):
            raise RuntimeError('Counterfactual control randomness differs')
    n=base['s'].shape[1];idx=torch.arange(n,device='cuda')
    ps,pr=rollout(model,norm,flatten(plus),idx,n,32,base['contract'])
    ms,mr=rollout(model,norm,flatten(minus),idx,n,32,base['contract'])
    ps=ps.cpu();ms=ms.cpu();pr=pr.cpu();mr=mr.cpu()
    result={'same_complete_prefix':True,'repeat_state_max_abs':float((base['sn']-repeat['sn']).abs().max()),
            'repeat_q_max_abs':float((base['sn'][...,:29]-repeat['sn'][...,:29]).abs().max()),
            'repeat_reward_max_abs':float((base['reward']-repeat['reward']).abs().max()),
            'all_physical_rollouts':{},'before_task_termination':{}}
    for h in (1,8,32):
        allcases=torch.ones(n,dtype=torch.bool)
        valid=~(plus['initial_done']|minus['initial_done'])
        if h>1:valid&=~(plus['done'][:h-1]|minus['done'][:h-1]).any(0)
        for key,mask in [('all_physical_rollouts',allcases),('before_task_termination',valid)]:
            result[key][str(h)]=response_stats(ps,ms,pr,mr,plus,minus,base,repeat,h,mask)
    return result,{'plus_state':ps,'minus_state':ms,'plus_reward':pr,'minus_reward':mr}


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    start=time.monotonic();result={};cached={};hashes={}
    raw_tests={s:torch.load(OUT/f'test_{s}.pt',weights_only=False) for s in ('core','unseen')}
    raw_probes={s:{b:torch.load(OUT/f'probe_{s}_{b}.pt',weights_only=False) for b in ('base','repeat','plus','minus')} for s in ('core','unseen')}
    for path in list(OUT.glob('test_*.pt'))+list(OUT.glob('probe_*.pt')):hashes[path.name]=digest(path)
    initial=[]
    for g in ('3','12','20','30'):
        ck=torch.load(OUT/f'wm_{g}/final.pt',map_location='cuda',weights_only=False)
        assert ck['step']==5000 and ck['train_transitions']==200000
        initial.append(ck['initial_weights_sha256'])
        model=WorldModel(ck['input_dim'],ck['hidden']).cuda();model.load_state_dict(ck['model']);model.eval()
        result[g]={};cached[g]={}
        for split in ('core','unseen'):
            regular,reg_cache=ordinary(model,ck['norm'],raw_tests[split])
            response,resp_cache=probes(model,ck['norm'],raw_probes[split])
            result[g][split]={'ordinary':regular,'response':response}
            cached[g][split]={'ordinary':reg_cache,'response':resp_cache}
        print(json.dumps({'group_complete':g,'seconds':time.monotonic()-start}),flush=True)
    assert len(set(initial))==1,'Group initialization is not identical'
    result['audit']={'same_initial_weights':True,'initial_weights_sha256':initial[0],
                     'test_data_sha256':hashes,'seconds':time.monotonic()-start,
                     'random_context':'recorded exogenous delay and mask outcomes; future PD targets recomputed from predicted q and control history',
                     'probe_scope':'all_physical_rollouts continue physics after official done without reset; task-valid results reported separately',
                     'normalization':'training only; no test fit, no tuning or checkpoint selection'}
    # Allow no NaN JSON: a divergence is a test failure, never silently dropped.
    text=json.dumps(result,indent=2,allow_nan=False)
    (OUT/'results.json').write_text(text)
    torch.save(cached,OUT/'evaluation_predictions.pt')
    print('All four models evaluated on identical independent test data.',flush=True)


if __name__=='__main__':main()

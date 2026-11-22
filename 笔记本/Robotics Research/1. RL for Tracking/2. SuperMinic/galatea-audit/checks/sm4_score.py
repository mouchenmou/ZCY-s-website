"""Frozen-WM errors on identical initial-policy PhysX data; no optimization."""
from pathlib import Path
import json,time
from isaacgym import gymapi
import torch
from sm4_model import Actor,WorldModel,control,advance_history,digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/supermimic4'
torch.set_num_threads(4)

def metric(pred,true,reward,actual):
    return {'q_deg_mse':float(((pred[:,:29]-true[:,:29])*180/torch.pi).square().mean()),
            'finger_q_deg_mse':float(((pred[:,7:29]-true[:,7:29])*180/torch.pi).square().mean()),
            'qd_mse':float((pred[:,29:58]-true[:,29:58]).square().mean()),
            'body_xyz_mse':float((pred[:,71:155]-true[:,71:155]).square().mean()),
            'object_xyz_mse':float((pred[:,58:61]-true[:,58:61]).square().mean()),
            'contact_brier':float((pred[:,287:292]-true[:,287:292]).square().mean()),
            'force_active_brier':float((pred[:,292:297]-true[:,292:297]).square().mean()),
            'reward_mae':float((reward-actual).abs().mean())}

@torch.no_grad()
def run():
    start=time.monotonic();result={};provenance={}
    for g in ['3','12','20','30']:
        ck=torch.load(OUT/'smoke'/('group_'+g)/'final.pt',map_location='cpu',weights_only=False)
        wm=WorldModel().cuda();wm.load_state_dict(ck['wm']);c=ck['contract'];del ck
        rows=[]
        for seed in [11401,11402]:
            d=torch.load(OUT/'evaluation'/f'initial_{seed}'/'common_physx.pt',map_location='cpu',weights_only=False)
            for split,condition in [('core_seen_all',d['motion']<3),('heldout_unseen_all',d['motion']>=3)]:
                choose=(d['selected']&condition).nonzero();summed={};n=0
                for index in choose.split(1024):
                    r={k:v[index[:,0],index[:,1]].cuda() for k,v in d.items()}
                    sn,rp,_,_=wm(r['s'],r['target'],r['action'],r['last'],r['history'],r['ref'],r['physics'],r['age'])
                    vals=metric(sn,r['sn'],rp,r['reward']);count=len(index);n+=count
                    for k,v in vals.items():summed[k]=summed.get(k,0.)+v*count
                rows.append({'seed':seed,'split':split,'horizon':1,'n':n,**{k:v/max(1,n) for k,v in summed.items()}})
                ok=d['selected']&condition
                ok=ok.clone()
                for j in range(1,8):
                    ok[:-j]&=d['selected'][j:]&~d['done'][j-1:-1];ok[-j:]=False
                starts=ok.nonzero()
                rng=torch.Generator().manual_seed(11499)
                if len(starts)>512:starts=starts[torch.randperm(len(starts),generator=rng)[:512]]
                if len(starts):
                    t,lane=starts.T;r={k:v[t,lane].cuda() for k,v in d.items()}
                    s=r['s'];prev=r['previous'];last=r['last'];mp=r['mask_previous'];hist=r['history']
                    for j in range(8):
                        r={k:v[t+j,lane].cuda() for k,v in d.items()}
                        target=control(s,r['action'],prev,last,mp,r['delay'],r['mask'],c)
                        s,rp,_,_=wm(s,target,r['action'],last,hist,r['ref'],r['physics'],r['age'])
                        prev=target;mp=target[:,7:];last=r['action'];hist=advance_history(hist,s)
                    vals=metric(s,r['sn'],rp,r['reward'])
                    rows.append({'seed':seed,'split':split,'horizon':8,'n':len(starts),**vals})
                else:rows.append({'seed':seed,'split':split,'horizon':8,'n':0})
        result[g]=rows;del wm
        replay=torch.load(OUT/'smoke'/('group_'+g)/'replay.pt',map_location='cpu',weights_only=False)
        provenance[g]={}
        for name,version in [('initial',0),('final',28)]:
            ck=torch.load(OUT/'smoke'/('group_'+g)/(name+'.pt'),map_location='cpu',weights_only=False)
            actor=Actor().cuda();actor.load_state_dict(ck['actor']);del ck
            pick=replay['policy_version']==version
            obs=replay['obs'][pick];mus=replay['actor_mean'][pick];err=0.
            for begin in range(0,len(obs),512):
                pred=actor(obs[begin:begin+512].cuda()).cpu()
                err=max(err,float((pred-mus[begin:begin+512]).abs().max()))
            assert err<3e-6,'Replay did not come from the recorded current Actor'
            provenance[g][name]={'version':version,'rows':len(obs),'max_action_mean_error':err,'actor_hash':digest(actor)}
            del actor
        del replay
    (OUT/'common_wm_metrics.json').write_text(json.dumps({'seconds':time.monotonic()-start,'groups':result,'replay_actor_recomputation':provenance},indent=2))
    print(json.dumps({'seconds':time.monotonic()-start,'status':'common_WM_evaluation_complete'}),flush=True)

if __name__=='__main__':run()

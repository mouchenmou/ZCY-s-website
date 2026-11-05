"""CPU-only frozen-weight checks of reward contact history and zero-effect actions."""
import json,sys,math
from pathlib import Path
import torch
import numpy as np
from wm4_model import WorldModel,predict,control,advance_history
from wm4_time_response import stats,sha
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/wm4_200k';OUT=EXP/'followup'
torch.set_num_threads(4)
results={};hashes={g:sha(EXP/f'local_run/wm_{g}/final.pt') for g in ['3','12','20','30']}
with torch.no_grad():
 for g in ['3','12','20','30']:
    ck=torch.load(EXP/f'local_run/wm_{g}/final.pt',map_location='cpu',weights_only=False);m=WorldModel(ck['input_dim'],ck['hidden']);m.load_state_dict(ck['model']);m.eval()
    history_true=[];history_pred=[];false_response=[];true_zero=[];contact_probs=[]
    for split in ['core','unseen']:
        d={k:torch.load(EXP/f'local_run/probe_{split}_{k}.pt',map_location='cpu',weights_only=False) for k in ['base','plus']}
        preds={}
        for k,b in d.items():
            s=b['s'][0];target=control(s,b['action'][0],b['previous'][0],b['last'][0],b['mask_previous'][0],b['delay'][0],b['mask'][0],b['contract'])
            pred,_,_=predict(m,ck['norm'],s,b['action'][0],target,b['previous'][0],b['last'][0],b['mask_previous'][0],b['history'][0],b['ref'][0],b['physics'][0],b['age'][0]);preds[k]=pred
            valid=~(b['initial_done']|b['done'][0])
            history_true.extend((b['sn'][0,valid,243:258].reshape(-1,5,3).norm(dim=-1)>0).float().mean(-1).tolist())
            history_pred.extend((pred[valid,243:258].reshape(-1,5,3).norm(dim=-1)>0).float().mean(-1).tolist())
            contact_probs.extend(pred[valid,287:].mean(-1).tolist())
        base,plus=d['base'],d['plus'];joint=7+torch.arange(len(base['initial_done']))%22
        valid=~(base['initial_done']|plus['initial_done']|base['done'][0]|plus['done'][0]);zero=valid&((base['target'][0]-plus['target'][0]).abs().amax(-1)<1e-7)
        idx=zero.nonzero().flatten();false_response.extend(((preds['plus']-preds['base'])[idx,joint[idx]].abs()*180/math.pi).tolist())
        true_zero.extend(((plus['sn'][0]-base['sn'][0])[idx,joint[idx]].abs()*180/math.pi).tolist())
    results[g]={'valid_branch_states':len(history_true),'true_fraction_fingers_force_gt0_mean':float(np.mean(history_true)),
       'predicted_fraction_fingers_force_gt0_mean':float(np.mean(history_pred)),
       'contact_head_mean_probability':float(np.mean(contact_probs)),
       'zero_target_cases':len(false_response),'false_q_response_deg':stats(false_response),'true_q_response_deg':stats(true_zero)}
assert all(sha(EXP/f'local_run/wm_{g}/final.pt')==h for g,h in hashes.items())
(OUT/'additional_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

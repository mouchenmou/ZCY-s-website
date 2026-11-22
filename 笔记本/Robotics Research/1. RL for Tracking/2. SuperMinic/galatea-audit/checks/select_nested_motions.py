"""Freeze nested training IDs using only the locked 30 training candidates.

Contact features are MANO-reference proximity, NOT observed PhysX pad contact.
No held-out reference file is opened. These feature statistics are for selection,
never for normalizing a WM or choosing hyperparameters from test performance.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources/manotorch'))
from manotorch.manolayer import ManoLayer
OUT=ROOT/'experiments/wm4_200k'
lock=json.loads((OUT/'split_lock.json').read_text())
ids=lock['train_candidate_ids']
assert not set(ids)&set(lock['test_motion_ids'])
model=ManoLayer(side='right',rot_mode='axisang',use_pca=False,flat_hand_mean=True,
    mano_assets_root=str(ROOT/'assets/licensed/mano_v1_2')).cuda()
# Exact official fingertip vertex order: thumb,index,middle,pinky,ring.
tips=[766,353,467,695,576]
records=[]; features=[]; hashes={}
for name in ids:
    base=ROOT/'assets/power_drill/data'/name
    raw=np.load(base/'motion.npy',allow_pickle=True).item()
    ik=np.load(ROOT/'assets/power_drill/ik'/(name+'_fr3_retarget.npz'))
    t=len(raw['object_transl']); idx=np.unique(np.linspace(0,t-1,128).astype(int))
    pose=np.concatenate([raw['rhand_global_orient'][idx],raw['rhand_hand_pose'][idx]],axis=1)
    with torch.no_grad():
        v=model(torch.tensor(pose,dtype=torch.float32,device='cuda'),
            torch.tensor(raw['rhand_betas'][idx],dtype=torch.float32,device='cuda')).verts
        v=v+torch.tensor(raw['rhand_transl'][idx],device='cuda',dtype=torch.float32)[:,None]
        rot=Rotation.from_rotvec(raw['object_global_orient'][idx]).as_matrix()
        # Official object transform is v @ R + translation.
        local=(v[:,tips]-torch.tensor(raw['object_transl'][idx],device='cuda',dtype=torch.float32)[:,None]) @ torch.tensor(rot.transpose(0,2,1),device='cuda',dtype=torch.float32)
        mesh=trimesh.load(base/'object.obj',process=False)
        np.random.seed(20261104)
        points,_=trimesh.sample.sample_surface(mesh,2048)
        dist=torch.cdist(local,torch.tensor(points,device='cuda',dtype=torch.float32)[None]).min(-1).values.cpu().numpy()
    near=dist<.02;close=dist<.005
    valid=np.asarray(ik['reset_collision_free'],bool)
    allowed=valid & (np.arange(t)<int(.98*t))
    # Equal-weight object/hand/contact-reset blocks after training-only scaling.
    obj=raw['object_transl'][idx];rotation=Rotation.from_rotvec(raw['object_global_orient'][idx])
    angular=(rotation[1:]*rotation[:-1].inv()).magnitude()
    block_obj=np.r_[np.ptp(obj,axis=0),np.sum(np.linalg.norm(np.diff(obj,axis=0),axis=1)),angular.sum(),
                    np.ptp(raw['rhand_transl'][idx]-obj,axis=0)]
    hand=raw['rhand_hand_pose'][idx]
    block_hand=np.r_[hand.mean(0),hand.std(0)]
    block_contact=np.r_[near.mean(0),close.mean(0),(near.sum(1)>=2).mean(),
        np.mean(near[1:]&near[:-1],axis=0),allowed.mean(),
        np.mean(near[allowed[idx]],axis=0) if allowed[idx].any() else np.zeros(5)]
    features.append([block_obj,block_hand,block_contact])
    records.append(dict(motion=name,frames=t,near_finger_fraction=float(near.mean()),
        close_finger_fraction=float(close.mean()),multi_finger_fraction=float((near.sum(1)>=2).mean()),
        valid_reset_fraction=float(allowed.mean()),object_path_m=float(block_obj[3]),
        object_rotation_path_rad=float(block_obj[4])))
    hashes[name]=hashlib.sha256((base/'motion.npy').read_bytes()).hexdigest()
blocks=[]
for b in range(3):
    x=np.stack([v[b] for v in features]);sd=x.std(0);sd[sd<1e-6]=1.
    blocks.append((x-x.mean(0))/sd/np.sqrt(x.shape[1]))
f=np.concatenate(blocks,axis=1)
selected=list(lock['core_motion_ids'])
trace=[]
while len(selected)<len(ids):
    selected_idx=[ids.index(x) for x in selected]
    candidates=[x for x in ids if x not in selected]
    scores={x:float(np.linalg.norm(f[ids.index(x)]-f[selected_idx],axis=1).min()) for x in candidates}
    pick=sorted(candidates,key=lambda x:(-scores[x],int(x[1:])))[0]
    trace.append({'added':pick,'min_distance':scores[pick]});selected.append(pick)
result={'groups':{str(n):selected[:n] for n in [3,12,20,30]},
    'heldout_ids':lock['test_motion_ids'],'ordering':selected,'motion_sha256':hashes,
    'rule':'128 phase samples; equal-weight standardized object, hand, and reference-contact/reset blocks; greedy farthest-first from g16/g25/g26; numeric-ID tie break',
    'contact_definition':'MANO fingertip distance to 2048 fixed-seed mesh samples: near <20mm, close <5mm; proxy, not PhysX contact',
    'selection_feature_fit_ids':ids,'trace':trace,'status':'ID freeze; pending actual PhysX coverage validation'}
dest=OUT/'nested_groups.json'
if dest.exists():assert json.loads(dest.read_text())==result,'Selection already frozen; do not overwrite'
else:dest.write_text(json.dumps(result,indent=2))
with (OUT/'selection_features.csv').open('w') as fp:
    writer=csv.DictWriter(fp,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
print(json.dumps(result['groups'],indent=2))

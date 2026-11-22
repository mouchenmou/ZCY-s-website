"""CPU geometric spot-check of official zero-finger reset poses; no asset writes.

Convex intersection via normalized half-space LP; positive radius is a verified
penetration witness. Separated shapes are NOT claimed to satisfy the 5 mm margin.
"""
import ast,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linprog, minimize
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/wm4_200k/followup';RL=ROOT/'sources/GALATEA/rl'
sys.path.insert(0,str(ROOT/'checks'));from wm4_time_response import write_csv
scope=globals()
source=ast.parse((RL/'preprocess/collision.py').read_text())
for node in source.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['_rpy','_T','_aa','parse_joints','fk_all','_origin']:
        exec(compile(ast.Module(body=[node],type_ignores=[]),'official_collision_extract','exec'),scope)
urdf=RL/'assets/robot/fr3_sharpa_right.urdf';joints=parse_joints(urdf)
root=sorted({j['parent'] for j in joints}-{j['child'] for j in joints})[0]
base=np.eye(4);base[:3,:3]=R.from_euler('z',-np.pi/2).as_matrix();base[:3,3]=[0,.95,0]
pieces={}
for link in ET.parse(urdf).getroot().findall('link'):
    if link.attrib['name'].startswith('fr3') or link.attrib['name']=='base':continue
    vals=[]
    for coll in link.findall('collision'):
        m=coll.find('geometry/mesh');v=np.asarray(trimesh.load(urdf.parent/m.attrib['filename'],force='mesh',process=False).vertices,dtype=float)
        v=v*np.fromstring(m.attrib.get('scale','1 1 1'),sep=' ');t=_origin(coll);v=v@t[:3,:3].T+t[:3,3]
        hull=ConvexHull(v);vals.append((v[hull.vertices],hull.equations))
    if vals:pieces[link.attrib['name']]=vals
ids=json.loads((OUT.parent/'nested_groups.json').read_text())['groups']['30'];rows=[]
# Transform exactly follows official scene.object_world / apply_arm_scene.
mr=R.from_euler('z',-np.pi/2).as_matrix()@R.from_euler('x',np.pi/2).as_matrix()@R.from_euler('x',-np.pi/2).as_matrix()@R.from_euler('z',np.pi/2).as_matrix()
mt=np.array([0,0,.415])+(R.from_euler('z',-np.pi/2).as_matrix()@R.from_euler('x',np.pi/2).as_matrix())@np.array([0,.018,0])
if '--g16-mid' in sys.argv:ids=['g16']
for mid in ids:
    ik=np.load(RL/f'data/power_drill/ik/{mid}_fr3_retarget.npz');folder=RL/f'data/power_drill/data/{mid}'
    if not folder.exists():
        manifest=json.loads((RL/'data/power_drill/data/manifest.json').read_text())
        folder=RL/'data/power_drill/data'/next(x['dir'] for x in manifest['entries'] if x['data_index']==mid)
    motion=np.load(folder/'motion.npy',allow_pickle=True).item();verts=np.asarray(trimesh.load(folder/'object.obj',force='mesh',process=False).vertices)
    oe=ConvexHull(verts).equations;seq=len(ik['arm_qpos'])
    for frac in ([.625] if '--g16-mid' in sys.argv else [0,.125,.375,.625,.875]):
        f=int(frac*seq);world=fk_all(root,base,joints,{f'fr3_joint{k+1}':v for k,v in enumerate(ik['arm_qpos_reset'][f])})
        rot=mr@R.from_rotvec(motion['object_global_orient'][f]).as_matrix().T
        pos=mr@motion['object_transl'][f]+mt
        theta=float(ik['corot_theta']);rz=R.from_euler('z',np.radians(theta)).as_matrix();piv=np.r_[ik['corot_pivot'].reshape(-1)[:2],0]
        pos=rz@(pos-piv)+piv;rot=rz@rot;pos[:2]+=ik['shift_xy'];pos[2]+=float(ik['deficit'][f])
        objn=oe[:,:3]@rot.T;objb=oe[:,3]-objn@pos
        ov=verts@rot.T+pos;omin=ov.min(0);omax=ov.max(0)
        best=-1.;bestlink=None;minz=1e9
        for link,ps in pieces.items():
            t=world[link]
            for v,e in ps:
                wv=v@t[:3,:3].T+t[:3,3];minz=min(minz,float(wv[:,2].min()))
                if (wv.max(0)<omin).any() or (wv.min(0)>omax).any():continue
                normals=e[:,:3]@t[:3,:3].T;offset=e[:,3]-normals@t[:3,3]
                n=np.r_[normals,objn];b=np.r_[offset,objb]
                res=linprog([0,0,0,-1],A_ub=np.c_[n,np.ones(len(n))],b_ub=-b,bounds=[(None,None)]*4,method='highs')
                if res.success and res.x[3]>best:best=float(res.x[3]);bestlink=link
        witness_distance=None
        if mid=='g16' and frac==.625 and bestlink:
            distances=[];t=world[bestlink]
            for v,e in pieces[bestlink]:
                normals=e[:,:3]@t[:3,:3].T;offset=e[:,3]-normals@t[:3,3]
                A=np.zeros((len(normals)+len(objn),6));A[:len(normals),:3]=normals;A[len(normals):,3:]=objn
                B=np.r_[offset,objb];x0=np.r_[(v@t[:3,:3].T+t[:3,3]).mean(0),ov.mean(0)]
                result=minimize(lambda x:.5*np.sum((x[:3]-x[3:])**2),x0,jac=lambda x:np.r_[x[:3]-x[3:],x[3:]-x[:3]],
                    constraints={'type':'ineq','fun':lambda x:-(A@x+B),'jac':lambda x:-A},method='SLSQP',options={'ftol':1e-13,'maxiter':300})
                if result.success and (A@result.x+B).max()<1e-8:distances.append(float(np.linalg.norm(result.x[:3]-result.x[3:])))
            if distances:witness_distance=min(distances)*1000
        rows.append({'motion':mid,'fraction':frac,'frame':f,'official_reset_valid':bool(ik['reset_collision_free'][f]),
            'object_hand_shared_interior_radius_mm':best*1000,'witness_link':bestlink,'hand_min_table_height_mm':(minz-.415)*1000,
            'verified_object_hand_intersection':best>.0002,'witness_link_distance_mm':witness_distance})
    print(mid,flush=True)
if '--g16-mid' in sys.argv:
    (OUT/'reset_g16_mid_clearance.json').write_text(json.dumps(rows,indent=2));print(rows);sys.exit(0)
write_csv(OUT/'reset_geometry_spot_checks.csv',rows)
summary={str(frac):{'n':sum(x['fraction']==frac for x in rows),'official_valid_n':sum(x['official_reset_valid'] for x in rows if x['fraction']==frac),
     'proven_object_hand_intersections':sum(x['verified_object_hand_intersection'] for x in rows if x['fraction']==frac)} for frac in [0,.125,.375,.625,.875]}
(OUT/'reset_geometry_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

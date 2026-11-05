"""Read checksum-verified official release assets; does not import simulation code."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
rows, raw_rows, summary = [], [], {}
for obj in ('power_drill', 'dog', 'old_camera2'):
    base = ROOT / 'assets' / obj
    entries = json.loads((base / 'data/manifest.json').read_text())['entries']
    for e in entries:
        d = base / 'data' / e.get('dir', e['data_index'])
        # Pickled dictionaries are loaded only from verified upstream assets.
        motion = np.load(d / 'motion.npy', allow_pickle=True).item()
        ik = np.load(base / 'ik' / (e['data_index'] + '_fr3_retarget.npz'))
        n = len(motion['object_global_orient'])
        length = min(n, len(ik['arm_qpos']), 2000)
        valid = np.asarray(ik['reset_collision_free'][:length], dtype=bool)
        allowed = valid & (np.arange(length) < max(1, int(length * .98)))
        rows.append(dict(object=obj, motion=e['data_index'], frames=n,
            ik_frames=len(ik['arm_qpos']), loaded_frames=length,
            file_span_at_30hz_s=(n-1)/30, loaded_span_at_30hz_s=(length-1)/30,
            derivative_span_at_60hz_s=(length-1)/60,
            valid_reset_frames=int(valid.sum()), sampled_reset_frames=int(allowed.sum()),
            mesh_sha256=hashlib.sha256((d/'object.obj').read_bytes()).hexdigest(),
            motion_sha256=hashlib.sha256((d/'motion.npy').read_bytes()).hexdigest(),
            assets_present=all((d/x).is_file() for x in ['object.obj','object_collision.obj','object_collision.urdf']),
            finite=all(np.isfinite(x).all() for x in motion.values() if isinstance(x,np.ndarray))))
    for f in sorted((base/'motion').glob('*.npz')):
        m = np.load(f,allow_pickle=True)
        n = int(m['n_frames']); fps = float(m['framerate'])
        contact = m['contact'].item()['object']
        contact = np.asarray(contact).astype(bool)
        raw_rows.append(dict(object=obj, source=f.stem, frames=n, fps=fps,
            duration_s=(n-1)/fps, contact_frames=int(contact.sum()),
            contact_onsets=int(np.count_nonzero(contact & ~np.concatenate([[False],contact[:-1]])))))
    rr = [r for r in rows if r['object']==obj]
    raw = [r for r in raw_rows if r['object']==obj]
    summary[obj] = dict(processed=len(rr), raw=len(raw),
        frames_min=min(r['frames'] for r in rr), frames_max=max(r['frames'] for r in rr),
        loaded_duration_s=sum(r['loaded_span_at_30hz_s'] for r in rr),
        file_duration_s=sum(r['file_span_at_30hz_s'] for r in rr),
        truncated=sum(r['loaded_frames']<r['frames'] for r in rr),
        mesh_hashes=len(set(r['mesh_sha256'] for r in rr)),
        motion_hashes=len(set(r['motion_sha256'] for r in rr)),
        reset_frames=sum(r['sampled_reset_frames'] for r in rr),
        raw_frames=dict(Counter(r['frames'] for r in raw)),
        raw_fps=sorted(set(r['fps'] for r in raw)),
        raw_duration_s=sum(r['duration_s'] for r in raw),
        all_assets_present=all(r['assets_present'] for r in rr),
        all_finite=all(r['finite'] for r in rr))
for name, data in [('motion_inventory',rows),('raw_inventory',raw_rows)]:
    with (ROOT/'evidence'/f'{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(ROOT/'evidence/inventory_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('Historical IDs in current release (identity with old data unverified):')
print(json.dumps([r for r in rows if r['object']=='power_drill' and r['motion'] in ['g16','g25','g26']],indent=2))

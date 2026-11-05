"""Single local, finite experiment. No cloud/remote calls and no Actor training."""
import json, os, sys, time, subprocess, shutil, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/wm4_200k';OUT=EXP/'local_run'
PY=ROOT/'venv/bin/python'


def main():
    start=time.monotonic();deadline=start+2700
    groups=json.loads((EXP/'nested_groups.json').read_text())
    config={'groups':groups['groups'],'heldout_ids':groups['heldout_ids'],
            'transitions_per_group':200000,'collector_envs':120,'collection_seed':20261105,
            'model_hidden':[2048,2048,2048],'activation':'SiLU','model_seed':45,
            'optimizer':'Adam','learning_rate':.0003,'batch':128,'updates':5000,'horizon':8,
            'normalization':'own selected training transitions only',
            'checkpoint_interval_updates':500,'max_seconds_total':2700,'max_gpu_gib':40,
            'max_experiment_gib':20,'min_free_disk_gib':20,
            'test_core_seed':8101,'test_unseen_seed':8102,'probe_core_seed':8201,'probe_unseen_seed':8202,
            'probe_envs_per_split':600,'probe_hand_delta':.2,'probe_pulse_steps':8,
            'evaluation_horizons':[1,8,32],'max_test_windows_per_split_per_horizon':2048,
            'external_backup_required':False,'actor_training':False,'response_loss':False,
            'official_source_modified':False,
            'method_status':'new engineering baseline, not recovered old WM or complete SuperMimic reproduction'}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'checks'/x for x in ('wm4_model.py','wm4_collect.py','wm4_train.py','wm4_probe.py')]}
    config['source_sha256']=hashes
    lock=EXP/'execution_lock.json'
    if lock.exists():raise RuntimeError('Execution already locked; no automatic rerun')
    lock.write_text(json.dumps(config,indent=2))
    event=open(OUT/'execution.jsonl','w',buffering=1)
    def emit(data):
        data['elapsed_seconds']=time.monotonic()-start
        event.write(json.dumps(data)+'\n');print(json.dumps(data),flush=True)
    def execute(name,script,args,limit):
        for filename,digest in hashes.items():
            if hashlib.sha256((ROOT/'checks'/filename).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Frozen collection/model/training source changed')
        emit({'event':'start','name':name})
        with (OUT/(name+'.log')).open('w') as log:
            p=subprocess.Popen([str(PY),str(ROOT/'checks'/script),*map(str,args)],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            began=time.monotonic();reason=None
            while p.poll() is None:
                elapsed=time.monotonic()-began
                size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
                if time.monotonic()>deadline or elapsed>limit:reason='wall time limit'
                elif size>20*2**30:reason='experiment disk limit'
                elif shutil.disk_usage(OUT).free<20*2**30:reason='disk reserve'
                try:
                    mem=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()[0])
                    if mem>40*1024:reason='GPU memory limit'
                except (subprocess.SubprocessError,ValueError):mem=None
                if reason:
                    import signal
                    os.killpg(p.pid,signal.SIGINT)
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise RuntimeError(reason+'; stopped, no budget expansion')
                if int(elapsed)%30<2:emit({'event':'running','name':name,'stage_seconds':elapsed,'gpu_mib':mem,'disk_gib':size/2**30})
                time.sleep(2)
            if p.returncode:raise RuntimeError(name+' failed; see local log')
        emit({'event':'complete','name':name,'stage_seconds':time.monotonic()-began})
    try:
        for count,ids in groups['groups'].items():
            path=OUT/f'train_{count}.pt'
            execute('collect_'+count,'wm4_collect.py',['--ids',','.join(ids),'--output',path,'--seconds',580],600)
            execute('train_'+count,'wm4_train.py',['--data',path,'--output',OUT/f'wm_{count}','--seconds',580],600)
        # First-ever access to heldout data for this round occurs after all four models finish.
        for split,ids,seed in [('core',groups['groups']['3'],8101),('unseen',groups['heldout_ids'],8102)]:
            execute('test_'+split,'wm4_collect.py',['--ids',','.join(ids),'--count',12000 if split=='core' else 20000,
                    '--seed',seed,'--output',OUT/f'test_{split}.pt'],240)
            for name,branch in [('base',0),('repeat',0),('plus',1),('minus',-1)]:
                args=['--ids',','.join(ids),'--seed',seed+100,'--envs',600,'--branch',branch,
                      '--output',OUT/f'probe_{split}_{name}.pt']
                if name!='base':args+=['--base',OUT/f'probe_{split}_base.pt']
                execute('probe_'+split+'_'+name,'wm4_probe.py',args,180)
        execute('evaluation','wm4_eval.py',[],240)
        emit({'event':'all_complete'})
    except BaseException as error:
        emit({'event':'stopped','reason':str(error)})
        raise
    finally:event.close()


if __name__=='__main__':main()

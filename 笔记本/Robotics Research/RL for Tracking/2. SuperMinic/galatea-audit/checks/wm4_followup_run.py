"""Bounded local diagnostic runner: never trains or uploads."""
import os,json,time,subprocess,signal,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/wm4_200k/followup'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(stage):
    if (OUT/"completion.json").exists():raise RuntimeError("This diagnostic is closed; no automatic further GPU runs")
    from wm4_budget import Budget
    budget=Budget(OUT/"budget_reservations.json")
    groups=json.loads((OUT.parent/'nested_groups.json').read_text())
    start=time.monotonic();jobs=[]
    if stage=='long':
        for split,ids,seed in [('core',groups['groups']['3'],8201),('unseen',groups['heldout_ids'],8202),('core2',groups['groups']['3'],8301),('unseen2',groups['heldout_ids'],8302)]:
            for branch,v in [('base',0),('repeat',0),('plus',1)]:
                args=['--ids',','.join(ids),'--seed',str(seed),'--branch',str(v),'--output',str(OUT/f'{split}_{branch}.pt')]
                if branch!='base':args+=['--base',str(OUT/f'{split}_base.pt')]
                jobs.append((f'{split}_{branch}','wm4_long_probe.py',args))
    elif stage=='expertprobe':
        for name,seed,warm in [('expert128',8401,128),('expert512',8402,512)]:
            for branch,v in [('base',0),('repeat',0),('plus',1)]:
                args=['--ids',','.join(groups['groups']['3']),'--seed',str(seed),'--branch',str(v),'--policy','expert','--warmup',str(warm),'--output',str(OUT/f'{name}_{branch}.pt')]
                if branch!='base':args+=['--base',str(OUT/f'{name}_base.pt')]
                jobs.append((f'{name}_{branch}','wm4_long_probe.py',args))
    elif stage=='repair_unseen2':
        jobs.append(('unseen2_plus','wm4_long_probe.py',['--ids',','.join(groups['heldout_ids']),'--seed','8302','--branch','1','--output',str(OUT/'unseen2_plus.pt'),'--base',str(OUT/'unseen2_base.pt')]))
    elif stage=='strictpilot':
        jobs.append(('coverage_expert_early_strict','wm4_coverage_collect.py',['--ids',','.join(groups['groups']['30']),'--mode','expert_early','--strict-hand-gate','--steps','1400','--envs','60','--output',str(OUT/'coverage_expert_early_strict.pt')]))
    elif stage=='explorepilot':
        jobs.append(('coverage_expert_early_explore','wm4_coverage_collect.py',['--ids',','.join(groups['groups']['30']),'--mode','expert_early','--strict-hand-gate','--exploration','0.05','--steps','1400','--envs','60','--output',str(OUT/'coverage_expert_early_explore.pt')]))
    elif stage=='coverage':
        for mode in ['script','expert','expert_early']:
            jobs.append(('coverage_'+mode,'wm4_coverage_collect.py',['--ids',','.join(groups['groups']['30']),'--mode',mode,'--steps','1400','--envs','60','--output',str(OUT/f'coverage_{mode}.pt')]))
    previous=sum(json.loads(l).get('seconds',0) for l in (OUT/'execution.jsonl').read_text().splitlines() if json.loads(l).get('event')=='complete') if (OUT/'execution.jsonl').exists() else 0
    weights={g:sha(OUT.parent/f'local_run/wm_{g}/final.pt') for g in ['3','12','20','30']}
    with (OUT/'execution.jsonl').open('a',buffering=1) as events:
        for name,script,args in jobs:
            if (OUT/f'{name}.json').exists():
                assert (OUT/f'{name}.pt').exists()
                continue
            allowed=budget.reserve(name,300)
            began=time.monotonic();peak=0
            with (OUT/f'{name}.log').open('w') as log:
                proc=subprocess.Popen(["timeout","--signal=INT","--kill-after=5",str(max(1,int(allowed)-7)),str(ROOT/'venv/bin/python'),str(ROOT/'checks'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while proc.poll() is None:
                    mem=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]);peak=max(peak,mem)
                    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
                    if time.monotonic()-began>allowed or previous+time.monotonic()-start>1200 or mem>16384 or size>8*2**30 or shutil.disk_usage(OUT).free<20*2**30:
                        os.killpg(proc.pid,signal.SIGINT)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                        raise RuntimeError('Automatic budget stop: '+name)
                    time.sleep(2)
                budget.settle(name,time.monotonic()-began)
                if proc.returncode:raise RuntimeError(name+' failed; see log')
            record={'event':'complete','name':name,'seconds':time.monotonic()-began,'gpu_peak_mib':peak};events.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
            assert all(sha(OUT.parent/f'local_run/wm_{g}/final.pt')==h for g,h in weights.items())
if __name__=='__main__':
    import sys
    if sys.argv[1]=='finalpilots':
        for stage in ['strictpilot','explorepilot']:run(stage)
    elif sys.argv[1]=='remaining':
        for stage in ['long','coverage','expertprobe']:run(stage)
    else:run(sys.argv[1])

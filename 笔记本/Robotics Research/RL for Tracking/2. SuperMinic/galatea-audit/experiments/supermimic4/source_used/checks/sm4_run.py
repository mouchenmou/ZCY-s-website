"""Hard-bounded smoke runner; deliberately has no formal training entry point."""
import argparse,subprocess,os,time,signal,json,shutil
from pathlib import Path
from wm4_budget import Budget
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/supermimic4'

def run(name,script,args,seconds):
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'completion.json').exists():raise RuntimeError('Short-test phase closed')
    budget=Budget(OUT/'budget.json',1800);allowed=budget.reserve(name,seconds)
    start=time.monotonic();peak=0;stop=None
    with (OUT/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(['timeout','--signal=INT','--kill-after=5',str(max(1,int(allowed)-7)),
              str(ROOT/'venv/bin/python'),'-u',str(ROOT/'checks'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            mem=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]);peak=max(peak,mem)
            size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if time.monotonic()-start>allowed or mem>24576 or size>10*2**30 or shutil.disk_usage(OUT).free<20*2**30:
                stop='budget_guard';os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(2)
    elapsed=time.monotonic()-start;budget.settle(name,elapsed)
    result={'name':name,'seconds':elapsed,'gpu_peak_mib':peak,'returncode':proc.returncode,'stop':stop}
    with (OUT/'execution.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    if proc.returncode or stop:raise RuntimeError(name+' stopped; inspect log and failure checkpoint before proceeding')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','smoke','evaluate']);p.add_argument('--attempt',default='1');a=p.parse_args()
    if a.stage=='preflight':run('preflight_'+a.attempt,'sm4_train.py',['--group','3','--preflight','--attempt',a.attempt,'--seconds','165'],180)
    elif a.stage=='smoke':
        for g in ['3','12','20','30']:run('smoke_'+g,'sm4_train.py',['--group',g,'--seconds','285'],300)
    else:
        for policy in ['initial','3','12','20','30']:
            for seed in [11401,11402]:run(f'eval_{policy}_{seed}','sm4_eval.py',['--policy',policy,'--seed',str(seed)],180)

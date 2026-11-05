"""Conservative crash-safe reservations for sequential diagnostic subprocesses.

Reserve a job's whole timeout before launch. Refund unused time only on a known
exit. A killed runner leaves its reservation charged, never silently frees it.
"""
import json,os
from pathlib import Path
class Budget:
    def __init__(self,path,limit=1200):
        self.path=Path(path);self.limit=float(limit)
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {'charged_seconds':0.,'jobs':{}}
    def flush(self):
        tmp=self.path.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(self.data,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.path)
    def reserve(self,name,cap=300):
        if name in self.data['jobs']:raise RuntimeError('Job already reserved; inspect prior attempt, never auto-refund')
        available=self.limit-self.data['charged_seconds'];cap=min(float(cap),available)
        if cap<1:raise RuntimeError('Persistent time budget exhausted')
        self.data['charged_seconds']+=cap;self.data['jobs'][name]={'reserved_seconds':cap,'status':'reserved'};self.flush();return cap
    def settle(self,name,seconds):
        row=self.data['jobs'][name]
        if row['status']!='reserved':raise RuntimeError('Job already settled')
        self.data['charged_seconds']+=max(0.,float(seconds))-row['reserved_seconds']
        row.update(status='exited',actual_seconds=float(seconds));self.flush()

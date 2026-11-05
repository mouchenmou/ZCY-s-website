"""Fixed-budget eight-step residual WM training, no policy optimization."""
import argparse, json, os, time, hashlib, shutil, random
from pathlib import Path
import numpy as np
import torch
from wm4_model import WorldModel, control, features, predict, advance_history


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fit_norm(d,selected):
    s=d['s'][selected];sn=d['sn'][selected]
    x=features(s,*[d[k][selected] for k in ('action','target','previous','last','mask_previous','history','ref','physics','age')])
    delta=sn[:,:287]-s[:,:287]
    floor=torch.ones(287,device=s.device)*1e-4
    floor[:29]=1e-3;floor[29:58]=.05;floor[65:71]=.01;floor[155:239]=.01;floor[243:]=.01
    return {'x_mean':x.mean(0),'x_std':x.std(0).clamp_min(.01),
            'delta_mean':delta.mean(0),'delta_std':torch.maximum(delta.std(0),floor),
            'reward_mean':d['reward'][selected].mean(),'reward_std':d['reward'][selected].std().clamp_min(.1)}


def save(path,model,opt,norm,step,info):
    # Keep metadata such as optimizer='Adam defaults' distinct from its tensors.
    # This format repair was made after the four fixed runs; their weights are unchanged.
    torch.save({**info,'model':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'norm':norm,'step':step,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},str(path)+'.tmp')
    os.replace(str(path)+'.tmp',path)


def train(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    start=time.monotonic();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    raw=torch.load(a.data,map_location='cpu',weights_only=False)
    n=raw['s'].shape[1];c=raw['contract']
    c=json.loads(json.dumps(c,default=lambda x:int(x) if isinstance(x,np.integer) else x))
    d={k:v.flatten(0,1).to('cuda') for k,v in raw.items() if isinstance(v,torch.Tensor)}
    selected=d['selected'].nonzero().flatten();norm=fit_norm(d,selected)
    model=WorldModel(norm['x_mean'].numel(),a.hidden).cuda()
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    info={'contract':c,'motion_ids':raw['ids'],'hidden':a.hidden,'seed':a.seed,
          'input_dim':norm['x_mean'].numel(),'dataset_sha256':sha(a.data),
          'train_transitions':int(selected.numel()),'initial_weights_sha256':initial,
          'loss':'equal-block normalized state MSE + contact BCE + normalized official reward MSE; k=8',
          'horizon':8,'batch':a.batch,'lr':3e-4,'optimizer':'Adam defaults','updates':a.updates}
    (out/'config.json').write_text(json.dumps(info,indent=2))
    size=len(d['selected'])
    for k in d:
        d[k]=torch.cat((d[k],torch.zeros((n*8,*d[k].shape[1:]),dtype=d[k].dtype,device='cuda')))
    log=open(out/'training.jsonl','w',buffering=1)
    saved=0;last_save=time.monotonic();timed=time.monotonic();trace=[];gpu_peak=0
    blocks=[(0,29),(29,58),(58,61),(61,65),(65,71),(71,155),(155,239),(239,243),(243,258),(258,287)]
    save(out/'latest.pt',model,opt,norm,0,info)
    try:
        for step in range(1,a.updates+1):
            idx=selected[torch.randint(len(selected),(a.batch,),device='cuda')]
            episode=d['episode'][idx];s=d['s'][idx];prev=d['previous'][idx];last=d['last'][idx]
            mp=d['mask_previous'][idx];hist=d['history'][idx];loss=s.new_tensor(0.);den=s.new_tensor(0.)
            components=torch.zeros(3,device='cuda');live=torch.ones(a.batch,dtype=torch.bool,device='cuda')
            for h in range(8):
                j=idx+h*n
                live=live & d['selected'][j] & (d['episode'][j]==episode)
                target=control(s,d['action'][j],prev,last,mp,d['delay'][j],d['mask'][j],c)
                sn,r,logits=predict(model,norm,s,d['action'][j],target,prev,last,mp,hist,d['ref'][j],d['physics'][j],d['age'][j])
                error=((sn[:,:287]-d['sn'][j,:287])/norm['delta_std']).square()
                dynamics=torch.stack([error[:,lo:hi].mean(-1) for lo,hi in blocks],-1).mean(-1)
                contact=torch.nn.functional.binary_cross_entropy_with_logits(logits,d['sn'][j,287:],reduction='none').mean(-1)
                reward=((r-d['reward'][j])/norm['reward_std']).square()
                weight=live.float();den=den+weight.sum()
                loss=loss+((dynamics+contact+reward)*weight).sum()
                components+=torch.stack(((dynamics*weight).sum(),(contact*weight).sum(),(reward*weight).sum())).detach()
                # Drop graph for dead windows; their labels are never included.
                s=torch.where(live[:,None],sn,d['s'][j])
                prev=target;last=d['action'][j];mp=target[:,7:];hist=advance_history(hist,s)
            loss=loss/den.clamp_min(1)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stopped without increasing budget')
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            opt.step()
            if step==1 or step%25==0:
                torch.cuda.synchronize()
                entry={'step':step,'loss':float(loss),'components':(components/den).tolist(),
                       'gradient_norm':float(grad),'seconds':time.monotonic()-timed,
                       'torch_peak_mib':torch.cuda.max_memory_allocated()/2**20}
                log.write(json.dumps(entry)+'\n');trace.append(entry)
                if step%100==0 or step==1:print(json.dumps(entry),flush=True)
                free,total=torch.cuda.mem_get_info()
                gpu_peak=max(gpu_peak,(total-free)/2**20)
                if total-free>40*2**30:raise RuntimeError('GPU 40 GiB limit exceeded')
                if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('Disk reserve below 20 GiB')
                if time.monotonic()-start>a.seconds:raise RuntimeError('Training time budget exceeded')
            if step%500==0 or time.monotonic()-last_save>300:
                save(out/'latest.pt',model,opt,norm,step,info);saved=step;last_save=time.monotonic()
        save(out/'final.pt',model,opt,norm,a.updates,info)
        # A real load/readability check on local final checkpoint.
        checkpoint=torch.load(out/'final.pt',map_location='cuda',weights_only=False)
        restored=WorldModel(info['input_dim'],a.hidden).cuda();restored.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            probe=torch.zeros(2,info['input_dim'],device='cuda')
            assert torch.equal(model(probe),restored(probe))
        summary={**info,'seconds':time.monotonic()-start,'training_seconds':time.monotonic()-timed,
                 'last_loss':trace[-1] if trace else None,'checkpoint_sha256':sha(out/'final.pt'),
                 'gpu_total_peak_sampled_mib':gpu_peak,'local_checkpoint_reload_identical':True,'status':'complete'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    except BaseException:
        save(out/'interrupted.pt',model,opt,norm,step,info)
        raise
    finally:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=5000);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--hidden',type=int,default=2048);p.add_argument('--seed',type=int,default=45)
    p.add_argument('--seconds',type=int,default=1800)
    train(p.parse_args())

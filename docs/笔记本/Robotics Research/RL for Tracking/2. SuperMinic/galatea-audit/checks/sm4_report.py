"""CPU-only reproducible short-loop report and reviewable formal-run proposal."""
import sys,json,csv,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/supermimic4'
import torch
torch.set_num_threads(4)


def mean(xs):return sum(xs)/max(1,len(xs))

def main():
    train={g:json.loads((OUT/'smoke'/('group_'+g)/'result.json').read_text()) for g in ['3','12','20','30']}
    score=json.loads((OUT/'common_wm_metrics.json').read_text());events=[json.loads(x) for x in (OUT/'execution.jsonl').read_text().splitlines()]
    freeze=json.loads((OUT/'source_freeze.json').read_text())
    assert all(hashlib.sha256((OUT/'source_used'/p if (OUT/'source_used'/p).exists() else ROOT/p).read_bytes()).hexdigest()==h for p,h in freeze.items())
    assert len({json.dumps(t['initial_hashes'],sort_keys=True) for t in train.values()})==1
    assert all(t['raw_transitions']==30720 and t['updates']=={'actor':28,'wm':224,'critic':112} for t in train.values())
    episodes={};aggregate={};per_motion=[]
    for g in ['initial','3','12','20','30']:
        ds=[json.loads((OUT/'evaluation'/f'{g}_{s}'/'result.json').read_text()) for s in [11401,11402]]
        episodes[g]=sum([d['episodes'] for d in ds],[]);aggregate[g]={}
        for split in ['core_seen_all','heldout_unseen_all']:
            e=[x for x in episodes[g] if x['split']==split]
            aggregate[g][split]={'n':len(e),**{k:mean([x[k] for x in e]) for k in ['return','length','success','reference_max_fraction','sustained_steps']}}
        for motion in sorted({x['motion'] for x in episodes[g]}):
            e=[x for x in episodes[g] if x['motion']==motion]
            per_motion.append({'policy':g,'motion':motion,'split':e[0]['split'],'n':len(e),
                               **{k:mean([x[k] for x in e]) for k in ['return','length','success','reference_max_fraction','sustained_steps']}})
    with (OUT/'per_motion_evaluation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(per_motion[0]));w.writeheader();w.writerows(per_motion)
    diagnostics={};model_metrics={};baselines={}
    for g in train:
        d=torch.load(OUT/'smoke'/('group_'+g)/'replay.pt',map_location='cpu',weights_only=False)
        valid=d['selected'];contact=d['sn'][...,287:292].bool().any(-1);run=torch.zeros(120,dtype=torch.long);contact8=0;contact32=0
        for t in range(len(valid)):
            run=torch.where(valid[t]&contact[t],run+1,torch.zeros_like(run));contact8+=int((run>=8).sum());contact32+=int((run>=32).sum())
            run[d['done'][t]]=0
        logs=[json.loads(x) for x in (OUT/'smoke'/('group_'+g)/'updates.jsonl').read_text().splitlines()]
        assert len({x['actor_hash'] for x in logs})==28
        diagnostics[g]={'contact_run_ge8_steps':contact8,'contact_run_ge32_steps':contact32,
                         'actor_mean_rms_initial':float(d['actor_mean'][d['policy_version']==0].square().mean().sqrt()),
                         'actor_mean_rms_final':float(d['actor_mean'][d['policy_version']==28].square().mean().sqrt()),
                         'action_saturation_fraction':float((d['action'].abs()>.99).float().mean()),
                         'actor_grad_max':max(x['actor_grad'] for x in logs),'critic_grad_max':max(x['critic_grad'] for x in logs),
                         'imagined_effective_steps_first':logs[0]['imagined_effective_steps'],
                         'imagined_effective_steps_final':logs[-1]['imagined_effective_steps'],
                         'wm_loss_first':logs[0]['wm_loss'],'wm_loss_final':logs[-1]['wm_loss'],
                         'actor_objective_first':-logs[0]['actor_loss']*10,'actor_objective_final':-logs[-1]['actor_loss']*10}
        del d
        model_metrics[g]={}
        for split in ['core_seen_all','heldout_unseen_all']:
            for h in [1,8]:
                rs=[r for r in score['groups'][g] if r['split']==split and r['horizon']==h];n=sum(r['n'] for r in rs)
                metrics={k:sum(r[k]*r['n'] for r in rs)/n for k in rs[0] if k not in ['seed','split','horizon','n']}
                model_metrics[g][f'{split}_{h}']={'n':n,**metrics}
    for split in ['core_seen_all','heldout_unseen_all']:
        baselines[split]={}
        for h in [1,8]:
            squared=0.;n=0;zero_brier=0.
            for seed in [11401,11402]:
                d=torch.load(OUT/'evaluation'/f'initial_{seed}'/'common_physx.pt',map_location='cpu',weights_only=False)
                ok=d['selected']&((d['motion']<3) if split=='core_seen_all' else (d['motion']>=3))
                for j in range(1,h):ok[:-j]&=d['selected'][j:]&~d['done'][j-1:-1];ok[-j:]=False
                starts=ok.nonzero()
                if h==8 and len(starts)>512:starts=starts[torch.randperm(len(starts),generator=torch.Generator().manual_seed(11499))[:512]]
                t,lane=starts.T;true=d['sn'][t+h-1,lane];pred=d['s'][t,lane];count=len(t)
                squared+=float(((pred[:,7:29]-true[:,7:29])*180/torch.pi).square().mean())*count
                zero_brier+=float(true[:,287:292].square().mean())*count;n+=count
            baselines[split][h]={'n':n,'hold_state_finger_q_rmse_deg':math.sqrt(squared/n),'always_no_contact_brier':zero_brier/n}
    estimates={g:{'collection_minutes':t['collection_seconds']*(20000/256)/60,
                  'update_minutes':t['update_seconds']*(2496/28)/60,
                  'compute_minutes':(t['collection_seconds']*(20000/256)+t['update_seconds']*(2496/28))/60} for g,t in train.items()}
    summary={'training':train,'evaluation':aggregate,'stability':diagnostics,'common_WM':model_metrics,'sanity_baselines':baselines,
             'formal_estimate':estimates,'total_job_seconds_including_failed_preflights':sum(e['seconds'] for e in events),
             'training_job_seconds':sum(e['seconds'] for e in events if e['name'].startswith('smoke_')),
             'source_freeze_verified':True,'formal_training_started':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=[]
    def add(s=''):lines.append(s)
    add('\n## 四组从零 SuperMimic 风格闭环短测试（sm4-smoke-v1）\n')
    add('**结论：四组完整 Actor–WM–Critic 交替训练已经真实跑通，但尚未出现任务成功，也没有出现“Reference 越多、Actor 越好”的单调趋势。12 条组在这次极短测试中相对稳定；3/20/30 条组的真实表现出现退化。正式长训没有启动，不建议把接口通过直接等同于可以放心长训。**\n')
    add('研究主线保留 3/12/20/30 条固定嵌套 Reference，不退回三条 Motion 或单独手指 WM。旧四个 WM checkpoint 未用于本次初始化；没有 Expert 权重、Expert 数据或固定手指动作脚本。此前 WM-only 泛化结果是本研究的动机，不能推导出当前闭环 Actor 已获益。\n')
    add('完整协议：[PLAN.zh-CN.md](experiments/supermimic4/PLAN.zh-CN.md)。数据、checkpoint、逐周期日志和共同真实评估轨迹都保存在 `experiments/supermimic4/`。\n')
    add('### 1. 本次实际执行与公平性\n')
    add('| Reference 数 | 原始/可用真实训练 transitions | WM/Actor/Critic 更新 | 采集秒 | 更新秒 | 作业总秒 | 峰值显存 GiB |')
    add('|---|---:|---:|---:|---:|---:|---:|')
    for g,t in train.items():
        ev=next(e for e in events if e['name']=='smoke_'+g)
        add(f"| {g} | 30,720 / {t['usable_transitions']:,} | 224 / 28 / 112 | {t['collection_seconds']:.2f} | {t['update_seconds']:.2f} | {ev['seconds']:.2f} | {ev['gpu_peak_mib']/1024:.2f} |")
    add(f"\n四组训练作业合计 {summary['training_job_seconds']:.1f} 秒；含两次失败接口预检、通过预检、十次评估和共同 WM 检查，GPU 作业累计占用墙钟 {summary['total_job_seconds_including_failed_preflights']:.1f} 秒。这里不是 CUDA kernel 活跃时间。训练原始交互严格相同；评估采用相同完整 episode 数，因存活时长不同，额外评估物理步数不同，单独记录且不返回训练。\n")
    add('每组 120 lanes，初始采 32 步，然后 28 次：WM 拟合 8-step 窗口 → 冻结 WM/Critic 参数、Actor 32-step BPTT → Critic GAE → 更新后的 Actor 采 8 步。实际每条 Motion 平均原始交互分别 10,240 / 2,560 / 1,536 / 1,024。reset 后首步计入预算但排除拟合；terminal transition 保留，之后不能跨 reset 拼接。\n')
    add('三网络初始参数哈希四组完全一致。每次 Actor 更新后的哈希都改变，0～28 共 29 个 policy_version 都进入真实 replay。独立重算初始/最终 Actor 对保存观测的动作，最大误差 <2e-7，确认真实采集来自当前 Actor。固定单位缩放、优化器和更新预算；测试 Motion 未用于拟合、归一化或超参选择。\n')
    add('运行源码已按冻结哈希归档到 `source_used/`。收尾审计发现 contract 中一个未参与计算的 state_dim 元数据继承了旧值 292；实际 replay、网络和计算始终一致为 312。当前源码已修正元数据，旧 checkpoint 保留原样，准确分段写入 `state_schema.json`；未改变任何权重、训练过程或结果。\n')
    add('### 2. 修复与检查\n')
    add('- 新 WM 单独预测 force-active 概率，按官方 `pad_force.norm > 0` 真值监督并推进历史；与距离门控手物 contact 分开。已验证真实历史逐位一致。\n- 补齐官方 Actor 观测所需的 5 个软垫位置：旧 28-body 状态不包含 elastomer，直接映射曾在预检报错。另一次预检发现 Critic 输入维度配置错误；两个失败均保留日志，修正后才统一启动四组。\n- 当前实际 PD target 是动力学输入；原始 Action 只经显式延迟/屏蔽/PD 转换影响物理预测。Reward/failure 分支仍需上一动作以表示官方动作变化惩罚。\n- 所有训练/评估 PD 重建误差 ≤1.2e-7 rad；Actor 可微观测重建误差 ≤6.6e-7。决策前统一刷新官方观测，消除帧号递增和 reset 刷新造成的时间口径差异。\n- Actor 想象只接收当前上下文、已知 Reference 及独立新采样的未来随机控制条件；污染 replay 中真实 delay/mask/下一状态不改变想象输出。Actor 更新期间 WM 哈希不变、无 WM 参数梯度；Actor 梯度非零。\n- 已检查 terminal 无 bootstrap、非 terminal 想象截断有 bootstrap。真实 Reward/Termination 源码保持干净；固定完整重力、严格官方手速门槛、合法 frame0 reset。\n')
    add('### 3. 真实 PhysX 结果\n')
    add('共同种子 11401/11402；每 Motion 每 seed 15 episodes，共核心 g16/g25/g26 90 episodes、五条 heldout 150 episodes。所有 episode 都在真实 Done 结束，没有预算截断。官方 eval_mode 关闭训练 mask，保留官方 delay 和传感器噪声。下面是每个完整 episode 累计官方 Reward，未做任意归一化；原始逐 Motion 结果见 [CSV](experiments/supermimic4/per_motion_evaluation.csv)。\n')
    add('| Policy | 核心3条 Reward | 核心长度 | 未见5条 Reward | 未见长度 | 成功数/评估数 |')
    add('|---|---:|---:|---:|---:|---:|')
    for g,a in aggregate.items():
        x,y=a['core_seen_all'],a['heldout_unseen_all']
        add(f"| {'共同随机初始' if g=='initial' else 'WM/Actor-'+g} | {x['return']:.2f} | {x['length']:.2f} | {y['return']:.2f} | {y['length']:.2f} | 0 / 240 |")
    add('\n12 条组未见 Motion Reward 较初始上升约 11.4%，核心 Reward 下降约 4.3%；这是本次短测试的描述性结果，不能据此宣布最优 Motion 数为 12。3/20/30 条组核心 Reward 分别下降约 58.2%/43.2%/60.1%。所有组成功率仍为零，28 次 Actor 更新和单个初始化种子不足以回答最终学习能力或统计显著性。\n')
    add('![真实评估 Reward 与 Episode Length，点为逐 Motion 均值](experiments/supermimic4/actor_evaluation.png)\n')
    add('随机初始 Actor 输出很小，在 Reference 早段保持姿态就能获得一定回报；这不代表它会操作电钻。训练后更主动的动作可能更快触发真实失败。失败以手指 keypoint 偏差为主：30 条组 200 次 thumb、60 次 index 原因（可同时发生）；3 条组额外有 72 次 object-position 失败。本次没有以手速/臂速门槛作为主要失败原因，不能把退化直接归咎于开启严格手速门槛。\n')
    add('### 4. 共同数据上的 WM 与接触\n')
    add('以下是同一随机初始 Actor 真实 PhysX 轨迹、相同 8-step 窗口上的未见 Motion 误差。每组同样 1,024 个窗口；条件控制使用记录的实际 delay/mask，仅用于诊断，未输入 Actor 训练。物体误差是 xyz 分量 RMSE。\n')
    add('| Motion 数 | 手指 q RMSE 度 | 物体位置 RMSE mm | Contact Brier | Reward MAE |')
    add('|---|---:|---:|---:|---:|')
    for g,mm in model_metrics.items():
        v=mm['heldout_unseen_all_8'];add(f"| {g} | {math.sqrt(v['finger_q_deg_mse']):.3f} | {1000*math.sqrt(v['object_xyz_mse']):.3f} | {v['contact_brier']:.4f} | {v['reward_mae']:.3f} |")
    b=baselines['heldout_unseen_all'][8]
    add(f"\n同窗口直接保持初始手指角度的基线 RMSE 为 {b['hold_state_finger_q_rmse_deg']:.3f}°，永远预测无接触的 Brier 为 {b['always_no_contact_brier']:.4f}。稀少接触下，低总体 Brier 不能证明模型学会了接触；本轮数据主要是随机 Actor 早段行为，不足以评价复杂操纵。1-step、核心/未见和速度/位置等完整指标见 `summary.json` 与 `common_wm_metrics.json`。\n")
    add('| Motion 数 | 两相邻步持续接触占可用数据 | 接触开始次数 | 连续接触≥8步的时间点 | 连续接触≥32步的时间点 |')
    add('|---|---:|---:|---:|---:|')
    for g,t in train.items():
        z=diagnostics[g];add(f"| {g} | {t['sustained_contact_transitions']/t['usable_transitions']*100:.2f}% | {t['independent_contact_onsets']} | {z['contact_run_ge8_steps']} | {z['contact_run_ge32_steps']} |")
    add('\n接触开始按 false→true 统计，不等于独立成功抓取；接触持续阈值按真实连续 trajectory 计算并在 Done 清零。所有训练 transition 都在 Reference 前 25%。本次每 lane 总共只有 256 控制步，而最短 Reference 四分之一也超过 256 步，因此本次短测试设计本身就无法验证中后段训练覆盖；同时真实评估也只到平均约 3%～10% 就失败。不能把该 100% 早段比例单独归咎于旧采集器，也不能声称新闭环已经解决完整 Reference 覆盖。\n')
    add('### 5. 还不能放心长训的原因\n')
    add('**已证实：** WM loss 均下降、想象目标改善，真实 Actor 表现却未同步改善；三组出现明显退化。短测试通过的是数据和梯度能流通，不是梯度已可信。当前全身、手、物体和 contact 都在同一完整闭环里训练，没有单独优化第 8/16/32 步手指 Response。\n')
    add('**待验证的原因：** 学习得到的 Reward/failure 分支是否给出错误动作梯度；Critic 自举是否放大早期模型偏差；预测关节/关键点/软垫位置的几何一致性不足是否被 Actor 利用。这些不是本轮已证明的根因。新 termination 监督用官方真实失败标签，但想象中的软 survival 必须进一步做校准检查。初期 H=32 的有效 survival 通常只有约 3 步，实际梯度跨度与配置 H 不等价。\n')
    add('实现沿用了完整交替训练原则，但与原论文的 humanoid 观测历史、未来目标编码、reset sampling、batch/更新比例及任务 reward 头不同；因此是 GALATEA 适配的 SuperMimic 风格实现。未知几何/梯度问题和单 seed 是研究限制，不应包装为作者完整复现已成功。\n')
    add('建议下一步仍保留四组，在训练候选 Motion 上检查完整 Actor 更新方向的真实回报一致性、Reward/termination 及 bootstrap 的数值正确性，再做统一的小闭环稳定性复验。不要用五条 heldout 反复选择超参；这些短测试结果已经看过，应透明披露，正式阶段协议须先冻结。这里不建议继续为某个手指定制 Response Loss，也不建议换 Expert 来掩盖冷启动问题。\n')
    add('### 6. 正式预算候选与停止机制（需另行确认）\n')
    add('四组均从同一随机初始化协议重新开始，每组 2,400,000 条训练真实交互，WM 19,968 / Actor 2,496 / Critic 9,984 次更新。它是统一预算的可行性对照，不保证已经达到收敛；不能根据某组表现好而偷偷增加预算。\n')
    add('| Motion 数 | 外推纯采集分钟 | 外推更新分钟 | 两者合计分钟 |')
    add('|---|---:|---:|---:|')
    for g,e in estimates.items():add(f"| {g} | {e['collection_minutes']:.1f} | {e['update_minutes']:.1f} | {e['compute_minutes']:.1f} |")
    add('\n以上按本服务器实测短循环线性外推，未计入完整评估、周期保存与存活分布变化；建议预留每组 50～60 分钟，四组约 3.3～4 小时，硬上限每组 60 分钟、总计 4 小时，单卡显存 24 GiB、本地结果 20 GiB、磁盘余量 20 GiB。当前仅短测试代码，没有正式长训启动入口；要在确认后补齐长期 ring-buffer 统计、恢复和评估调度。\n')
    add('正式候选保护规则：每 300,000 训练交互在共同核心三条、独立固定验证种子评估；从 600,000 交互后，连续三次验证的平均累计 Reward 和 Episode Length 都比此前最佳同预算检查点下降 ≥20%，则该组保存并停止。任何 NaN、PD/观测/label/梯度错误或资源超限立即停止。该规则只看训练候选，不看五条 heldout；若某组提前停止，必须报告实际预算并只在共同已完成里程碑做等预算比较。成功率与逐 Motion 数值同时公开，不以奖励单项好看代替任务成功。\n')
    add('Checkpoint 至少每 7 周期覆盖 latest，且每 300,000 交互保留里程碑：三网络、三个 optimizer、固定缩放、RNG、replay、数据索引、计数与代码哈希。本地恢复必须承认 PhysX hidden state 未保存，重新合法初始化；不做外部备份。\n')
    add('**本次停在四组短闭环与真实评估完成处。正式训练需要你按本轮要求另行确认；鉴于真实退化信号，我建议先完成上述闭环梯度/回报一致性检查，再决定是否投入这 3～4 小时。**\n')
    report=ROOT/'REPORT.zh-CN.md';text=report.read_text();marker='## 四组从零 SuperMimic 风格闭环短测试（sm4-smoke-v1）'
    if marker in text:text=text.split(marker)[0].rstrip()+'\n'
    report.write_text(text+'\n'.join(lines))
    # Standard standalone plot; no generated illustration or external service.
    sys.path.insert(0,str(ROOT/'audit-lib/plots'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));groups=['initial','3','12','20','30'];xx=np.arange(5)
    for ax,key,title in zip(axes,['return','length'],['True PhysX episode return','True PhysX episode length']):
        for j,(sp,label,color) in enumerate([('core_seen_all','Core 3 (seen by all)','#2563eb'),('heldout_unseen_all','Heldout 5 (unseen by all)','#f97316')]):
            offset=(j-.5)*.36;means=[aggregate[g][sp][key] for g in groups]
            ax.bar(xx+offset,means,width=.34,color=color,alpha=.65,label=label)
            for k,g in enumerate(groups):
                vals=[r[key] for r in per_motion if r['policy']==g and r['split']==sp]
                ax.scatter(k+offset+np.linspace(-.1,.1,len(vals)),vals,color=color,s=18,edgecolors='black',linewidths=.3,zorder=3)
        ax.set_xticks(xx);ax.set_xticklabels(['Random init','3 refs','12 refs','20 refs','30 refs']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('28 Actor updates; 240 real episodes/policy; zero successes. Points: motion means.')
    fig.tight_layout();fig.savefig(OUT/'actor_evaluation.png',dpi=180);plt.close(fig)
    print(json.dumps({'report':str(report),'total_seconds':summary['total_job_seconds_including_failed_preflights'],'formal_started':False}))

if __name__=='__main__':main()

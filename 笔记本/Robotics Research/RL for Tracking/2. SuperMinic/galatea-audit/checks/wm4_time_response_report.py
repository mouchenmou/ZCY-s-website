"""Summarize fixed read-only diagnostic outputs; no simulation or model calls."""
import csv,json
from pathlib import Path
import numpy as np
from wm4_time_response import stats,write_csv

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/wm4_200k/time_response'


def main():
    meta=json.loads((OUT/'selection_and_audit.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    inventory=list(csv.DictReader((OUT/'case_inventory.csv').open()))
    cases=list(csv.DictReader((OUT/'per_case_metrics.csv').open()))
    time_rows=list(csv.DictReader((OUT/'all_valid_time_series.csv').open()))
    controls=list(csv.DictReader((OUT/'all_valid_controls.csv').open()))
    control_map={(x['split'],x['lane'],x['step']):x for x in controls}
    curves={(x['split'],x['lane'],x['model'],x['step']):x for x in time_rows}
    phase_rows=[];models=['WM-3','WM-12','WM-20','WM-30']
    for case in cases:
        if case['model']=='PhysX':continue
        split,lane,model=(case[k] for k in ('split','lane','model'))
        for phase in ('no_contact_steps','any_contact_steps','own_finger_contact_steps'):
            values=[];velocities=[]
            for t in range(1,33):
                c=control_map[split,lane,str(t)]
                any_contact=bool(int(c['any_contact_base']) or int(c['any_contact_plus']))
                own=bool(int(c['own_contact_base']) or int(c['own_contact_plus']))
                use=(phase=='no_contact_steps' and not any_contact) or (phase=='any_contact_steps' and any_contact) or (phase=='own_finger_contact_steps' and own)
                if use:
                    a=curves[split,lane,model,str(t)];b=curves[split,lane,'PhysX',str(t)]
                    values.append(abs(float(a['q_delta_deg'])-float(b['q_delta_deg'])))
                    velocities.append(abs(float(a['qd_delta_deg_s'])-float(b['qd_delta_deg_s'])))
            if values:
                phase_rows.append({'split':split,'lane':lane,'model':model,'phase':phase,
                    'clear_response':case['clear_response'],'steps':len(values),
                    'q_response_mae_deg':float(np.mean(values)),'qd_response_mae_deg_s':float(np.mean(velocities))})
    write_csv(OUT/'contact_phase_metrics.csv',phase_rows)
    additional={'all_valid_including_weak':{},'contact_step_metrics':{},'paired_improved_cases':{}}
    for split in ('core','unseen','pooled'):
        keep=lambda row:split=='pooled' or row['split']==split
        additional['all_valid_including_weak'][split]={m:stats([float(x['trajectory_q_mae_deg']) for x in cases if keep(x) and x['model']==m]) for m in models}
        additional['contact_step_metrics'][split]={}
        for phase in ('no_contact_steps','any_contact_steps','own_finger_contact_steps'):
            additional['contact_step_metrics'][split][phase]={m:stats([float(x['q_response_mae_deg']) for x in phase_rows
                if keep(x) and x['phase']==phase and x['model']==m and x['clear_response']=='True']) for m in models}
        ref={(x['split'],x['lane']):float(x['trajectory_q_mae_deg']) for x in cases if keep(x) and x['model']=='WM-3' and x['clear_response']=='True'}
        additional['paired_improved_cases'][split]={}
        for m in models[1:]:
            rows=[x for x in cases if keep(x) and x['model']==m and x['clear_response']=='True']
            additional['paired_improved_cases'][split][m]={'n':len(rows),'better_than_wm3':sum(float(x['trajectory_q_mae_deg'])<ref[x['split'],x['lane']] for x in rows)}
    (OUT/'supplementary_statistics.json').write_text(json.dumps(additional,indent=2))
    prefix='experiments/wm4_200k/time_response/'
    lines=['\n\n# 2026-11-05：8步动作脉冲的完整时间响应诊断\n',
        '**结论：真实手指的额外角度通常先升高，撤回额外命令后再逐渐下降，但第32步并未归零。其实际PD target差也仍有残留。四个WM都能画出大致回落形状；更多Motion改善未见状态下的整段预测，但不单调，也没有消除错误的提前响应。**\n',
        '这里“回落”指扰动分支相对基准分支的额外角度减小，不是手指绝对姿态回到最初位置。基准Action本身仍随时间变化。所有曲线都计算 **正扰动分支−基准分支**，没有拿此前正负两条扰动分支的差冒充这次结果。\n',
        '## 数据与有效性\n',
        '- 复用此前120环境、seed 8201/8202产生的真实GPU PhysX轨迹；没有新增仿真转移，没有训练更新，没有修改四个WM、Actor、Reward、Termination或Critic。\n- 核对新建PhysX进程的完整前缀与上下文；基准重复运行所有已保存张量逐位一致，包括32步轨迹、控制和Reward。基准与正扰动分支的初始状态、控制历史、随机化参数、延迟与mask一致。\n- 本轮要求两条分支从初始到第32步都没有done，连第32步终止的案例也排除；同时验证状态连续、年龄每步加1。240个候选中仅48例有效（core 18、unseen 30），192例不混入有效结果。\n- 48例中36例真实响应峰值≥1°且有效PD目标差≥0.1°，作为“明显响应”统计；其余12例保留在逐例CSV和图册，并提供含弱响应的单独统计。36例中28例全程无手指接触，8例有某个手指接触；只有2例受扰手指自身短暂接触。没有足够案例支持“持续接触下已验证可靠”的结论。\n- 扰动为受扰关节Action在1–8步增加0.2，第9步撤回。示例均无Action裁剪、扰动期间受扰关节无mask。控制频率30Hz。关节名称来自Isaac Gym真实DOF顺序，未使用URDF文本排列代替。\n- WM采用同一个真实初态和同一Action序列、已知外生控制条件独立自回归32步；未来物理状态不回灌。PD target从模型控制历史重算，未直接输入未来真实target。四个模型的手指PD target差与实际PhysX最大差约7.63e-6°，图中曲线重合是正确控制映射的结果，不是WM学到了动力学。\n- 四个Checkpoint和所有源PhysX文件的SHA-256在诊断前后相同。绘图依赖安装在独立audit-lib/plots目录，未改动训练环境或模型。\n',
        '## 具体案例：五条曲线和指定时刻的角度差\n',
        '所有角度数值单位为度，速度单位为度/秒；q/qd取每个控制步结束后的值，PD target为该步实际施加值。黄色区域为额外命令持续期间，绿色竖带为受扰手指实际接触阶段。总览图依次包含角度、速度、PD target、提交/延迟后的Action及mask/接触状态；单独角度图恰好五条曲线。示例按真实响应峰值的中位数选择，不按哪个WM看起来更好挑选。\n']
    labels={('core',3):'原三条：无接触，g16食指末端关节',('core',73):'原三条：出现接触，g25中指末端关节',
            ('unseen',69):'未见Motion：无接触，g33食指末端关节',('unseen',53):'未见Motion：出现接触，g22小指掌指屈伸关节'}
    for case in meta['examples']:
        split=case['split'];lane=case['lane'];stem=f'case_{split}_{lane:03d}'
        lines.extend([f"### {labels[split,lane]}（lane {lane}）\n",
            f"![五条角度差曲线]({prefix}{stem}_angle.png)\n",
            f"[角度、速度、PD target与控制状态完整图]({prefix}{stem}.png) · [矢量PDF]({prefix}{stem}.pdf)\n",
            '|步数|真实PhysX|WM-3|WM-12|WM-20|WM-30|实际PD target差|',
            '|---:|---:|---:|---:|---:|---:|---:|'])
        for t in (1,4,8,16,24,32):
            values=[float(curves[split,str(lane),m,str(t)]['q_delta_deg']) for m in ['PhysX',*models]]
            target=float(curves[split,str(lane),'PhysX',str(t)]['pd_target_delta_deg'])
            lines.append('|'+str(t)+'|'+'|'.join(f'{x:.3f}' for x in values)+f'|{target:.3f}|')
        lines.append(f"\n真实峰值{case['peak_deg']:.3f}°（第{case['peak_step']}步），第32步仍有{case['angle32_deg']:.3f}°，实际目标差{case['target32_deg']:.3f}°。")
        if case['own_contact_steps']:
            lines.append(f"受扰手指接触出现在第{','.join(map(str,case['own_contact_steps']))}步；任意手指接触的逐步标记见图。它包含无接触→短暂自身接触→其他手指接触阶段，不能称为32步持续的受扰手指接触。\n")
        else:lines.append('初始及1–32步两条分支的所有手指均无物体pad接触。\n')
    lines.extend(['## 如何理解这些曲线\n',
        '1. **撤回Action，不等于PD target立刻回到基准。** 延迟会使第9步仍执行第8步的命令；平滑、deadband和控制历史让目标差继续存在。例如g33无接触例，第9步额外提交命令已为0，但延迟后额外命令仍为0.2；第32步目标差仍0.854°，真实角度差0.893°。不能把任何非零尾部都算作WM虚假残留。\n',
        '2. **真实速度差先为正，随后转负，额外角度逐渐减少。** g33例在第8步速度差为+12.48°/s，第16步为−4.90°/s，第32步仍为−1.10°/s：在窗口结束时仍在回落，不能声称已经完全回到基准。控制目标的停顿/变化使实际速度曲线不完全光滑，模型经常把这些细节抹平。\n',
        '3. **更确定的错误是提前运动。** g33无接触例第1步由于延迟，实际PD target差、q差和qd差均为0；WM-3/12/20/30却预测额外角度0.105/0.169/0.133/0.186°及约4–5°/s的速度差。控制映射已经正确重算，错误在WM对动作后果的预测。\n',
        '4. **最后一个点接近，不等于过程正确。** g33例WM-3的第32步0.851°接近真实0.893°，但第8步仅2.190°（真实3.382°），第16步仅1.897°（真实3.311°）。其他三组改善了前半段，却也普遍过早开始下降、漏掉真实峰值附近的平台。\n',
        '5. **不是所有接触案例都随Motion增加而变好。** g25接触例WM-12明显低估峰值和中段，WM-20较接近；g22接触例WM-12整段误差较小，WM-30则高估幅度并留下偏大的尾部。具体接触时间已逐步标出；自身接触案例只有两例，不能进行强统计归纳。\n',
        '## 案例差异与完整32步误差\n',
        '以下先对每个案例计算32步 |预测额外角度−真实额外角度| 的平均值，再报告案例间中位数和四分位区间；不会把正负响应直接平均相消。统计对象是36个明显响应案例，原三条15例、未见Motion 21例。\n',
        '|模型|原三条整段MAE，°（四分位区间）|未见整段MAE，°（四分位区间）|未见撤回后9–32步MAE中位数，°|未见第32步绝对误差中位数，°|',
        '|---|---:|---:|---:|---:|'])
    for m in models:
        core=summary['core']['all']['models'][m];u=summary['unseen']['all']['models'][m]
        fmt=lambda x:f"{x['median']:.3f}（{x['q25']:.3f}–{x['q75']:.3f}）"
        lines.append(f"|{m}|{fmt(core['trajectory_q_mae_deg'])}|{fmt(u['trajectory_q_mae_deg'])}|{u['withdrawal_q_mae_deg']['median']:.3f}|{u['error32_abs_deg']['median']:.3f}|")
    lines.extend(['\n未见Motion的整段速度差误差中位数（°/s）依次为2.964、2.389、2.445、2.484。12/30条改善明显，20条未持续优于12；原三条上的整段角度误差没有改善。单一案例可能与总体趋势相反。\n',
        '|模型|无接触案例整段MAE中位数，°（28例）|出现任意手指接触的案例整段MAE中位数，°（8例）|全体明显案例的整段MAE范围，°|',
        '|---|---:|---:|---:|'])
    for m in models:
        n=summary['pooled']['no_contact']['models'][m]['trajectory_q_mae_deg'];c=summary['pooled']['any_contact']['models'][m]['trajectory_q_mae_deg'];a=summary['pooled']['all']['models'][m]['trajectory_q_mae_deg']
        lines.append(f"|{m}|{n['median']:.3f}|{c['median']:.3f}|{a['min']:.3f}–{a['max']:.3f}|")
    lines.extend(['\n上述接触分组是整条轨迹的分类，包含接触前后时段。仅在实际接触时间步计算的误差另见contact_phase_metrics.csv和supplementary_statistics.json，避免把无接触部分当接触性能。\n',
        '真实第32步残留绝对角度：36例中位数0.715°，四分位区间0.545–0.920°，范围0.363–1.440°。相对峰值的残留中位数约17.8%，范围10.9%–81.3%；unseen lane 86仍保留约81%的峰值，说明并非所有案例都充分回落。事先定义“接近基准”为绝对残留≤0.5°且≤峰值10%，36例均未达到。这只是明确的诊断口径，不是唯一可用的工程容差。\n',
        '因此，这次不能得出“PhysX已归零但WM不归零”的结论。支持的判断是：PhysX多数已明显回落，但控制器仍保留非零目标差；WM通常预测了回落趋势，却在启动时机、峰值、平台和回落速度上存在偏差。增加Motion改善未见状态下的完整过程，但不保证每一例、更不保证接触过程可靠。\n',
        f'![每个案例的误差分布]({prefix}case_error_distribution.png)\n',
        '## 全部本机文件\n',
        f'- [48个完全有效案例的逐例曲线PDF]({prefix}all_48_valid_cases.pdf)，包括12个弱/无明显响应案例。\n',
        f'- [所有有效案例逐步q/qd/PD target及差值CSV]({prefix}all_valid_time_series.csv)。\n',
        f'- [每步Action、延迟、mask及接触标记CSV]({prefix}all_valid_controls.csv)。\n',
        f'- [指定时刻数值表CSV]({prefix}selected_case_timepoints.csv)。\n',
        f'- [逐例误差与分类]({prefix}per_case_metrics.csv)，[候选案例与排除原因]({prefix}case_inventory.csv)。\n',
        f'- [数据、模型哈希和完整历史审计]({prefix}selection_and_audit.json)。\n',
        '本轮仅诊断，已结束；没有模型更新、权重重写、Actor训练或外部上传。\n'])
    report=ROOT/'REPORT.zh-CN.md';text=report.read_text();marker='\n\n# 2026-11-05：8步动作脉冲的完整时间响应诊断'
    if marker in text:text=text.split(marker)[0]
    report.write_text(text+'\n'.join(lines))
    print(json.dumps(additional['paired_improved_cases'],indent=2))


if __name__=='__main__':main()

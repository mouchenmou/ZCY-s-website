# GALATEA × SuperMimic：Power Drill WM 数据规模研究

本仓库保存研究源码、配置、固定数据划分、诊断结果和恢复说明。大型数据、模型、MANO、Isaac Gym与认证文件不进入Git。

**新服务器请先读 [2026-11-05 交接文档](HANDOFF_2026-11-05.zh-CN.md)。** 后续已完成四组从零 Actor–WM–Critic 极短闭环；最新拟议主线是 3/12/20/30 条 Reference × GALATEA PPO/SuperMimic 风格 MBRL，共八个实验单元，尚未启动正式长训。下文 WM-only 与“未启动 Actor”的描述属于历史阶段，不代表当前最新状态。最新修订见 [八单元方案](experiments/supermimic4/PLAN.zh-CN.md)。

2026-11-04：已上传至 `mouchenmou/galatea-wm-study` 私有仓库，并从GitHub重新克隆、逐文件校验源码/配置/日志。验证记录见 [github_backup_verification.json](evidence/github_backup_verification.json)。这是历史备份记录。2026-11-05用户明确撤销全部实例外备份要求，接受实例关闭后丢失结果的风险；本轮不再配置R2或上传任何文件。

## 当前实验

- 3 / 12 / 20 / 30条嵌套Power Drill Reference；每组已完成200000条有效训练PhysX transitions及5000次WM更新。
- 只训练从零初始化的World Model；不训练Actor，不新增Response Loss，不修改环境Reward、Termination或Critic。
- 独立测试Motion：g3、g9、g14、g22、g33；不得用于WM训练、归一化或超参数选择。
- 正式四组实验已完成并停止；外部备份已由用户取消。数据、Checkpoint和运行日志位于 `experiments/wm4_200k/local_run/`。

结果：3→12条明显改善未见Motion的泛化，20/30条没有持续提升；单步动作响应与Reward方向仍不足以支持Actor BPTT。详见[正式对照报告](REPORT.zh-CN.md#2026-11-05四组正式-world-model-对照结果)和[完成记录](experiments/wm4_200k/completion.json)。


最新追加诊断已完成：长时动作脉冲对照延长到256步；旧采集器前段偏置主要来自随机手指脚本的9步失败。冻结Expert从合法起点连续执行，严格手速门槛下30条Motion均覆盖中段，第三分位约19.74%、持续接触约49.17%。旧WM仍有PD平台/回落预测与接触历史语义问题，未重训，也未启动Actor优化。详见[长时诊断与采集修复报告](REPORT.zh-CN.md#长时动作响应reference-覆盖根因与最小采集修复)，全部结果位于`experiments/wm4_200k/followup/`。

## 阅读入口

- [最新准备状态](experiments/wm4_200k/PREPARATION_STATUS.zh-CN.md)
- [四组执行协议](experiments/wm4_200k/PROTOCOL.zh-CN.md)
- [固定嵌套ID与选择规则](experiments/wm4_200k/nested_groups.json)
- [历史备份方案（本轮已取消）](BACKUP_SETUP.zh-CN.md)
- [首轮研究报告](REPORT.zh-CN.md)（初始环境快照；后续安装结果见最新准备状态）

## 已完成检查

GPU PhysX、上传MANO、官方GALATEA环境及控制接口短测通过；官方Expert三条核心Motion评估65/96成功。该结果不是WM训练结果，也不能与旧6144例评估直接比较。

代码检查程序位于`checks/`，版本和实测证据位于`evidence/`。上游源码固定版本见`evidence/source_versions.json`，Python环境见`evidence/environment-freeze.txt`。

## 恢复范围

Git clone可恢复本仓库的脚本、配置、报告和小日志；完整运行还需要按固定commit恢复上游代码，并取得匹配哈希的官方数据、Expert、Isaac Gym和获许可的MANO。不要把代码克隆成功解释为大型数据/Checkpoint已经备份，或整个训练环境已经恢复。

暂不启用GitHub Actions；这里的GPU检查需要本地CUDA/PhysX及受限资产。

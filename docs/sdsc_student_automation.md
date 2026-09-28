# 已验收 teacher 的 student 预检与自动接续

当前自动接续只负责预检 `54506703` → 一次
`qwen3-v2-adapted-calibration` → 校准终态核验与小结果回传。
使用已独立接受的 v2 科学源码 `28c1026cece772a9e3d167d9cc64a64aaa0fd1b3`
和 teacher 资格作业 `54496291`。校准固定为 **2 H100 / 24 CPU /
192 GiB / 最长 2 小时**，原 canonical-SFT 全参数训练、seed 42、global
batch 64、1536-token 输入上限、120 步／200 万 token 上限及所有验收阈值
保持不变。校准成功不代表完整 G0、seed-42 pilot 或全因子实验完成。

用户已明确授权监控预检并通过后自动提交此训练阶段。流程不需要逐阶段
再次确认，但必须核验真正的计算结果。实时状态和下一步以
[current handoff](refactor/current_handoff.md) 和 `.sdsc/supervision/` 中的
真实计划、状态及提交回执为准；工具文件存在不代表已启动自动接续。

## 独立运行目录与真实源码来源

`tools/sdsc_release_replay.py` 为校准准备已有已审核快照的明确重放。
它只复制原快照的完整文件清单和相同字节，创建不同 `run_id` 的独立目录；
原预检使用的源码和结果不被覆盖。若当前 Quest 文件已经更新，工具必须
恢复并验证原字节，不能混入未审查的修改。所需文件仅限原本允许的有界
源码快照，不读取凭证、环境文件夹、模型权重或训练数据。

新 wrapper 保留原源码的真实 `git_head`；当前 Quest HEAD 单独记录为准备
操作的元数据，不被冒称为该源码版本。原 Git bundle 经过完整验证后保留
不变，新 provenance 只重新绑定新 wrapper/run。工具不移动 Git refs，
不 reset、checkout 或创建 worktree，也不修改现有科学接受记录。

顺序为本地准备和文件清单／大小 dry-run → 冻结计划 → 排他上传新 release
→ 验证新 provenance 上传回执。上传前记录状态；回执缺失时保留未知状态，
先核对远端实际文件，不能盲目重复部署。现有科学来源校验与 Slurm 提交
门槛不因重放而放宽。

## 有限 Quest 监控进程

`tools/sdsc_student_supervise` 只在 Quest 运行，不安装系统或用户服务，
也不在 SDSC 运行 Codex、编辑器服务器或常驻控制进程。
启动前，校准的新 release 与 provenance 必须部署并核验完毕。
计划固定当前 Quest 根目录和主机、控制工具哈希、预检回执、科学协议、
teacher 身份、运行时、校准资源及结果位置。

流程每 **300 秒**查询一次，最多运行 **14 天**；超期后不再开始新操作。
每次远程操作先检查运行时计算的
`$HOME/.ssh/cm/sdsc-$(hostname -s)`，仅以 BatchMode 复用既有认证。
共享连接必须在启动该监控的同一 Quest 主机上保持可用。普通认证终端
退出可能断开其前台 SSH master；监控进程与 Slurm 作业的存活不是 SSH
连接仍可用的证据。

预检必须满足真实 `sacct` 主作业及步骤 `COMPLETED / 0:0`、队列中已结束、
持久化报告与文件校验通过，以及 job/run/源码/协议/teacher 身份全部一致。
只从 `squeue` 消失、返回零退出码或存在一个报告都不够。通过后仅提交一次
固定校准任务；提交前持久记录已尝试提交，收到回执后保存真实 job ID。
全局阶段声明、互斥锁及已有 intent 检查防止两个监控或一次重启重复提交。

SSH 失效、控制文件变化、未知会计状态、失败验收或提交结果不确定都会
停止流程，保留原始状态和证据。不重新认证、不自动重试提交、不取消作业，
也不自动启动 G0、pilot、三种子全因子或 Gemma。校准完成后再核验会计及
实际结果，只把有限日志、报告与工件绑定元数据取回 `.sdsc/fetched/`；不下载权重、
大型 checkpoint 或反向覆盖 Quest 源码。

## 新会话检查

1. 先读 handoff、计划、`state.json`、启动记录及 `.sdsc/submissions/`。
   有活跃进程时不改其固定控制文件、不另起竞争监控。
2. 对已有作业使用 `tools/sdsc status JOB_ID`、
   `tools/sdsc logs JOB_ID --lines 60` 和必要的有界 fetch。
3. 如果状态是 `submitting_calibration`、`sending` 或 `unknown`，先用原 intent 核对
   是否提交成功。不得删除阶段声明、替换 run ID 或重新启动来绕过记录。
4. 停止后的流程须先明确停止原因和实际 Slurm 状态，再设计独立恢复步骤。
   恢复 SSH 本身不会自动重启停止的监控。

该进程会把变化写入项目状态和日志。不要将它描述为会自动唤醒 Codex
会话或发送聊天通知；其职责是核验、受限提交和结果回传。

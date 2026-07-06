# 方案二：Comp-ACT 模仿学习实施方案

> 创建日期：2026-07-03 ｜ 状态：方案定稿，待启动
> 定位：在方案一力控底层之上，用"遥操作演示 + ACT 变刚度策略"替换手工搜孔/插入状态机，完成同样三任务（插头插入/拔出、螺丝入螺母）
> 配套文档：[方案一详细实施方案.md](方案一详细实施方案.md)（底层架构不变）｜[项目进度与代码计划.md](项目进度与代码计划.md)（任务 #1–36）｜[调研文档](双臂协同装配方案调研与实现路线.md)（路线三论证）

---

## 1. 调研结论（2026-07-03）

### 1.1 Comp-ACT 开源代码现状

仓库：https://github.com/omron-sinicx/CompACT （MIT 许可，IROS 2024）

| 项目 | 事实 | 对我们的含义 |
|------|------|-------------|
| 结构 | `/config` 配置 + `/libs`（基于 ACT/DETR）+ `/scripts`（训练/评估） | 核心算法代码量小，易读易移植 |
| 数据格式 | 支持 HDF5 和 LeRobot 格式（2024 年的旧版 v2.x） | 与现行 LeRobot v3.0 不兼容，需转换 |
| 仿真 | 自维护的 Robosuite fork（双臂擦拭任务） | 我们不用其仿真 |
| 真机 | 双臂 UR5e + ROS + FDCC 柔顺控制器 + Vive VR 遥操作（遥操作在独立仓库） | **CB3 无法直接复用其真机栈**（无 ROS、无 e-series 控制接口） |
| 环境 | Python 3.8.10 | 偏老 |
| 活跃度 | 27 stars / 13 commits / 无 release | 视为"参考实现"而非"可依赖框架" |

**方法要点**（论文 [arXiv:2406.14990](https://arxiv.org/abs/2406.14990)）：

- 策略 = ACT（CVAE + Transformer 动作分块），动作向量在位姿之外**扩出 12 维刚度**（平动/转动刚度矩阵各取 Cholesky 上三角 6 维，连续值）；
- 执行层 = FDCC（阻抗/导纳/力控混合，ROS `cartesian_controllers`）；
- 观测 = 2–3 路 RGB（腕部 + 全局）+ 腕部六维力 + 笛卡尔位姿；
- 遥操作 = Vive VR 90Hz，**手柄按键在 Low=250 / Mid=500 / High=750 三档预设刚度间切换**（即：刚度标签来自离散模式切换，策略学出连续输出——这一点对我们的实现方式很关键）；力反馈 = 腕部力映射为手柄振动强度；
- 实验：双臂 UR5e 圆柱 peg-in-hole **20 条演示 100%**，方形 30 条 70%；单臂拾取+插入 30 条 70%；
- 训练开销：**RTX 3070（8GB）2–5 小时/任务**——GPU 门槛远低于 VLA。

### 1.2 LeRobot 现状

- 当前 **v0.4.x**，数据集格式 **[LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)**（Parquet + MP4 分块存储、流式加载，随 [v0.4.0 发布](https://huggingface.co/blog/lerobot-release-v040)，提供 v2.1→v3.0 转换脚本）；
- 内置策略：ACT、Diffusion Policy、VQ-BeT、TDMPC、HIL-SERL、π0/π0.5、GR00T N1.5、SmolVLA 等；
- 硬件接入走 **Robot / Teleoperator 插件基类**，社区已有 [UR5e + GELLO + RTDE 的完整插件参考实现](https://github.com/F-Fer/lerobot_ur5e_gello)（含数据采集、训练、部署全流程，Python ≥3.11）；
- 结论：**LeRobot 是当前事实标准，数据一次采成 v3.0 格式，训练 ACT 零框架改动。**

### 1.3 关键判断

Comp-ACT 相对普通 ACT 的**全部增量就是"动作向量多 N 维刚度 + 执行层接受时变刚度"**。而 LeRobot 的 ACT 对动作维度无假设（任意维向量都能训）。因此：

> **不整体采用 CompACT 仓库，而是"LeRobot v0.4 为主干 + 在动作向量中扩入刚度维 + 我们已有的导纳层做执行器"。** CompACT 仓库仅作为超参和实现细节的对照参考。

这样做的额外好处：我们的 `AdmittanceController.set_params()` 本来就支持运行中热切参数——**执行层几乎零改动**。

---

## 2. 总体架构

```
┌─ 采集期 ─────────────────────────────────────────────────────┐
│ SpaceMouse(位姿增量+模式切换键)                                │
│   └→ teleop 主环 125Hz: ref_pose 更新 + 导纳叠加 + 安全层      │
│         └→ UR5 servoL   （UR10 呈现臂: 每 episode 随机呈现位姿）│
│ 同步记录 25Hz → LeRobotDataset v3.0:                          │
│   obs  = 腕部RGB + 全局RGB + TCP位姿 + wrench_H + 夹爪         │
│   action = 目标TCP位姿(9D) + 刚度k/kr(6D) + 夹爪(1D) = 16D     │
└──────────────────────────────────────────────────────────────┘
┌─ 训练期 ─────────────────────────────────────────────────────┐
│ lerobot ACT (chunk=50, CVAE) ← 30–50 条演示/任务, 3070 级 GPU  │
└──────────────────────────────────────────────────────────────┘
┌─ 部署期 ─────────────────────────────────────────────────────┐
│ 策略推理 ~10Hz 出动作块 → temporal ensemble 平滑               │
│   → 125Hz 执行环: 目标位姿插值 + set_params(k) 热切            │
│   → 导纳叠加 + SafetyMonitor(原封不动) → servoL                │
└──────────────────────────────────────────────────────────────┘
```

**复用清单**：drivers/ 全部、control/ 全部（admittance/safety/task_frame/primitives 的力管线）、calib/ 全部、方案一 M0–M2 的全部标定与验收成果、`scripts/evaluate.py` 评测协议。
**替换对象**：`tasks/insertion.py` 状态机与 `admittance.yaml` 中需真机调参的搜孔参数（保留作 baseline 对比）。

### 2.1 设计决策表

| # | 决策 | 选择 | 理由与备选 |
|---|------|------|-----------|
| D1 | 训练框架 | LeRobot v0.4.x（pin 版本），ACT 策略 | 数据格式/训练/部署一条龙；CompACT 仓库仅参考。备选：直接用 CompACT 仓库（旧格式+低维护，不选） |
| D2 | 刚度表示 | **6D 对角刚度**（k 3维 + kr 3维，log 空间归一化），阻尼按临界阻尼 d=2√(k·m) 派生，m 固定 | 我们的导纳是逐通道对角的，无需 Comp-ACT 的 12D Cholesky 全矩阵；表达能力足够（Comp-ACT 消融未显示全矩阵必要性） |
| D3 | 刚度标签来源 | 遥操作时按键在 3 组预设（free/search/insert，即 admittance.yaml 现有模式）间切换，记录当刻 k 值 | 与 Comp-ACT 的 Low/Mid/High 档位切换完全同构，已被验证可行 |
| D4 | 遥操作设备 | **SpaceMouse Compact**（已在采购清单）；GELLO 为升级备选 | 插入任务动作幅度小、以笛卡尔微调为主，SpaceMouse 够用且与导纳参考位姿输入天然匹配；若演示效率不满意再上 GELLO |
| D5 | 动作空间 | 世界系绝对 TCP 位姿：xyz(3) + rot6d(6) + 刚度(6) + 夹爪(1) = 16D | rot6d 避免轴角/四元数不连续；绝对位姿与 Comp-ACT 一致 |
| D6 | 观测状态 | TCP 位姿 xyz+rot6d(9) + {H}系补偿后 wrench(6) + 夹爪(1) = 16D；图像 2 路（腕部 D405 + 全局 D435，428×240@25Hz） | 力觉输入是 Comp-ACT 消融证明的关键项（100% vs 40%） |
| D7 | 频率 | 采集/策略 25Hz（125Hz 主环每 5 拍记 1 帧），chunk=50（2 秒），推理 ~10Hz + temporal ensemble | ALOHA/Comp-ACT 惯例折算到 CB3 的 125Hz |
| D8 | UR10 角色 | 每 episode 在呈现位姿上叠 ±5mm/±5° 随机扰动后**静止**，不遥操作 | 任务非对称；扰动即数据随机化，教会策略纠偏 |
| D9 | 安全 | SafetyMonitor 每周期检查不变；策略目标限幅 + 块间平滑防突跳 | 部署期策略输出经过与手工状态机相同的安全层 |

---

## 3. 新增代码模块设计

> **隔离原则（硬性约束）**：方案一代码（drivers/ control/ common/ calib/ perception/ tasks/ scripts/ config/）处于"待真机验收"冻结状态，**一个字都不改**。方案二全部代码放独立顶层包 `il/`，对方案一模块**只读 import**；需要扩展行为一律用包装/子类实现在 `il/` 内；配置、日志目录全部独立。

```
il/                     # ← 方案二唯一落点，方案一目录零改动
  adapters.py           # 对方案一的全部适配(不改原文件):
                        #   params_from_stiffness(k,kr)→AdmittanceParams(临界阻尼派生)
                        #   导纳/安全层/驱动的只读组装(仿 common/setup.py 但独立实现)
  config_il.yaml        # 采集/训练/部署全部参数(频率/分辨率/chunk/归一化/刚度档位)
  teleop/
    spacemouse.py       # hidapi 读 SpaceMouse: 6DOF 速度 + 按键; 死区+缩放
    teleop_loop.py      # 125Hz: ref_pose += spacemouse增量; 键切换刚度模式;
                        #        导纳叠加(演示自带柔顺) + 安全层 + servoL
  learning/
    robot_dualarm.py    # LeRobot Robot 插件: DualArmUR(观测/动作 schema 如 D5/D6)
    cameras.py          # D405/D435 双路封装(import drivers/camera.py), 时间戳对齐
    record_episodes.py  # 采集入口: UR10随机呈现 → 遥操作演示 → 存 v3.0;
                        #        按键: 开始/结束/废弃本条/标记成功
    convert_h5.py       # 方案一 HDF5 日志 → LeRobotDataset v3.0 (兼做原任务#36)
    train_compact.py    # lerobot ACT 训练封装: 16D 动作、力觉输入、归一化统计
    deploy_policy.py    # 部署执行器: 10Hz 推理 → chunk 缓冲 + temporal ensemble
                        #        → 125Hz 插值下发 + set_params(k) + SafetyMonitor
  logs_il/              # 独立日志/数据集输出目录(不与 logs/ 混放)
```

估计代码量：teleop ~300 行，learning ~800 行，adapters ~150 行，均为纯 Python，无 ROS 依赖。
对方案一的 import 面收窄到四个稳定接口：`drivers.ur_arm.URArm`、`drivers.ft_sensor.FTReader/GravityCompensator`、`control.admittance.AdmittanceController/AdmittanceParams`、`control.safety.SafetyMonitor`（外加 `drivers.camera/gripper`）——这些类接口一旦方案一真机测试期间有变，只需改 `il/adapters.py` 一处。

### 3.1 运行环境（与方案一同一台 Ubuntu 22.04 控制 PC）

全部采集/训练/部署都在 **Ubuntu 22.04 真机控制 PC** 上运行（Windows 开发机仍只跑离线自检），无新增操作系统要求：

| 组件 | Ubuntu 22.04 上的方案 | 备注 |
|------|----------------------|------|
| 控制环境（现有） | 系统 Python 3.10 venv：ur_rtde/numpy/scipy/pymodbus/h5py | 方案一原环境不动 |
| 学习环境（新增） | conda 或 uv 建 **Python 3.11 独立环境**：lerobot v0.4.x + torch + CUDA | 与控制环境隔离，互不污染依赖 |
| 双环境衔接 | 部署时策略进程（3.11 环境）↔ 控制进程（3.10 环境）经 localhost socket/共享内存传动作块 | 顺带规避 GIL 与推理抖动干扰 125Hz 实时环 |
| SpaceMouse | `pyspacemouse`/hidapi + udev 规则（免 root 读 HID） | Linux 支持成熟，比 Windows 驱动省事 |
| 相机 | librealsense2 apt 源 + pyrealsense2 | 22.04 官方支持 |
| GPU | 控制 PC 若插了 ≥8G 显卡可本机训练；否则数据集拷到实验室训练机/云端，产物只有 checkpoint 文件 | 训练与部署可分机，推理本机需 GPU（ACT 模型小，8G 够） |
| 实时性 | 与方案一同样要求：CPU 性能模式、控制进程 `chrt`/taskset 绑核；PREEMPT_RT 非必需（125Hz 周期 8ms 裕量大） | 环境手册已有的设置继续沿用 |

---

## 4. 硬件增补清单（在方案一采购单之外）

| 状态 | 项目 | 说明 | 阻塞 |
|------|------|------|------|
| ⬜ | SpaceMouse Compact | **方案一采购单已列**，提前到现在买 | IL-1 |
| ⬜ | 全局相机 D435 ×1 | 俯视呈现区；腕部 D405 已在单上 | IL-2 |
| ⬜ | GPU：RTX 3070 8G 即可训练，3090/4090 更稳 | 实验室已有则无需购置；也可云端训练 | IL-3 |
| ⬜ | 相机支架/龙门 | 全局相机固定——**位置一旦定死不许再动**（动了数据作废） | IL-2 |

---

## 5. 分阶段实施计划（任务 #37–50，接续进度看板）

> 前置依赖：方案一 **M0（通讯）、M1（标定）、M2（力控可用）必须完成**——遥操作以导纳为底层。
> M3（手工状态机调参）不再是必经路径，但**保留作论文 baseline**，可与 IL-2 并行做。

**—— IL-0：软件管线搭建（无硬件依赖，现在就能开工）——**

| # | 任务 | 内容 | 验收 | 预估 |
|---|------|------|------|------|
| 37 | LeRobot 环境 + 格式打通 | 独立 venv 装 lerobot v0.4.x；跑通官方 ACT 训练示例；读写 v3.0 数据集 | 示例训练收敛 | 1天 |
| 38 | `convert_h5.py` 转换器 | 方案一 HDF5 日志 → v3.0（含视频占位）；兼收原 #36 | 转换后 lerobot 可加载可视化 | 1–2天 |
| 39 | `robot_dualarm.py` 插件 + 观测/动作 schema | 按 D5/D6 定 schema；无硬件时以 mock 驱动跑通 record 流程 | 离线自检：录 mock 数据→训练冒烟 | 2–3天 |
| 40 | `train_compact.py` + 16D 动作管线自测 | 用 CompACT 仓库公开数据或合成数据验证"位姿+刚度"扩维动作可训、损失下降 | 冒烟训练全绿 | 2天 |

**—— IL-1：遥操作（依赖 M2 + SpaceMouse 到货）——**

| # | 任务 | 内容 | 验收 | 预估 |
|---|------|------|------|------|
| 41 | `spacemouse.py` 驱动 | hidapi 读数、死区/缩放标定、按键映射（模式切换/夹爪/录制控制） | 读数稳定 125Hz 无丢帧 | 1天 |
| 42 | `teleop_loop.py` 遥操作主环 | SpaceMouse 增量 → ref_pose，导纳叠加，模式热切；⚠ 速度滑块 30% 首测 | 手感顺滑；能徒手遥操作完成 1 次插入 | 2–3天 |

**—— IL-2：数据采集 ——**

| # | 任务 | 内容 | 验收 | 预估 |
|---|------|------|------|------|
| 43 | 采集协议 + 试录 | 固定相机位并拍照存档；写采集 SOP（扰动范围/失败处理/命名）；试录 10 条并回放审查 | 回放图像/力/动作对齐无跳变 | 1–2天 |
| 44 | 正式采集 | 每任务 30–50 条（插入 40 / 拔出 30 / 螺丝 40 起步）；UR10 呈现位姿 ±5mm/±5° 随机；操作节奏多样化 | 数据集通过质检脚本（时长/力峰值/成功标记） | 3–5天 |

**—— IL-3：训练与离线评估 ——**

| # | 任务 | 内容 | 验收 | 预估 |
|---|------|------|------|------|
| 45 | 训练 | 每任务两版：①普通 ACT（无刚度维，导纳参数固定）②Comp-ACT 版（16D）；各 ~2万步 | 损失收敛，验证集动作误差 <阈值 | 2天 |
| 46 | 离线回放评估 | 开环回放对比演示轨迹；检查刚度输出是否在接触段自动降低（定性正确性） | 回放曲线合理 | 1天 |

**—— IL-4：部署与评测 ——**

| # | 任务 | 内容 | 验收 | 预估 |
|---|------|------|------|------|
| 47 | `deploy_policy.py` 执行器 | chunk 缓冲 + temporal ensemble + 125Hz 插值 + set_params 热切 + 安全层；首跑 ⚠ 30% 滑块 | 单 episode 全流程真机跑通 | 2–3天 |
| 48 | 真机评测 | 复用 `evaluate.py` 协议：±3mm/±3° 扰动 ×30 次/任务 | 插入 ≥90%、拔出 ≥95%、螺丝 ≥85%（与方案一同标准） | 2天 |
| 49 | 对比实验（论文素材） | 方案一 vs ACT vs Comp-ACT 同协议对比；消融：去力觉输入 / 去变刚度 | 对比报表 | 3天 |
| 50 | （可选）失败回收再训练 | 失败 episode 人工接管纠正后并入数据集重训（DAgger-lite） | 成功率提升 | 按需 |

**里程碑**：M7 = #42 遥操作可用 ｜ M8 = #47 首个策略真机跑通 ｜ M9 = #48–49 三任务达标 + 对比报告

时间线（硬件与 M2 就绪后起算）：IL-0 可提前完成；IL-1→IL-4 约 **4–6 周**。

---

## 6. 数据采集协议要点（#43 展开）

1. **随机化**：UR10 呈现位姿逐 episode 均匀扰动 ±5mm/±5°（略大于评测的 ±3mm/±3°）；工件在爪中姿态每 5–10 条微调一次；
2. **多样性**：演示者刻意变化接近方向与速度；包含少量"先接触偏位再纠偏"的演示（教会搜索行为）；
3. **刚度切换纪律**：接近段 free（低刚度快速跟随）、接触/搜索段 search、插入段 insert——切换时机自然即可，无需精确；
4. **每条 episode 记录**：成功标志、任务名、扰动真值（供事后分析）；失败/手滑的条目当场废弃重录；
5. **冻结项**（改动即数据作废，从 #43 起生效）：相机内外参与安装位、TCP/夹爪安装、传感器安装矩阵、图像分辨率与裁剪。

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| CB3 125Hz + servoL 延迟劣于 Comp-ACT 的 e-series | 高刚度段可能振荡 | M2 验收就是闸门；刚度上限 clamp 到 admittance.yaml 已验证范围 |
| SpaceMouse 单手 6DOF 姿态控制难 | 演示效率低 | 插入演示以 3–4 DOF 为主；启用分轴模式；不行升级 GELLO（几千元） |
| 演示质量参差 | 策略学到坏习惯 | 试录审查 + 质检脚本 + 当场废弃机制（#43） |
| 策略动作块切换突跳 | 触发 max_servo_jump 或真突跳 | temporal ensemble + 相邻块插值 + 目标限幅；安全层兜底 |
| lerobot API 迭代快 | 管线断裂 | pin v0.4.x + dataset v3.0；升级另开分支验证 |
| 腕部相机插入时被遮挡 | 视觉信息缺失 | 全局相机补视角（Comp-ACT 同款配置）；这正是力觉输入的价值 |
| 力信号训练/部署分布偏移 | 成功率下降 | 演示与部署共用同一导纳底层与重力补偿（架构上已保证） |
| 刚度标签只有 3 档 | 学不出精细变刚度 | Comp-ACT 同为 3 档标签、连续输出，已验证可行；不足时增加档位 |

---

## 8. 与方案一的关系（论文视角）

- 方案一 = **baseline**（传统管线：视觉定位 + 手工状态机 + 固定参数导纳）；
- ACT（无力觉/固定刚度）与 Comp-ACT（力觉 + 变刚度）= 两级学习方法；
- 同平台、同评测协议（`evaluate.py`，±3mm/±3° × 30 次）三方对比 + 双消融，即调研存档中"方向 B：同平台 IL vs 传统管线对比"的完整实验矩阵；
- 后续可叠加：方向 A（呈现位姿优化，采数据时顺带做倾角扫描）与 VLA 课题（见记忆存档，数据格式已兼容）。

## 9. 参考资源

- Comp-ACT：[论文](https://arxiv.org/abs/2406.14990) ｜ [代码](https://github.com/omron-sinicx/CompACT) ｜ [项目页](https://omron-sinicx.github.io/CompACT/)
- LeRobot：[仓库](https://github.com/huggingface/lerobot) ｜ [Dataset v3.0 文档](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) ｜ [v0.4.0 发布说明](https://huggingface.co/blog/lerobot-release-v040)
- UR + GELLO 插件参考实现：[F-Fer/lerobot_ur5e_gello](https://github.com/F-Fer/lerobot_ur5e_gello)
- ACT 原始实现：[act-plus-plus](https://github.com/MarkFzp/act-plus-plus)

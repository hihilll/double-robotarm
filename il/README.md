# il/ — 方案二：Comp-ACT 模仿学习

方案文档：[../方案二_CompACT模仿学习实施方案.md](../方案二_CompACT模仿学习实施方案.md)（任务 #37–50）

> **隔离约束**：方案一代码处于真机验收冻结状态。本包对其**只读 import、零修改**，
> 唯一接口面在 `adapters.py`。配置用 `config_il.yaml`，输出全部进 `logs_il/`。

## 模块

```
adapters.py         方案一只读适配(刚度→导纳参数派生/配置加载/组装转发)
action_repr.py      16D 观测/动作向量编解码(pos+rot6d+刚度+夹爪)
config_il.yaml      采集/训练/部署全部参数
teleop/spacemouse.py     SpaceMouse 输入(#41)
teleop/teleop_loop.py    125Hz 遥操作主环: 导纳叠加+刚度档切换(#42)
learning/cameras.py      腕部D405+全局D435 双路采集
learning/robot_dualarm.py 硬件聚合 + LeRobotDataset schema/写入兼容层(#39)
learning/record_episodes.py 演示采集入口(#43/44)
learning/convert_h5.py   方案一HDF5日志→LeRobot v3.0(#38, 兼原#36)
learning/train_compact.py 组装 lerobot-train 命令(#40/45)
learning/deploy_policy.py 策略服务+125Hz实时执行器(#47)
_selftest_il.py     离线自检(无硬件/无 lerobot 可跑)
```

## 环境（Ubuntu 22.04 控制 PC，任务#37）

两个环境并存，互不污染：

```bash
# ① 控制环境(方案一已有, Python 3.10): 不动。
#    il 的 teleop / deploy --run 在此环境跑，另需:
pip install pyspacemouse            # + sudo apt install libhidapi-dev
# udev 规则(免 root 读 SpaceMouse):
sudo tee /etc/udev/rules.d/99-spacemouse.rules <<'EOF'
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger

# ② 学习环境(新建, Python 3.11): record/convert/train/deploy --serve 在此跑
conda create -n il python=3.11 -y && conda activate il
pip install "lerobot==0.4.*" torch torchvision   # CUDA 版 torch 按显卡装
pip install ur-rtde scipy pyyaml h5py opencv-python pyrealsense2 pyspacemouse
```

说明：采集脚本(单进程双线程: 125Hz 控制 + 25Hz 写盘)整体跑在**学习环境**
（它需要 lerobot 写数据集，又需要 ur_rtde 控臂，所以两者都装）；
部署时才是双进程跨环境（`--serve` 学习环境 / `--run` 控制环境）。

## 使用顺序（对应任务号）

```bash
python il/_selftest_il.py                                   # 改完配置/代码先自检
# --- 无硬件(IL-0) ---
python il/learning/convert_h5.py --h5 logs/*.h5 --repo dualarm_smoke   # #38
python il/learning/train_compact.py --repo dualarm_smoke --dry-run     # #40
# --- 遥操作(IL-1, 依赖方案一 M2 + SpaceMouse) ---
python il/teleop/teleop_loop.py --payload none              # #42 手感(⚠30%滑块)
# --- 采集(IL-2) ---
python il/learning/record_episodes.py --task plug_insert --episodes 40 # #43/44
# --- 训练(IL-3) ---
python il/learning/train_compact.py --repo dualarm_plug_insert         # #45
# --- 部署(IL-4) ---
python il/learning/deploy_policy.py --serve --ckpt il/logs_il/train/<run>/checkpoints/last/pretrained_model
python il/learning/deploy_policy.py --run --task plug_insert           # #47
```

## 真机前必须核对（标 ⚠/TODO 处）

1. `config_il.yaml`：SpaceMouse `axis_sign` 逐轴核对；相机 serial；刚度三档真机调
2. 刚度上限以方案一 M2（手推无振荡）验收结果为准，`stiffness_clamp` 同步收紧
3. lerobot 小版本差异：`lerobot-train --help` 核对参数名（train_compact.py 头注释）
4. 采集纪律：相机位/TCP/传感器安装从 #43 起冻结，变动=数据作废

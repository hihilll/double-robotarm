# 双臂协同插拔/装配（UR5 + UR10, CB3）

方案与背景文档：
- [双臂协同装配方案调研与实现路线.md](双臂协同装配方案调研与实现路线.md) — 文献调研、路线选型
- [方案一详细实施方案.md](方案一详细实施方案.md) — 系统架构、通讯（§1.4）、六阶段计划、参数表、采购
- [项目进度与代码计划.md](项目进度与代码计划.md) — 进度看板（任务#1–36、里程碑 M0–M6）
- [环境配置与测试手册.md](环境配置与测试手册.md) — **环境搭建 + 现阶段/到货后逐步测试流程（新人从这里开始）**

## 代码结构

```
config/       robots.yaml(硬件/标定参数) + admittance.yaml(导纳/安全/任务/状态机参数)
common/       setup(系统装配工厂) / logger(125Hz HDF5 日志)
drivers/      ur_arm(RTDE封装) / ft_sensor(采集+重力补偿) / gripper(大寰Modbus) / camera(D405) / bus
control/      task_frame({H}系变换) / admittance(导纳) / safety(安全层) / primitives(6原语+125Hz主环)
calib/        payload_id(负载辨识) / dual_arm_calib(共触点法) / handeye(手眼)
perception/   tag_locator(AprilTag→{H})
tasks/        insertion(通用插入状态机) / plug_extract / run_bench(桌面调试) / dual_arm_manager(双臂主控)
scripts/      验收脚本 + evaluate(自动评测) + plot_log + _selftest_*(离线回归测试)
```

**代码已全部完成**（含自检），到货后按 `项目进度与代码计划.md` 的任务编号依次真机验收。

## 到货后执行顺序（详见进度计划 §3.2）

```bash
pip install -r requirements.txt
python scripts/_selftest_wiring.py   # 改完 IP/配置先跑离线自检
python scripts/check_comm.py         # #13 通讯(只读)
python scripts/jog_test.py --arm ur5 # #14 点动(⚠ 速度滑块30%, 手扶急停)
python scripts/workspace_check.py --center 0.45 0 0.25   # #15 定呈现位姿
python calib/payload_id.py --payload none                # #16 负载辨识→回填yaml
python scripts/demo_admittance.py    # #21 手推柔顺
python scripts/test_hold_force.py    # #22 恒力±1N → M2
python calib/dual_arm_calib.py       # #19 双臂标定
python tasks/run_bench.py --task plug_insert --H <实测孔位> --repeat 20 --perturb-mm 3  # #25–28 → M3
python calib/handeye.py              # #20 手眼
python tasks/dual_arm_manager.py --task plug_insert --vision  # #29–31
python scripts/evaluate.py --task plug_insert --n 30          # #34 → M6
```

## 真机前必须核对（代码中标 ⚠/TODO）

1. `robots.yaml` 所有 TODO 项（TCP 标定、传感器安装矩阵、负载参数、双臂标定）
2. 力传感器 wrench 符号约定 & 量纲（见 `control/primitives.py` 头注释）
3. 夹爪 Modbus 寄存器映射（`drivers/gripper.py`，按到货手册核对）
4. UR 示教器：安装 TCP/负载、安全平面、速度滑块 30% 起

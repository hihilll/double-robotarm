"""策略部署（任务#47）：双进程 = 策略服务(学习环境) + 实时执行器(控制环境)。

  终端A(学习env):  python il/learning/deploy_policy.py --serve --ckpt <训练输出目录>
  终端B(控制env):  python il/learning/deploy_policy.py --run --task plug_insert --seconds 30

执行器 125Hz 主环(与遥操作同构, 策略替代人):
  推理线程 ~10Hz: 组观测 → socket → 服务端 ACT 推理 → 动作块(K,16)
  主环每拍: temporal ensemble 平滑 → 解包(ref位姿+刚度+夹爪) → 刚度限幅热切
           → 导纳叠加 → 相邻目标限幅(max_step_mm) → 方案一安全层 → servoL
⚠ 首跑: 速度滑块 30%、手扶急停; 建议先 --no-ur10 台架验证。
协议: localhost TCP + pickle(长度前缀)——仅限本机回环, 勿暴露外网。
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from il import action_repr as ar
from il.adapters import clamp_stiffness, load_configs, load_il_config, params_from_stiffness

# ------------------------------------------------------------ socket 协议
def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock: socket.socket):
    hdr = _recv_exact(sock, 4)
    return pickle.loads(_recv_exact(sock, struct.unpack(">I", hdr)[0]))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("对端关闭")
        buf += chunk
    return buf


# ------------------------------------------------------------ 时间集成
class TemporalEnsembler:
    """ACT 式动作块平滑: 同一时刻被多个重叠块覆盖时按 exp(-m·rank) 加权,
    最旧的块权重最高(ACT 论文约定, 换块瞬间不跳变)。时间单位 = 25Hz 步。"""

    def __init__(self, m: float, chunk_len: int):
        self.m = m
        self.K = chunk_len
        self.chunks: list[tuple[int, np.ndarray]] = []   # (起始步, (K,D))
        self._lock = threading.Lock()

    def add_chunk(self, t0: int, chunk: np.ndarray) -> None:
        with self._lock:
            self.chunks.append((int(t0), np.asarray(chunk, float)))
            self.chunks = [(s, c) for s, c in self.chunks if s + self.K > t0]  # 剪过期

    def action_at(self, t: float) -> np.ndarray | None:
        """t 可为小数步——相邻两整数步的集成结果线性插值(供 125Hz 环取值)。"""
        lo = int(np.floor(t))
        a0 = self._at_step(lo)
        if a0 is None:
            return None
        a1 = self._at_step(lo + 1)
        if a1 is None:
            return a0
        f = t - lo
        return (1 - f) * a0 + f * a1

    def _at_step(self, s: int) -> np.ndarray | None:
        with self._lock:
            preds = [c[s - t0] for t0, c in self.chunks if 0 <= s - t0 < self.K]
        if not preds:
            return None
        w = np.exp(-self.m * np.arange(len(preds)))      # preds 天然按旧→新排列
        return np.average(np.stack(preds), axis=0, weights=w)


# ------------------------------------------------------------ 策略服务端
def serve(ckpt: str, host: str, port: int) -> None:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(ckpt)
    policy.eval()
    device = next(policy.parameters()).device
    chunk_len = int(policy.config.chunk_size)
    print(f"[serve] 已加载 {ckpt} (chunk={chunk_len}, device={device})")

    def to_batch(obs: dict) -> dict:
        batch = {}
        for key, val in obs.items():
            v = torch.as_tensor(np.asarray(val))
            if v.ndim == 3:                     # HWC uint8 → 1CHW float
                v = v.permute(2, 0, 1).float() / 255.0
            v = v.unsqueeze(0).to(device)
            batch[key] = v.float() if v.dtype is torch.float64 else v
        return batch

    @torch.no_grad()
    def infer_chunk(obs: dict) -> np.ndarray:
        batch = to_batch(obs)
        if hasattr(policy, "predict_action_chunk"):
            return policy.predict_action_chunk(batch)[0].cpu().numpy()
        policy.reset()                          # 兜底: 队列式 API, 单次前向逐个弹出
        return np.stack([policy.select_action(batch)[0].cpu().numpy()
                         for _ in range(chunk_len)])

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[serve] 监听 {host}:{port}, 等待执行器...")
    while True:
        conn, addr = srv.accept()
        print(f"[serve] 执行器接入 {addr}")
        try:
            send_msg(conn, {"chunk_len": chunk_len})
            while True:
                obs = recv_msg(conn)
                t = time.monotonic()
                chunk = infer_chunk(obs)
                send_msg(conn, chunk)
                print(f"\r[serve] 推理 {1000 * (time.monotonic() - t):.0f} ms", end="")
        except (ConnectionError, EOFError):
            print("\n[serve] 执行器断开, 等待重连")
        finally:
            conn.close()


# ------------------------------------------------------------ 实时执行器
def run(task_name: str, seconds: float, use_ur10: bool, payload: str) -> None:
    from control.admittance import AdmittanceController
    from control.task_frame import TaskFrame
    from il.learning.robot_dualarm import OBS_STATE, DualArmRig, image_key

    il_cfg = load_il_config()
    dep = il_cfg["deploy"]
    _, ac = load_configs()
    rig = DualArmRig(il_cfg, payload=payload, use_ur10=use_ur10)
    arm, prim = rig.stack.arm, rig.stack.prim
    task = TaskFrame(np.zeros(6))                 # 与采集一致: 世界系
    fps = il_cfg["freq_record"]

    sock = socket.create_connection((dep["host"], dep["port"]))
    chunk_len = recv_msg(sock)["chunk_len"]
    ens = TemporalEnsembler(dep["ensemble_m"], chunk_len)
    print(f"[run] 已连策略服务 (chunk={chunk_len})")

    if rig.ur10 is not None:
        rng = np.random.default_rng()
        rig.present_ur10(task_name, rng)

    state = {"grip": 1.0, "stop": False}
    t_start = time.monotonic()
    step_of = lambda: (time.monotonic() - t_start) * fps

    def infer_loop() -> None:                     # ~10Hz 观测→动作块
        period = 1.0 / dep["infer_hz"]
        while not state["stop"]:
            t0 = time.monotonic()
            w, _ = prim.wrench_H(task)
            obs = {OBS_STATE: ar.pack_state(arm.tcp_pose(), w, state["grip"])}
            if rig.cameras is not None:
                for name, img in rig.cameras.frames().items():
                    obs[image_key(name)] = img
            try:
                send_msg(sock, obs)
                ens.add_chunk(int(round(step_of())), recv_msg(sock))
            except ConnectionError:
                state["stop"] = True
                return
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    th = threading.Thread(target=infer_loop, daemon=True)
    th.start()
    while ens.action_at(0.0) is None and not state["stop"]:
        time.sleep(0.02)                          # 等首个动作块

    k_prev = None
    adm = AdmittanceController(params_from_stiffness(
        *clamp_stiffness([100] * 3, [4] * 3, il_cfg), il_cfg, ac["limits"]))
    prev_target = arm.tcp_pose()
    max_step = dep["max_step_mm"] * 1e-3
    input("⚠ 确认速度滑块30%、可随时急停, 回车开始执行...")
    print(f"[run] 执行 {seconds:.0f}s, Ctrl+C 提前终止")
    try:
        t_start = time.monotonic()                # 重置步计时(从确认后起算)
        while time.monotonic() - t_start < seconds and not state["stop"]:
            t0 = arm.init_period()
            a = ens.action_at(step_of())
            if a is None:
                arm.wait_period(t0)
                continue
            act = ar.unpack_action(a)
            k, kr = clamp_stiffness(act["k"], act["kr"], il_cfg)
            if k_prev is None or np.max(np.abs(k - k_prev)) > 1.0:
                adm.set_params(params_from_stiffness(k, kr, il_cfg, ac["limits"]),
                               keep_state=True)
                k_prev = k
            # 夹爪滞回: <0.4 闭合, >0.6 打开
            if rig.gripper is not None:
                if act["gripper"] < 0.4 and state["grip"] > 0.5:
                    state["grip"] = il_cfg["teleop"]["gripper_close_permille"] / 1000.0
                    rig.gripper.move(int(state["grip"] * 1000), wait=False)
                elif act["gripper"] > 0.6 and state["grip"] < 0.5:
                    state["grip"] = 1.0
                    rig.gripper.move(1000, wait=False)

            w, age = prim.wrench_H(task)
            dpos, drot = adm.step(w, arm.dt)
            target = task.compose_target(act["pose6"], dpos, drot)
            jump = target[:3] - prev_target[:3]   # 块切换限幅
            d = np.linalg.norm(jump)
            if d > max_step:
                target[:3] = prev_target[:3] + jump / d * max_step
            prim.safety.check(w, age, arm.tcp_pose(), target)
            arm.servo_l(target)
            prev_target = target
            arm.wait_period(t0)
    except KeyboardInterrupt:
        print("\n[run] 人工终止")
    finally:
        state["stop"] = True
        arm.servo_stop()
        sock.close()
        rig.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--ckpt", help="--serve: 训练输出的 checkpoint 目录")
    ap.add_argument("--task", default="plug_insert")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--no-ur10", action="store_true")
    args = ap.parse_args()

    il_cfg = load_il_config()
    if args.serve:
        assert args.ckpt, "--serve 需要 --ckpt"
        serve(args.ckpt, il_cfg["deploy"]["host"], il_cfg["deploy"]["port"])
    else:
        payload = args.payload or {"plug_insert": "plug", "plug_extract": "none",
                                   "screw_place": "screw"}[args.task]
        run(args.task, args.seconds, use_ur10=not args.no_ur10, payload=payload)


if __name__ == "__main__":
    main()

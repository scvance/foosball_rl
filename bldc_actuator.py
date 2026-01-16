from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import pybullet as p


def rpm_per_volt_to_Ke_SI(kv_rpm_per_v: float) -> float:
    """
    Convert Kv [rpm/V] -> Ke [V/(rad/s)].

    Ke = 60 / (2*pi*Kv)
    In SI, Kt [Nm/A] == Ke [V/(rad/s)] numerically.  (ideal motor)
    """
    if kv_rpm_per_v <= 0:
        raise ValueError("kv_rpm_per_v must be > 0")
    return 60.0 / (2.0 * math.pi * kv_rpm_per_v)


def sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


@dataclass
class BLDCParams:
    # electrical
    kv_rpm_per_v: float = 740.0
    vbus: float = 11.1               # volts (3S nominal ~11.1V)
    R: float = 0.10                  # ohms (effective phase/line model; tune)
    Imax: float = 30.0               # amps (controller/battery limit)

    # drivetrain
    gear: float = 4.0                # motor_speed / joint_speed for rotary; same meaning for linear mapping via rad_per_m
    efficiency: float = 0.90         # drivetrain efficiency (0..1)

    # friction at joint (in joint space)
    viscous: float = 0.0             # (Nm)/(rad/s) for rotary OR (N)/(m/s) for linear
    coulomb: float = 0.0             # Nm for rotary OR N for linear

    # safety
    torque_limit_joint: Optional[float] = None  # Nm for rotary OR N for linear


class BLDCServoBase:
    """
    Base class that:
      - reads joint state
      - computes motor-limited joint torque/force using a simple DC motor model:
          v = R*i + Ke*omega_m
          tau_m = Kt*i
        with voltage saturation to +/-Vbus and current clamp to +/-Imax
      - can run either:
          (A) outer-loop PD position servo -> desired joint torque/force
          (B) direct joint torque/force command
    """

    def __init__(
        self,
        robot_uid: int,
        joint_idx: int,
        params: BLDCParams,
        is_prismatic: bool,
        # Outer-loop gains (joint-space)
        kp: float,
        kd: float,
    ):
        self.robot_uid = robot_uid
        self.joint_idx = joint_idx
        self.p = params
        self.is_prismatic = is_prismatic

        self.Ke = rpm_per_volt_to_Ke_SI(self.p.kv_rpm_per_v)
        self.Kt = self.Ke  # in SI units, ideal: Kt == Ke numerically :contentReference[oaicite:2]{index=2}

        self.kp = float(kp)
        self.kd = float(kd)

        # command state
        self.q_des: Optional[float] = None
        self.qd_des: float = 0.0
        self.u_ff: float = 0.0  # feedforward joint torque/force

        # instrumentation
        self.last_i_est: float = 0.0
        self.last_v_req: float = 0.0
        self.last_u_cmd: float = 0.0

    def set_position_target(self, q_des: float, qd_des: float = 0.0, u_ff: float = 0.0) -> None:
        self.q_des = float(q_des)
        self.qd_des = float(qd_des)
        self.u_ff = float(u_ff)

    def set_direct_effort(self, u_joint: float) -> None:
        """Bypass PD servo; command joint torque (rotary) or force (linear)."""
        self.q_des = None
        self.u_ff = float(u_joint)

    def _read_joint(self) -> Tuple[float, float]:
        js = p.getJointState(self.robot_uid, self.joint_idx)
        q = float(js[0])
        qd = float(js[1])
        return q, qd

    def _apply_joint_effort(self, u_joint: float) -> None:
        # For prismatic joints, PyBullet still uses TORQUE_CONTROL and expects "force=" as linear force.
        p.setJointMotorControl2(
            self.robot_uid,
            self.joint_idx,
            controlMode=p.TORQUE_CONTROL,
            force=float(u_joint),
        )

    def _outer_loop_pd(self, q: float, qd: float) -> float:
        if self.q_des is None:
            return self.u_ff
        e = (self.q_des - q)
        ed = (self.qd_des - qd)
        return self.kp * e + self.kd * ed + self.u_ff

    def _motor_limit_joint_effort(self, u_des_joint: float, qd_joint: float) -> float:
        """
        Convert desired joint effort -> motor-limited joint effort using:
          v = R*i + Ke*omega_m
          tau_m = Kt*i
        and gearbox mapping.

        For rotary:
          omega_m = gear * omega_joint
          tau_joint = tau_m * gear * eff  (ignoring reflected inertia; fine for control-feel)
        For linear:
          mapping handled in subclass via omega_m and tau_m conversions.
        """
        raise NotImplementedError

    def step(self) -> dict:
        """
        Call once per physics step.
        Returns a small dict of telemetry.
        """
        q, qd = self._read_joint()

        # desired joint effort from PD or direct
        u_des_joint = self._outer_loop_pd(q, qd)

        # add simple joint friction (opposes motion) in joint space
        u_des_joint -= self.p.viscous * qd
        u_des_joint -= self.p.coulomb * sign(qd)

        # optional joint effort clamp (hard safety)
        if self.p.torque_limit_joint is not None:
            lim = float(self.p.torque_limit_joint)
            u_des_joint = max(-lim, min(lim, u_des_joint))

        # motor electrical limiting
        u_cmd = self._motor_limit_joint_effort(u_des_joint, qd)

        # apply to pybullet
        self._apply_joint_effort(u_cmd)

        self.last_u_cmd = float(u_cmd)

        return {
            "q": q,
            "qd": qd,
            "u_des_joint": float(u_des_joint),
            "u_cmd_joint": float(u_cmd),
            "i_est": float(self.last_i_est),
            "v_req": float(self.last_v_req),
        }


class RotaryBLDCServo(BLDCServoBase):
    """Revolute joint: outputs joint torque [Nm]."""

    def __init__(self, robot_uid: int, joint_idx: int, params: BLDCParams, kp: float, kd: float):
        super().__init__(robot_uid, joint_idx, params, is_prismatic=False, kp=kp, kd=kd)

    def _motor_limit_joint_effort(self, tau_des_joint: float, omega_joint: float) -> float:
        # Map joint speed to motor speed
        omega_m = self.p.gear * omega_joint

        # Convert desired joint torque to motor torque
        # tau_joint ~= tau_m * gear * eff  => tau_m_des = tau_joint / (gear*eff)
        denom = max(1e-9, self.p.gear * self.p.efficiency)
        tau_m_des = tau_des_joint / denom

        # Desired motor current
        i_cmd = tau_m_des / max(1e-9, self.Kt)

        # Current clamp
        i_cmd = max(-self.p.Imax, min(self.p.Imax, i_cmd))

        # Voltage required
        v_req = self.p.R * i_cmd + self.Ke * omega_m
        self.last_v_req = float(v_req)

        # Voltage saturation -> adjust current
        if abs(v_req) > self.p.vbus:
            v_sat = sign(v_req) * self.p.vbus
            i_cmd = (v_sat - self.Ke * omega_m) / max(1e-9, self.p.R)
            i_cmd = max(-self.p.Imax, min(self.p.Imax, i_cmd))

        self.last_i_est = float(i_cmd)

        # Achievable motor torque
        tau_m = self.Kt * i_cmd

        # Map back to joint torque
        tau_joint = tau_m * self.p.gear * self.p.efficiency

        # final clamp
        if self.p.torque_limit_joint is not None:
            lim = float(self.p.torque_limit_joint)
            tau_joint = max(-lim, min(lim, tau_joint))

        return float(tau_joint)


class LinearBLDCServo(BLDCServoBase):
    """
    Prismatic joint: outputs joint force [N].

    Uses a linear transmission parameter:
      rad_per_m : motor radians per 1 meter of joint travel
    so:
      omega_m = qd_linear * rad_per_m
      tau_m  = F_linear / rad_per_m       (power match: tau*omega = F*v)
    """

    def __init__(
        self,
        robot_uid: int,
        joint_idx: int,
        params: BLDCParams,
        kp: float,
        kd: float,
        rad_per_m: float,
    ):
        super().__init__(robot_uid, joint_idx, params, is_prismatic=True, kp=kp, kd=kd)
        if rad_per_m <= 0:
            raise ValueError("rad_per_m must be > 0")
        self.rad_per_m = float(rad_per_m)

    def _motor_limit_joint_effort(self, F_des_joint: float, v_joint: float) -> float:
        # motor speed from linear speed
        omega_m = v_joint * self.rad_per_m * self.p.gear

        # desired motor torque from desired joint force
        # tau_m * omega_m ~= (F * v) / eff  => tau_m = (F / rad_per_m) / (gear*eff)
        denom = max(1e-9, self.rad_per_m * self.p.gear * self.p.efficiency)
        tau_m_des = F_des_joint / denom

        # desired current
        i_cmd = tau_m_des / max(1e-9, self.Kt)

        # current clamp
        i_cmd = max(-self.p.Imax, min(self.p.Imax, i_cmd))

        # voltage required
        v_req = self.p.R * i_cmd + self.Ke * omega_m
        self.last_v_req = float(v_req)

        # voltage saturation
        if abs(v_req) > self.p.vbus:
            v_sat = sign(v_req) * self.p.vbus
            i_cmd = (v_sat - self.Ke * omega_m) / max(1e-9, self.p.R)
            i_cmd = max(-self.p.Imax, min(self.p.Imax, i_cmd))

        self.last_i_est = float(i_cmd)

        # achievable motor torque
        tau_m = self.Kt * i_cmd

        # back to joint force
        # tau_joint_equiv = tau_m * gear * eff
        # F = tau_equiv * rad_per_m
        F_joint = (tau_m * self.p.gear * self.p.efficiency) * self.rad_per_m

        if self.p.torque_limit_joint is not None:
            lim = float(self.p.torque_limit_joint)
            F_joint = max(-lim, min(lim, F_joint))

        return float(F_joint)

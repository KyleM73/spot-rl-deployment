from collections import OrderedDict as odict
import enum
from typing import List

from copy import deepcopy
import glob
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
from operator import add
import os
from pathlib import Path
import pickle

from isaaclab.isaaclab_configuration import IsaaclabConfig
from isaaclab.isaaclab_constants import ordered_joint_names_isaaclab
from utils.dict_tools import dict_to_list


class Plotter:
    def __init__(
            self,
            config: IsaaclabConfig,
            file_regex: str,
            history_len: int = 1,
            estimate: bool = False
    ) -> None:
        self.config = config
        self.files = glob.glob(file_regex)
        self.H = history_len
        self.dt = 1. / 50.
        self.Ka = config.action_scale
        self.estimate_bool = estimate

        # Plotters
        subplot_kw_args = {
            "sharex": True,
            # "sharey": True,
        }
        # Velocity
        # self.fig_vel, self.ax_vel = plt.subplots(2, 1, **subplot_kw_args)
        self.fig_vel, self.ax_vel = plt.subplots()
        # Projected Gravity
        self.fig_proj_g, self.ax_proj_g = plt.subplots()
        # Velocity Command
        self.fig_vel_cmd, self.ax_vel_cmd = plt.subplots()
        # Joint Position
        self.fig_q, self.ax_q = plt.subplots(3, 1, **subplot_kw_args) # type: ignore
        # Joint Velocity
        self.fig_dq, self.ax_dq = plt.subplots(3, 1, **subplot_kw_args) # type: ignore
        # Action
        self.fig_action, self.ax_action = plt.subplots(3, 1, **subplot_kw_args) # type: ignore
        # # Torque
        # self.fig_torque, self.ax_torque = plt.subplots(3, 1, **subplot_kw_args)
        # # Estimate
        if self.estimate_bool:
            # self.fig_est, self.ax_est = plt.subplots(3, 1, **subplot_kw_args)
            self.fig_est, self.ax_est = plt.subplots()
            self.fig_f, self.ax_f = plt.subplots()
        # # Wrench
        # if store_dynamics:
        #     self.fig_wrench, self.ax_wrench = plt.subplots(2, 1, **subplot_kw_args)
        #     self.fig_grf, self.ax_grf = plt.subplots(3, 1, **subplot_kw_args)
        self.lw = 2

    def merge_data(self) -> None:
        objects = {}
        for file in self.files:
            with open(file, "rb") as f:
                try:
                    while True:
                        objects = {**objects, **pickle.load(f)}
                except EOFError:
                    pass

        objects = odict(sorted(objects.items()))
        self.t = []
        self.vel = {
            "vx": [],
            "vy": [],
            "vz": [],
            "avx": [],
            "avy": [],
            "avz": [],
        }
        self.vel_cmd = {
            "vx": [],
            "vy": [],
            "avz": [],
        }
        self.proj_g = {
            "gx": [],
            "gy": [],
            "gz": [],
        }
        # always in isaaclab order
        self.q = odict([(k, []) for k in ordered_joint_names_isaaclab])
        self.q_des = deepcopy(self.q)
        self.dq = deepcopy(self.q)
        self.action = deepcopy(self.q)
        if self.estimate_bool:
            self.estimates = {
                # "fl": {"x": [], "y": [], "z": []},
                # "fr": {"x": [], "y": [], "z": []},
                # "hl": {"x": [], "y": [], "z": []},
                # "hr": {"x": [], "y": [], "z": []},
                "fl": {"F": []},
                "fr": {"F": []},
                "hl": {"F": []},
                "hr": {"F": []},
            }
            self.estimates_f = {
                "Fx": [],
                "Fy": [],
                "Fz": [],
            }
        for k, v in objects.items():
            self.t.append(k * self.dt)
            obs = v["obs"]
            act = v["action"]
            if self.estimate_bool:
                est = v["estimate"]
            self.vel["vx"].append(obs[0])
            self.vel["vy"].append(obs[1])
            self.vel["vz"].append(obs[2])
            self.vel["avx"].append(obs[3*self.H])
            self.vel["avy"].append(obs[3*self.H+1])
            self.vel["avz"].append(obs[3*self.H+2])
            self.proj_g["gx"].append(obs[6*self.H])
            self.proj_g["gy"].append(obs[6*self.H+1])
            self.proj_g["gz"].append(obs[6*self.H+2])
            self.vel_cmd["vx"].append(obs[9*self.H])
            self.vel_cmd["vy"].append(obs[9*self.H+1])
            self.vel_cmd["avz"].append(obs[9*self.H+2])
            default_joints = dict_to_list(self.config.default_joints, ordered_joint_names_isaaclab)
            q = list(map(add, obs[12*self.H:12*self.H+12], default_joints))
            for idx, k in enumerate(self.q.keys()):
                self.q[k].append(q[idx])
                self.q_des[k].append(q[idx] + self.Ka*act[idx])
                self.dq[k].append(obs[24*self.H+idx])
                self.action[k].append(act[idx])
            if self.estimate_bool:
                for idx, k in enumerate(["Fx", "Fy", "Fz"]):
                    self.estimates_f[k].append(est[idx])
                for idx, k in enumerate(["fl", "fr", "hl", "hr"]):
                    # self.estimates[k]["x"].append(est[idx*3])
                    # self.estimates[k]["y"].append(est[idx*3+1])
                    # self.estimates[k]["z"].append(est[idx*3+2])
                    self.estimates[k]["F"].append(est[3+idx])

            print(len(self.estimates["fl"]["F"]))
            print(len(self.estimates_f["Fx"]))
            # assert False



    def plot(self, idx: List[int] | None = None) -> None:
        path = os.path.dirname(os.path.realpath(self.files[0]))
        if path[-1] != "/":
            path += "/"
        path += "plots/"
        path += self.files[0].split(".")[0].split("/")[-1] + "/"
        Path(path).mkdir(parents=True, exist_ok=True)
        if idx is None:
            idx_start, idx_end = 0, -1
        else:
            idx_start, idx_end = idx[0], idx[1]
            path += f"Step_{idx_start}-{idx_end}_"
        # set removes duplicates
        style_keys = list({k.split("_")[0] for k in self.q.keys() if len(k.split("_")) > 1})
        style_values = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        # style_values = ["royalblue", "mediumseagreen", "goldenrod", "tomato"]
        style_values = ["royalblue", "mediumseagreen", "tomato", "goldenrod"]
        styles = {
            k: v for k, v in zip(style_keys, style_values)
        }
        # Velocity
        idx_lin, idx_ang = 0, 0
        # idx_lin = 1
        for k, v in self.vel.items():
            # if "y" not in k: continue
            if "a" in k:
                continue
                # self.ax_vel[1].plot(
                #     self.t[idx_start:idx_end],
                #     v[idx_start:idx_end],
                #     c=style_values[idx_ang], label=k
                # )
                # if "z" in k:
                #     self.ax_vel[1].plot(
                #         self.t[idx_start:idx_end],
                #         self.vel_cmd["avz"][idx_start:idx_end],
                #         c=style_values[idx_ang], ls="--"
                #     )
                # idx_ang += 1
            else:
                ax = k[-1]
                # self.ax_vel[0].plot(
                self.ax_vel.plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c=style_values[idx_lin], label=rf'$v_{ax}$',
                    lw=self.lw,
                )
                if k in self.vel_cmd.keys():
                    # self.ax_vel[0].plot(
                    self.ax_vel.plot(
                        self.t[idx_start:idx_end],
                        self.vel_cmd[k][idx_start:idx_end],
                        c=style_values[idx_lin], ls="--"
                    )
                idx_lin += 1
        self.ax_vel.legend(loc="upper left")
        # self.ax_vel[1].legend(loc="upper right")
        self.ax_vel.set_title("Velocity")
        self.ax_vel.set_ylabel("Linear Velocity [m/s]")
        # self.ax_vel[1].set_ylabel("Angular Velocity [rad/s]")
        # self.ax_vel[1].set_xlabel("Time [s]")
        self.ax_vel.set_xlabel("Time [s]")
        self.fig_vel.savefig(path + "velocity.png")
        # Projected Gravity
        for k, v in self.proj_g.items():
            self.ax_proj_g.plot(
                self.t[idx_start:idx_end],
                v[idx_start:idx_end],
                label=k
            )
        self.ax_proj_g.legend(loc="upper right")
        self.ax_proj_g.set_title("Projected Gravity")
        self.ax_proj_g.set_ylabel("Force [N]")
        self.ax_proj_g.set_xlabel("Time [s]")
        self.fig_proj_g.savefig(path + "proj_g.png")
        # Velocity Command
        for k, v in self.vel_cmd.items():
            self.ax_vel_cmd.plot(
                self.t[idx_start:idx_end],
                v[idx_start:idx_end],
                label=k)
        self.ax_vel_cmd.legend(loc="upper right")
        self.ax_vel_cmd.set_title("Velocity Command")
        self.ax_vel_cmd.set_ylabel("Velocity [m/s, rad/s]")
        self.ax_vel_cmd.set_xlabel("Time [s]")
        self.fig_vel_cmd.savefig(path + "vel_cmd.png")
        # Joint Position
        for k, v in self.q.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_q[0].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_q[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_q[1].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_q[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_q[2].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_q[2].legend(loc="upper right")
        # Desired
        for k, v in self.q_des.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_q[0].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, ls="--"
                )
                self.ax_q[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_q[1].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, ls="--"
                )
                self.ax_q[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_q[2].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, ls="--"
                )
                self.ax_q[2].legend(loc="upper right")
        self.ax_q[0].set_title("Joint Position")
        self.ax_q[1].set_ylabel("Angle [rad]")
        self.ax_q[2].set_xlabel("Time [s]")
        self.fig_q.savefig(path + "q.png")
        # Joint Velocity
        for k, v in self.dq.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_dq[0].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_dq[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_dq[1].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_dq[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_dq[2].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_dq[2].legend(loc="upper right")
        self.ax_dq[0].set_title("Joint Velocity")
        self.ax_dq[1].set_ylabel("Angular Rate [rad/s]")
        self.ax_dq[2].set_xlabel("Time [s]")
        self.fig_dq.savefig(path + "dq.png")
        # Action
        for k, v in self.action.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_action[0].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_action[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_action[1].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_action[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_action[2].plot(
                    self.t[idx_start:idx_end],
                    v[idx_start:idx_end],
                    c, label=k.split("_")[0].upper()
                )
                self.ax_action[2].legend(loc="upper right")
        self.ax_action[0].set_title("Action")
        self.ax_action[1].set_ylabel("Angle Offset [rad]")
        self.ax_action[2].set_xlabel("Time [s]")
        self.fig_action.savefig(path + "action.png")
        # Estimates
        for k, v in self.estimates.items():
            c = styles[k.split("_")[0]]
            if "h" in k.lower():
                k = "r" + k[-1]
            for idx, (dir, val) in enumerate(v.items()):
                self.ax_est.plot(
                    self.t[idx_start:idx_end],
                    self.running_average(val[idx_start:idx_end], 5, clip=True),
                    c, label=k.upper(),
                    linewidth=self.lw,
                )
        self.ax_est.legend(loc="upper left")
        self.ax_est.set_title("GRF Estimate")
        self.ax_est.set_ylabel("F")
        # self.ax_est[0].set_ylabel("Fx")
        # self.ax_est[1].set_ylabel("Fy")
        # self.ax_est[2].set_ylabel("Fz")
        # self.ax_est[2].set_xlabel("Time [s]")
        self.ax_est.set_xlabel("Time [s]")
        self.fig_est.savefig(path + "estimate.png")
        for id, (k, v) in enumerate(self.estimates_f.items()):
            # if "y" not in k: continue
            k = k.upper()
            self.ax_f.plot(
                self.t[idx_start:idx_end],
                self.running_average(v[idx_start:idx_end], 5),
                label=rf'${k[0]}_{k[1].lower()}$', linewidth=self.lw,
                c=style_values[id]
            )
        self.ax_f.legend(loc="upper left")
        self.ax_f.set_title("Force Estimate")
        self.ax_f.set_ylabel("F [N]")
        self.ax_f.set_xlabel("Time [s]")
        self.fig_f.savefig(path + "estimate_f.png")

    def running_average(self, data, steps = 5, clip=False):
        data_padded = [data[0]]*(steps-1) + data
        data_smooth = []
        for i in range(len(data)):
            data_smooth.append(round(sum(data_padded[i:i+steps])/steps))
        if clip:
            data_smooth = [max(d,0) for d in data_smooth]
        return data_smooth

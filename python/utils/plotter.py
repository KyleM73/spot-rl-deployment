from __future__ import annotations
from collections import OrderedDict as odict
from copy import deepcopy
import glob
import matplotlib.pyplot as plt
from operator import add
import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Sequence
import torch
import numpy as np

from isaaclab.isaaclab_configuration import IsaaclabConfig
from isaaclab.isaaclab_constants import ordered_joint_names_isaaclab
from utils.dict_tools import dict_to_list
from utils.pickle_tools import get_pickle_data

from .plot_config import PlotConfig

class Plotter:
    def __init__(
        self,
        config: IsaaclabConfig,
        file_regex: str,
        plot_configs: Dict[str, PlotConfig],
        history_len: int = 1,
        estimate: bool = False,
        store_history: bool = True,
        debug: bool = False
    ) -> None:
        """Initialize the plotter.
        
        Args:
            config: Robot configuration
            file_regex: Regex pattern to match pickle files
            plot_configs: Dictionary of plot configurations
            history_len: Length of history to maintain
            estimate: Whether to plot estimates
            debug: Whether to print debug information
        """
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.family'] = 'Sans'
        
        self.config = config
        self.files = glob.glob(file_regex)
        self.data = get_pickle_data(file_regex, history_len)
        self.H = history_len
        self.store_history = store_history
        self.dt = 1. / 50.
        self.t = []
        self.counter = 0
        self.Ka = config.action_scale
        self.estimate_bool = estimate
        self.debug = debug

        # Default style configuration
        self.default_styles = {
            "colors": ["tab:blue", "tab:orange", "tab:green", "tab:red"],
            "linestyles": ["solid"], # ["-", "--", ":", "-."],
            "linewidths": [3]
        }
        
        # Initialize storage
        self.data_storage = {}
        self.figs = {}
        self.axes = {}
        self.plot_configs = plot_configs
        
        # Create figures based on configs
        self._setup_figures()

    def _setup_figures(self) -> None:
        """Create matplotlib figures based on configurations."""
        for plot_name, config in self.plot_configs.items():
            subplot_kw = config.subplot_kwargs or {"sharex": True}
            
            if config.num_subplots > 1:
                fig, axes = plt.subplots(config.num_subplots, 1, **subplot_kw) # type: ignore
                self.figs[plot_name] = fig
                self.axes[plot_name] = axes
            else:
                fig, ax = plt.subplots(1, 1, **subplot_kw) # type: ignore
                self.figs[plot_name] = fig
                self.axes[plot_name] = [ax]

    def log(self, **kwargs) -> None:
        """Log data with flexible key-value pairs."""
        # Update time
        self.t.append(self.counter * self.dt)
        self.counter += 1

        # Store all provided data
        for key, value in kwargs.items():
            if key not in self.data_storage:
                self.data_storage[key] = []
            
            # Flatten and convert to list if torch tensor
            if isinstance(value, torch.Tensor):
                value = value.view(-1).tolist()

            # Convert to list if numpy array
            if isinstance(value, np.ndarray):
                value = value.tolist()
            
            # Store the data
            if self.store_history:
                self.data_storage[key].append(value)
            else:
                self.data_storage[key] = value

            if self.debug:
                print(f"Logged {key}: {value}")

    def plot(self, path: str, idx: List[int] | None = None, custom_plots: Dict[str, List[str]] | None = None) -> None:
        """Plot data based on configurations and custom plot requests.
        
        Args:
            path: Path to save plots
            idx: Optional start and end indices to plot
            custom_plots: Dictionary of plot name to list of data keys to plot
        """
        if path[-1] != "/":
            path += "/"

        # Set plot range
        if idx is None:
            idx_start, idx_end = 0, -1
        else:
            idx_start, idx_end = idx[0], idx[1]

        # Plot based on default configs
        self._plot_default_configs(path, idx_start, idx_end)

        # Plot custom combinations if provided
        if custom_plots:
            self._plot_custom_combinations(custom_plots, path, idx_start, idx_end)

    def _plot_default_configs(self, path: str, idx_start: int = 0, idx_end: int = -1) -> None:
        """Plot using default configurations."""
        for plot_name, config in self.plot_configs.items():
            if plot_name in self.data_storage:
                self._plot_single_config(plot_name, config, path, idx_start, idx_end)

    def _plot_single_config(self, plot_name: str, config: PlotConfig, path: str, idx_start: int = 0, idx_end: int = -1) -> None:
        """Plot a single configuration."""
        t = [i - self.t[idx_start] for i in self.t[idx_start:idx_end]]
        axes = self.axes[plot_name]
        fig = self.figs[plot_name]
        data = self.data_storage[plot_name]
        if isinstance(data[0], (list, np.ndarray)):
            idx_per_plot = len(data[0]) // config.num_subplots
            subplot_groups = [
                min(n//idx_per_plot, config.num_subplots-1) 
                for n in range(len(data[0]))
            ]
        colors = config.colors if config.colors is not None else self.default_styles["colors"]
        linewidth = config.linewidths if config.linewidths is not None else self.default_styles["linewidths"]
        linestyle = config.linestyles if config.linestyles is not None else self.default_styles["linestyles"]
        labels = config.keys

        for i, ax in enumerate(axes):
            if isinstance(data[0], (list, np.ndarray)):
                for j, d in enumerate(zip(*data)):
                    if i != subplot_groups[j] and config.num_subplots > 1:
                        continue
                    c = colors[j % len(colors)]
                    style = {
                        "ls": linestyle[j % len(linestyle)],
                        "lw": linewidth[j % len(linewidth)],
                        }
                    name = labels[j] if labels is not None else f"{j}"
                    ax.plot(t, d[idx_start:idx_end], c, **style, label=f"{name}")
            else:
                ax.plot(t, data[idx_start:idx_end], colors[0], label=plot_name)
            
            if labels is not None:
                ax.legend(
                    loc="upper left", ncol=2, fontsize=20, 
                    borderpad=0.5, handlelength=1.5, handletextpad=0.5,
                    labelspacing=0.3, columnspacing=0.5
                )
            if i == 0:
                ax.set_title(config.title, fontsize=40)
            ax.set_ylabel(config.ylabel, fontsize=30)
            if i == len(axes) - 1:
                ax.set_xlabel(config.xlabel, fontsize=30)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)

        fig.savefig(f"{path}{plot_name}.png")
        plt.close(fig)

    def _plot_custom_combinations(self, custom_plots: Dict[str, List[str]], path: str, idx_start: int = 0, idx_end: int = -1) -> None:
        """Plot custom combinations of data."""
        t = [i - self.t[idx_start] for i in self.t[idx_start:idx_end]]
        for plot_name, data_keys in custom_plots.items():
            fig, ax = plt.subplots()
            
            for key in data_keys:
                if key in self.data_storage:
                    data = self.data_storage[key]
                    if isinstance(data[0], (list, np.ndarray)):
                        for j, d in enumerate(zip(*data)):
                            ax.plot(t, d[idx_start:idx_end], label=f"{key}_{j}")
                    else:
                        ax.plot(t, data[idx_start:idx_end], label=key)
            
            ax.legend(
                    loc="upper left", ncol=2, fontsize=20, 
                    borderpad=0.5, handlelength=1.5, handletextpad=0.5,
                    labelspacing=0.3, columnspacing=0.5
                )
            ax.set_title(plot_name, fontsize=40)
            ax.set_xlabel("Time [s]", fontsize=30)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)
            fig.savefig(f"{path}{plot_name}.png")
            plt.close(fig)


    def merge_data(self) -> None:
        """Load and merge data from pickle files."""
        for v in self.data.values():
            self.log(**v)

    # def _initialize_data_storage(self) -> None:
    #     """Initialize data storage dictionaries."""
    #     self.data_storage = {
    #         "velocity": {"vx": [], "vy": [], "vz": [], "avx": [], "avy": [], "avz": []},
    #         "velocity_command": {"vx": [], "vy": [], "avz": []},
    #         "projected_gravity": {"gx": [], "gy": [], "gz": []},
    #         "joint_position": odict([(k, []) for k in ordered_joint_names_isaaclab]),
    #         "joint_position_des": odict([(k, []) for k in ordered_joint_names_isaaclab]),
    #         "joint_velocity": odict([(k, []) for k in ordered_joint_names_isaaclab]),
    #         "action": odict([(k, []) for k in ordered_joint_names_isaaclab]),
    #         "acceleration": {"x": [], "y": [], "z": []}
    #     }
        
    #     if self.estimate_bool:
    #         self.data_storage.update({
    #             "grf_estimate": {leg: {"F": []} for leg in ["fl", "fr", "hl", "hr"]},
    #             "force_estimate": {k: [] for k in ["Fx", "Fy", "Fz"]}
    #         })

    # def _process_timestep(self, timestep_data: Dict) -> None:
    #     """Process and store data from a single timestep.
        
    #     Args:
    #         timestep_data: Dictionary containing observation, action and estimate data
    #     """
    #     obs = timestep_data["obs"]
    #     act = timestep_data["action"]
        
    #     # Process velocity data
    #     self.data_storage["velocity"]["vx"].append(obs[0])
    #     self.data_storage["velocity"]["vy"].append(obs[1])
    #     self.data_storage["velocity"]["vz"].append(obs[2])
    #     self.data_storage["velocity"]["avx"].append(obs[3*self.H])
    #     self.data_storage["velocity"]["avy"].append(obs[3*self.H+1])
    #     self.data_storage["velocity"]["avz"].append(obs[3*self.H+2])
        
    #     # Process projected gravity
    #     self.data_storage["projected_gravity"]["gx"].append(obs[6*self.H])
    #     self.data_storage["projected_gravity"]["gy"].append(obs[6*self.H+1])
    #     self.data_storage["projected_gravity"]["gz"].append(obs[6*self.H+2])
        
    #     # Process velocity commands
    #     self.data_storage["velocity_command"]["vx"].append(obs[9*self.H])
    #     self.data_storage["velocity_command"]["vy"].append(obs[9*self.H+1])
    #     self.data_storage["velocity_command"]["avz"].append(obs[9*self.H+2])
        
    #     # Process joint positions and desired positions
    #     default_joints = dict_to_list(self.config.default_joints, self.config.ordered_joint_names)
    #     q = list(map(add, obs[12*self.H:12*self.H+12], default_joints))
        
    #     for idx, joint_name in enumerate(self.data_storage["joint_position"].keys()):
    #         # Current position
    #         self.data_storage["joint_position"][joint_name].append(q[idx])
    #         # Desired position (current + scaled action)
    #         self.data_storage["joint_position_des"][joint_name].append(q[idx] + self.Ka*act[idx])
    #         # Joint velocity
    #         self.data_storage["joint_velocity"][joint_name].append(obs[24*self.H+idx])
    #         # Action
    #         self.data_storage["action"][joint_name].append(act[idx])
        
    #     # Process estimates if enabled
    #     if self.estimate_bool and "estimate" in timestep_data:
    #         est = timestep_data["estimate"]
            
    #         # Process force estimates
    #         for idx, key in enumerate(["Fx", "Fy", "Fz"]):
    #             self.data_storage["force_estimate"][key].append(est[idx])
            
    #         # Process GRF estimates for each leg
    #         for idx, leg in enumerate(["fl", "fr", "hl", "hr"]):
    #             self.data_storage["grf_estimate"][leg]["F"].append(est[3+idx])
            
    #         if self.debug:
    #             print(f"Processed estimates - Force: {est[0:3]}, GRF: {est[3:7]}")
        
    #     if self.debug:
    #         print(f"Processed timestep data - Shape: obs={len(obs)}, act={len(act)}")

    @staticmethod
    def running_average(data: List[float], steps: int = 5, clip: bool = False) -> Sequence[float]:
        """Compute running average of data.
        
        Args:
            data: List of values to average
            steps: Window size for averaging
            clip: Whether to clip negative values to zero
        
        Returns:
            Smoothed data list
        """
        data_padded = [data[0]]*(steps-1) + data
        data_smooth = []
        for i in range(len(data)):
            data_smooth.append(round(sum(data_padded[i:i+steps])/steps))
        if clip:
            data_smooth = [max(d,0) for d in data_smooth]
        return data_smooth
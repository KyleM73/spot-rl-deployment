# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass
from datetime import datetime
from operator import add, mul
from pathlib import Path
import pickle
from threading import Event
from typing import List

import numpy as np
import onnxruntime as ort
import isaaclab.observations as ob
from bosdyn.api import robot_command_pb2
from bosdyn.api.robot_command_pb2 import JointControlStreamRequest
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse
from bosdyn.util import seconds_to_timestamp, set_timestamp_from_now, timestamp_to_sec
from isaaclab.isaaclab_configuration import IsaaclabConfig
from isaaclab.isaaclab_constants import ordered_joint_names_isaaclab
from spot.constants import DEFAULT_K_Q_P, DEFAULT_K_QD_P, ordered_joint_names_bosdyn
from utils.dict_tools import dict_to_list, find_ordering, reorder


class TestController:
    def __init__(self, duration: float, f: float = 50) -> None:
        self.steps = int(duration * f)
        self.fwd = [1, 0, 0]
        self.bkd = [-1, 0, 0]
        self.left = [0, 1, 0]
        self.right = [0, -1, 0]
        self.ccw = [0, 0, 1]
        self.cw = [0, 0, -1]
        self.stance = [0, 0, 0]
        self.gaits = [
            self.stance,
            self.fwd, self.bkd,
            self.left, self.right,
            self.ccw, self.cw,
            self.stance,
        ]
        self.cmds = []
        for gait in self.gaits:
            for i in range(self.steps):
                self.cmds.append(gait)
        self.count = 0

    def __call__(self):
        if self.count < len(self.cmds):
            self.count += 1
            return self.cmds[self.count-1]
        return self.stance


@dataclass
class OnnxControllerContext:
    """data class to hold runtime data needed by the controller"""

    event = Event()
    latest_state = None
    velocity_cmd = [0, 0, 0]
    count = 0


class StateHandler:
    """Class to be used as callback for state stream to put state date
    into the controllers context
    """

    def __init__(self, context: OnnxControllerContext) -> None:
        self._context = context

    def __call__(self, state: RobotStateStreamResponse):
        """make class a callable and handle incoming state stream when called

        arguments
        state -- proto msg from spot containing most recent data on the robots state"""
        self._context.latest_state = state
        self._context.event.set()


def print_observations(observations: List[float]):
    """debug function to print out the observation data used as model input

    arguments
    observations -- list of float values ready to be passed into the model
    """
    print("base_linear_velocity:", observations[0:3])
    print("base_angular_velocity:", observations[3:6])
    print("projected_gravity:", observations[6:9])
    print("joint_positions", observations[12:24])
    print("joint_velocity", observations[24:36])
    print("last_action", observations[36:48])


class OnnxCommandGenerator:
    """class to be used as generator for spots command stream that executes
    an onnx model and converts the output to a spot command"""

    def __init__(
        self,
        context: OnnxControllerContext,
        config: IsaaclabConfig,
        policy_file_name: os.PathLike,
        verbose: bool,
        record: str = None,
        history: int = 1,
        test: bool = False,
        estimate: bool = False,
    ):
        self._context = context
        self._config = config
        self._inference_session = ort.InferenceSession(policy_file_name)
        self._last_action = [0] * 12
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose
        self.record_path = record
        if self.record_path is not None:
            self.record = True
            self.recording_dict = {}
            Path(f"{self.record_path}/logs").mkdir(parents=True, exist_ok=True)
            self.file = f"{self.record_path}/logs/spot_{datetime.now().strftime('%m-%d-%Y_%H%M')}"
        else:
            self.record = False
        self.H = history
        if self.H > 1:
            self.lin_vel = [0] * 3 * self.H
            self.ang_vel = [0] * 3 * self.H
            self.vel_cmd = [0] * 3 * self.H
            self.proj_g = [0] * 3 * self.H
            self.q = [0] * 12 * self.H
            self.dq = [0] * 12 * self.H
            self.last_a = [0] * 12 * self.H
        self.test = test
        if self.test:
            self.test_controller = TestController(5, 50)
        self.estimate_bool = estimate

    def __call__(self):
        """makes class a callable and computes model output for latest controller context

        return proto message to be used in spots command stream
        """

        # cache initial joint position when command stream starts
        if self._init_pos is None:
            self._init_pos = self._context.latest_state.joint_states.position
            self._init_load = self._context.latest_state.joint_states.load

        # extract observation data from latest spot state data
        input_list = self.collect_inputs(self._context.latest_state, self._config)
        # print("observations", input_list)

        # execute model from onnx file
        if self.H > 1:
            input_list = self.collect_history(input_list)
        input = [np.array(input_list).astype("float32")]
        outputs = self._inference_session.run(None, {"obs": input})
        output = outputs[0].tolist()[0]
        if self.estimate_bool:
            estimate = outputs[1].tolist()[0]

        if self.record:
            self.recording_dict[self._count] = {"obs": input[0].tolist(), "action": output}
            if self.estimate_bool:
                self.recording_dict[self._count]["estimate"] = estimate
            # 500 steps @ ~50hz --> 10s
            if (self._count - 1) % 500 == 0:
                with open(f"{self.file}_{self._count // 500}.pkl", "wb") as f:
                    pickle.dump(self.recording_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.recording_dict = {}

        # post process model output apply action scaling and return to spots
        # joint order and offset
        test_scale = min(0.1 * self._count, 1)

        scaled_output = list(map(mul, [self._config.action_scale] * 12, output))
        test_scaled = list(map(mul, [test_scale] * 12, scaled_output))

        default_joints = dict_to_list(self._config.default_joints, ordered_joint_names_isaaclab)
        shifted_output = list(map(add, test_scaled, default_joints))

        isaaclab_to_spot = find_ordering(ordered_joint_names_isaaclab, ordered_joint_names_bosdyn)
        reordered_output = reorder(shifted_output, isaaclab_to_spot)

        # generate proto message from target joint positions
        proto = self.create_proto(reordered_output)

        # cache data for history and logging
        self._last_action = output
        self._count += 1
        self._context.count += 1

        return proto

    def collect_history(self, observation: List[float]) -> List[float]:
        self.lin_vel = observation[0:3] + self.lin_vel[:-3]
        self.ang_vel = observation[3:6] + self.ang_vel[:-3]
        self.proj_g = observation[6:9] + self.proj_g[:-3]
        self.vel_cmd = observation[9:12] + self.vel_cmd[:-3]
        self.q = observation[12:24] + self.q[:-12]
        self.dq = observation[24:36] + self.dq[:-12]
        self.last_a = observation[36:48] + self.last_a[:-12]
        return self.lin_vel + self.ang_vel + self.proj_g + self.vel_cmd + self.q + self.dq + self.last_a

    def collect_inputs(self, state: JointControlStreamRequest, config: IsaaclabConfig):
        """extract observation data from spots current state and format for onnx

        arguments
        state -- proto msg with spots latest state
        config -- model configuration data from isaaclab

        return list of float values ready to be passed into the model
        """
        observations = []
        observations += ob.get_base_linear_velocity(state)
        observations += ob.get_base_angular_velocity(state)
        observations += ob.get_projected_gravity(state)
        if self.test:
            vel_cmd = self.test_controller()
        else:
            vel_cmd = self._context.velocity_cmd
        observations += vel_cmd
        if self.verbose:
            print("[INFO] cmd", vel_cmd)
        observations += ob.get_joint_positions(state, config)
        observations += ob.get_joint_velocity(state)
        observations += self._last_action
        return observations

    def create_proto(self, pos_command: List[float]):
        """generate a proto msg for spot with a given pos_command

        arguments
        pos_command -- list of joint positions see spot.constants for order

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = dict_to_list(self._config.kp, ordered_joint_names_bosdyn)
        k_qd_p = dict_to_list(self._config.kd, ordered_joint_names_bosdyn)

        N_DOF = len(pos_command)
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = pos_command[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = 0

        # Fill in gains the first dt
        if self._count == 1:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto

    def create_proto_hold(self):
        """generate a proto msg that holds spots current pose useful for debugging

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        update_proto.Clear()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = DEFAULT_K_Q_P[0:12]
        k_qd_p = DEFAULT_K_QD_P[0:12]

        N_DOF = 12
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = self._init_pos[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = self._init_load[joint_ind]

        # Fill in gains the first dt
        if self._count == 1:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto

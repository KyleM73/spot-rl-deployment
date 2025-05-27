# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from operator import sub

from bosdyn.api import robot_state_pb2
from isaaclab.isaaclab_configuration import IsaaclabConfig
from isaaclab.isaaclab_constants import ordered_joint_names_isaaclab
from spatialmath import UnitQuaternion
from spot.constants import ordered_joint_names_bosdyn
from utils.dict_tools import dict_to_list, find_ordering, reorder


def get_base_linear_velocity(state: robot_state_pb2.RobotStateStreamResponse):
    """calculate linear velocity of spots base in the base frame from data
    available in spots state update.  note spot gives velocity in odom frame
    so we need to rotate it to current estimated pose of the base

    arguments
    state -- proto msg from spot containing data on the robots state
    """
    msg = state.kinematic_state.velocity_of_body_in_odom.linear

    odom_r_base_msg = state.kinematic_state.odom_tform_body.rotation
    scalar = odom_r_base_msg.w
    vector = [odom_r_base_msg.x, odom_r_base_msg.y, odom_r_base_msg.z]
    odom_r_base = UnitQuaternion(scalar, vector)

    velocity_odom = [msg.x, msg.y, msg.z]
    velocity_base = odom_r_base.inv() * velocity_odom

    return velocity_base.tolist()


def get_base_angular_velocity(state: robot_state_pb2.RobotStateStreamResponse):
    """calculate angular velocity of spots base in the base frame from data
    available in spots state update.  note spot gives velocity in odom frame
    so we need to rotate it to current estimated pose of the base

    arguments
    state -- proto msg from spot containing data on the robots state
    """
    msg = state.kinematic_state.velocity_of_body_in_odom.angular

    odom_r_base_msg = state.kinematic_state.odom_tform_body.rotation
    scalar = odom_r_base_msg.w
    vector = [odom_r_base_msg.x, odom_r_base_msg.y, odom_r_base_msg.z]
    odom_r_base = UnitQuaternion(scalar, vector)

    angular_velocity_odom = [msg.x, msg.y, msg.z]
    angular_velocity_base = odom_r_base.inv() * angular_velocity_odom

    return angular_velocity_base.tolist()


def get_base_linear_acceleration(state: robot_state_pb2.RobotStateStreamResponse):
     """calculate linear acceleration of spots base in the base frame from data
    available in spots state update.  note spot gives acceleration in link frame
    so we need to rotate it to current estimated pose of the base through the odom frame

    arguments
    state -- proto msg from spot containing data on the robots state
    """
    if not state.inertial_state.packets:
        raise ValueError("No acceleration packets found in inertial state")
    imu_packet = state.inertial_state.packets[-1]
    acceleration_link_msg = imu_packet.acceleration_rt_odom_in_link_frame
    acceleration_link = [
        acceleration_link_msg.x,
        acceleration_link_msg.y,
        acceleration_link_msg.z
    ]

    odom_r_link_msg = imu_packet.odom_rot_link
    odom_r_link = UnitQuaternion(
        odom_r_link_msg.w,
        [odom_r_link_msg.x, odom_r_link_msg.y, odom_r_link_msg.z]
    )

    acceleration_odom = odom_r_link * acceleration_link
    gravity = [0, 0, -9.81]
    acceleration_odom += gravity

    odom_r_base_msg = state.kinematic_state.odom_tform_body.rotation
    odom_r_base = UnitQuaternion(
        odom_r_base_msg.w,
        [odom_r_base_msg.x, odom_r_base_msg.y, odom_r_base_msg.z]
    )

    acceleration_base = odom_r_base.inv() * acceleration_odom

    # print(f"Link: {acceleration_link}")
    # print(f"Odom: {acceleration_odom}")
    # print(f"{acceleration_base}")
    # print()

    return acceleration_base.tolist()

def OLD_get_base_linear_acceleration(state: robot_state_pb2.RobotStateStreamResponse):
    """calculate linear acceleration of spots base in the base frame from data
    available in spots state update.  note spot gives acceleration in link frame
    so we need to rotate it to current estimated pose of the base through the odom frame

    arguments
    state -- proto msg from spot containing data on the robots state
    """
    # Extract acceleration in the link frame
    if not state.inertial_state.packets:
        raise ValueError("No acceleration packets found in inertial state")
    acc_msg = state.inertial_state.packets[-1].acceleration_rt_odom_in_link_frame
    acceleration_link = [acc_msg.x, acc_msg.y, acc_msg.z]

    # Get rotation from mounting link to odom frame
    odom_r_link_msg = state.inertial_state.packets[-1].odom_rot_link
    odom_r_link = UnitQuaternion(odom_r_link_msg.w, 
                                 [odom_r_link_msg.x, 
                                  odom_r_link_msg.y, 
                                  odom_r_link_msg.z])

    # Rotate acceleration from link frame to odom frame
    acceleration_odom = odom_r_link * acceleration_link

    # Get rotation from odom to base frame
    odom_r_base_msg = state.kinematic_state.odom_tform_body.rotation
    odom_r_base = UnitQuaternion(odom_r_base_msg.w, 
                                 [odom_r_base_msg.x, 
                                  odom_r_base_msg.y, 
                                  odom_r_base_msg.z])

    # Rotate acceleration from odom frame to base frame
    acceleration_base = odom_r_base.inv() * acceleration_odom
    print(f"Acceleration: {acceleration_base}")
    # acceleration_base *= 0

    # account for gravity
    gravity_odom = [0, 0, -9.81]
    gravity_base = odom_r_base.inv() * gravity_odom

    acceleration_net = acceleration_base + gravity_base

    return acceleration_net.tolist()


def get_projected_gravity(state: robot_state_pb2.RobotStateStreamResponse):
    """calculate direction of gravity in spots base frame
        the assumption here is that the odom frame Z axis is opposite gravity
        this is the case if spots body is parallel to the floor when turned on

    arguments
    state -- proto msg from spot containing data on the robots state
    """
    odom_r_base_msg = state.kinematic_state.odom_tform_body.rotation

    scalar = odom_r_base_msg.w
    vector = [odom_r_base_msg.x, odom_r_base_msg.y, odom_r_base_msg.z]
    odom_r_base = UnitQuaternion(scalar, vector)

    gravity_odom = [0, 0, -1]
    gravity_base = odom_r_base.inv() * gravity_odom
    return gravity_base.tolist()


def get_joint_positions(state: robot_state_pb2.RobotStateStreamResponse, config: IsaaclabConfig):
    """get joint position from spots state update a reformat for isaaclab by
    reordering to match isaaclabs expectation and shifting so 0 position is the
    same as was used in training

    arguments
    state -- proto msg from spot containing data on the robots state
    config -- dataclass with values loaded from isaaclabs training data
    """

    spot_to_isaaclab = find_ordering(ordered_joint_names_bosdyn, ordered_joint_names_isaaclab)
    pos = reorder(state.joint_states.position, spot_to_isaaclab)
    default_joints = dict_to_list(config.default_joints, ordered_joint_names_isaaclab)
    pos = list(map(sub, pos, default_joints))
    return pos


def get_joint_velocity(state: robot_state_pb2.RobotStateStreamResponse):
    """get joint velocity from spots state update a reformat for isaaclab by
    reordering to match isaaclabs expectation

    arguments
    state -- proto msg from spot containing data on the robots state
    """
    spot_to_isaaclab = find_ordering(ordered_joint_names_bosdyn, ordered_joint_names_isaaclab)
    vel = reorder(state.joint_states.velocity, spot_to_isaaclab)
    return vel

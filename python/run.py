# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path

import bosdyn.client.util
import isaaclab.isaaclab_configuration
from hid.gamepad import (
    Gamepad,
    GamepadConfig,
    joystick_connected,
    load_gamepad_configuration,
)
from isaaclab.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from spot.mock_spot import MockSpot
from spot.spot import Spot
from utils.event_divider import EventDivider


def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("-m", "--mock", action="store_true")
    parser.add_argument("--gamepad-config", type=Path)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--estimate", action="store_true", default=False)
    options = parser.parse_args()

    conf_file = isaaclab.isaaclab_configuration.detect_config_file(options.policy_file_path)
    policy_file = isaaclab.isaaclab_configuration.detect_policy_file(options.policy_file_path)

    context = OnnxControllerContext()
    config = isaaclab.isaaclab_configuration.load_configuration(conf_file)
    print(config)

    state_handler = StateHandler(context)
    print(options.verbose)
    if options.record:
        record_path = options.policy_file_path
        print(f"[INFO] saving logs to {record_path}/logs/")
    else:
        record_path = None
    command_generator = OnnxCommandGenerator(
        context, config, policy_file,
        options.verbose, record_path,
        options.history, options.test,
        options.estimate
    )

    # 333 Hz state update / 6 => ~56 Hz control updates
    timeing_policy = EventDivider(context.event, 6)

    gamepad = None
    if joystick_connected():
        if options.gamepad_config is not None:
            print("[INFO] loading gamepad config from file")
            gamepad_config = load_gamepad_configuration(options.gamepad_config)
        else:
            print("[INFO] using default gamepad configuration")
            gamepad_config = GamepadConfig()

        gamepad = Gamepad(context, gamepad_config)
        gamepad.start_listening()

    if options.mock:
        spot = MockSpot()
    else:
        spot = Spot(options)

    with spot.lease_keep_alive():
        try:
            spot.power_on()
            spot.stand(0.0)
            spot.start_state_stream(state_handler)

            input()
            spot.start_command_stream(command_generator, timeing_policy)
            input()

        except KeyboardInterrupt:
            print("killed with ctrl-c")

        finally:
            print("stop command stream")
            spot.stop_command_stream()
            print("stop state stream")
            spot.stop_state_stream()
            print("stop game pad")
            if gamepad is not None:
                gamepad.stop_listening()
            print("all stopped")


if __name__ == "__main__":
    if not main():
        sys.exit(1)

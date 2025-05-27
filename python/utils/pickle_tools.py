import glob
import pickle
from typing import Any, Dict, Sequence
from collections import OrderedDict as odict

def get_pickle_data(
    regex: str,
    history_len: int = 1,
) -> Dict:
    files = glob.glob(regex)
    objects = {}
    for file in files:
        with open(file, "rb") as f:
            try:
                while True:
                    objects = {**objects, **pickle.load(f)}
            except EOFError:
                pass
    objects = odict(sorted(objects.items()))
    logs = {}

    for k, v in objects.items():
        obs = v["obs"]
        action = v["action"]
        estimate = v["estimate"]

        log = {}

        # observations
        log["base_lin_vel"] = obs[0:3]
        offset = 3 * history_len
        log["base_ang_vel"] = obs[offset:offset+3]
        offset = 6 * history_len
        log["projected_gravity"] = obs[offset:offset+3]
        offset = 9 * history_len
        log["velocity_commands"] = obs[offset:offset+3]
        offset = 12 * history_len
        log["joint_pos"] = obs[offset:offset+12]
        offset = 24 * history_len
        log["joint_vel"] = obs[offset:offset+12]
        offset = 36 * history_len
        log["last_actions"] = obs[offset:offset+12]
        if len(obs) > 48 * history_len:
            offset = 48 * history_len
            log["body_lin_acc"] = obs[offset:offset+3]
        
        # actions
        log["actions"] = action

        # estimates
        log["net_wrench"] = estimate[0:6]
        log["ground_reaction_forces"] = estimate[6:10]
        log["force_bool"] = estimate[10]
        log["force_norm"] = estimate[11]
        log["force_vec"] = estimate[12:15]
        
        logs[k] = log
    
    return logs

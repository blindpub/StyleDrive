# import json

# style_path_root = '/data_storage/jingbowen/projects/friday_project/results/scene_attribute.json'
# with open(style_path_root, 'r', encoding='utf-8') as f:
#     style_dict = json.load(f)

# style_data = {}

# for key, value in style_dict.items():
#     try:
#         style_data[key] = value["ANC_result"]
#     except KeyError:
#         print(f"[Warning] Missing 'ANC_result' in token: {key}")


import pickle
import numpy as np
import numpy.typing as npt

from typing import List

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.metric_caching.metric_cache import MetricCache

def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    return InterpolatedTrajectory([initial_ego_state] + agent_states)

pickle_path = "/data_storage/haoruiyang/projects/mme2e/StyleDrive/DiffusionDrive/exp/eval_transfuser_agent/2025.05.15.13.03.57/traj_and_metric/ffd68c5733d35ebc.pkl"

# 读取 Pickle 文件
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

data2_path = "/data_storage/haoruiyang/projects/mme2e/navsim/frame_tokens_with_dict_navtest.pkl"
with open(data2_path, 'rb') as f:
    data2 = pickle.load(f)

human_data = data2['ffd68c5733d35ebc']
metric_cache = data['metric_cache']
model_trajectory = data['pred_traj']

initial_ego_state = metric_cache.ego_state

pdm_trajectory = metric_cache.trajectory
pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

print(pdm_states)
print(pred_states)
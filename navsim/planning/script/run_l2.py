from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import uuid
import os
import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig
from navsim.planning.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_eval_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    eval_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(scene_loader.tokens):
        logger.info(
            f"Processing scenario {idx + 1} / {len(scene_loader.tokens)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:

            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            
            gt_trajectory = scene.get_future_trajectory()

            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                pred_trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                pred_trajectory = agent.compute_trajectory(agent_input, token)

            L2_2 = np.linalg.norm(pred_trajectory.poses[3, :2] - gt_trajectory.poses[3, :2])
            L2_3 = np.linalg.norm(pred_trajectory.poses[5, :2] - gt_trajectory.poses[5, :2])
            L2_4 = np.linalg.norm(pred_trajectory.poses[7, :2] - gt_trajectory.poses[7, :2])

            eval_result = {
                "l2_2": round(float(L2_2), 3),
                "l2_3": round(float(L2_3), 3),
                "l2_4": round(float(L2_4), 3),
                "l2_avg": round(float(np.mean([L2_2, L2_3, L2_4])), 3),
            }
            
            score_row.update(eval_result)
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        eval_results.append(score_row)
    return eval_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    
    logger.info("Starting scoring of %s scenarios...", str(len(cfg.train_test_split.scene_filter.tokens)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_eval_score, data_points)

    # score_rows: List[Dict[str, Any]] = run_eval_score(data_points)

    eval_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = eval_score_df["valid"].sum()
    num_failed_scenarios = len(eval_score_df) - num_sucessful_scenarios
    average_row = eval_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = eval_score_df["valid"].all()
    eval_score_df.loc[len(eval_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    eval_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average l2 of valid results: {eval_score_df['l2_avg'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()

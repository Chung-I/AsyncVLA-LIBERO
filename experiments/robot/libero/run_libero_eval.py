"""
run_libero_eval.py

Stock OpenVLA-OFT LIBERO-Spatial evaluation harness, ported from
`moojink/openvla-oft`'s `experiments/robot/libero/run_libero_eval.py`,
trimmed to a single `--mode stock` path.

`--mode stock` loads the frozen base exactly as Task 2.1 verified
(`experiments.robot.libero.base_config.build_frozen_base`), attaches the
checkpoint's native L1-regression action head + proprio projector, and steps
LIBERO-Spatial rollouts using the base's native parallel-decoded 8-step action
chunk (open-loop, `num_open_loop_steps == NUM_ACTIONS_CHUNK == 8`), exactly as
the reference harness does. This is the project's gate: proving the frozen
base reproduces stock LIBERO behavior before any edge is attached.

Only `--mode stock` is implemented in this task (Task 3.1). `--num_tasks` is
an addition (not in the reference) to allow limiting how many of the task
suite's tasks are run, for smoke testing.
"""

import argparse
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
from libero.libero import benchmark

# --- torch>=2.6 compatibility shim ---
# `benchmark.get_task_init_states` calls plain `torch.load(init_states_path)` on
# LIBERO's own bundled `.pruned_init` files (numpy object arrays pickled with an
# older torch/numpy). Since PyTorch 2.6, `torch.load`'s default `weights_only`
# flipped True->False is now True, which rejects the `numpy.core.multiarray._reconstruct`
# global these files use and raises `UnpicklingError`. This is a real integration
# point surfaced during Task 3.1's smoke run, not a stock-eval code path we can
# change (LIBERO is an installed, unmodified dependency). These are local,
# trusted files bundled with the LIBERO install, so restoring the old default is
# safe here.
_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    OPENVLA_IMAGE_SIZE,
    get_action_head,
    get_vla_action,
    resize_image_for_policy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Longest training demo per task suite (same table as the reference harness).
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class StockEvalConfig(LiberoBaseConfig):
    """Extends Task 2.1's `LiberoBaseConfig` (model-loading fields) with the
    additional fields `get_vla_action`/`get_action_head` (in openvla_utils.py)
    read off `cfg`."""

    center_crop: bool = True
    use_l1_regression: bool = True
    use_diffusion: bool = False
    unnorm_key: str = ""
    model_family: str = "openvla"


def set_seed_everywhere(seed: int) -> None:
    """Set random seed for all random number generators for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Normalize gripper action from [0,1] to [-1,+1] range (binarized)."""
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip the sign of the gripper action (last dim), since the RLDS dataloader
    aligns gripper actions to (0=close, 1=open), but the LIBERO env expects
    (-1=open, +1=close)."""
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action


def process_action(action: np.ndarray) -> np.ndarray:
    """Normalize + invert the gripper action before sending it to the LIBERO env."""
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def prepare_observation(obs, resize_size):
    """Prepare a LIBERO env observation for the VLA policy: extract + resize the
    two camera images, and build the 8-dim proprio state vector."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def run_episode(
    cfg: StockEvalConfig,
    env,
    task_description: str,
    vla,
    processor,
    action_head,
    proprio_projector,
    resize_size,
    initial_state,
    num_steps_wait: int,
    max_steps: int,
):
    """Run a single open-loop episode: requery the policy every 8 steps
    (NUM_ACTIONS_CHUNK), executing its native parallel-decoded action chunk."""
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = deque()
    t = 0
    replay_images = []
    success = False
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_vla_action(
                    cfg=cfg,
                    vla=vla,
                    processor=processor,
                    obs=observation,
                    task_label=task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        logger.exception(f"Episode error: {e}")

    return success, replay_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock OpenVLA-OFT LIBERO-Spatial eval harness.")
    parser.add_argument("--mode", type=str, default="stock", choices=["stock"],
                        help="Eval mode. Only 'stock' (native base action prediction) is implemented.")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                         choices=list(TASK_MAX_STEPS.keys()), help="LIBERO task suite name.")
    parser.add_argument("--num_trials_per_task", type=int, default=50, help="Number of rollouts per task.")
    parser.add_argument("--num_tasks", type=int, default=None,
                         help="Limit the number of tasks run from the suite (default: all tasks). Added for smoke testing.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                         help="Number of steps to wait for objects to stabilize in sim before querying the policy.")
    parser.add_argument("--pretrained_checkpoint", type=str,
                         default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    parser.add_argument("--env_img_res", type=int, default=256, help="LIBERO env camera resolution.")
    parser.add_argument("--no_center_crop", action="store_true", help="Disable center-crop image preprocessing.")
    parser.add_argument("--no_save_video", action="store_true", help="Skip saving MP4 rollout videos.")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.mode == "stock", f"Only --mode stock is implemented (Task 3.1); got {args.mode!r}."

    set_seed_everywhere(args.seed)

    # --- Load frozen base (Task 2.1 interface) + native action head ---
    base_cfg = StockEvalConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        lora_rank=0,
        center_crop=not args.no_center_crop,
        use_l1_regression=True,
        use_diffusion=False,
    )
    vla, processor, proprio_projector = build_frozen_base(base_cfg)
    action_head = get_action_head(base_cfg, vla.llm_dim)
    action_head.requires_grad_(False)
    action_head.eval()

    # Resolve the unnorm_key (task suite name, with "_no_noops" fallback).
    unnorm_key = args.task_suite
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, (
        f"Action un-norm key {unnorm_key!r} not found in VLA norm_stats! Available: {list(vla.norm_stats.keys())}"
    )
    base_cfg.unnorm_key = unnorm_key
    logger.info(f"Resolved unnorm_key={unnorm_key!r}")

    resize_size = OPENVLA_IMAGE_SIZE

    # --- Initialize LIBERO task suite ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks if args.num_tasks is None else min(args.num_tasks, task_suite.n_tasks)
    max_steps = TASK_MAX_STEPS[args.task_suite]

    logger.info(f"Task suite: {args.task_suite} | running {num_tasks}/{task_suite.n_tasks} tasks, "
                f"{args.num_trials_per_task} trial(s)/task")

    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=args.env_img_res)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task {task_id}: {task_description}"):
            logger.info(f"Task: {task_description}")
            initial_state = initial_states[episode_idx]

            success, replay_images = run_episode(
                base_cfg, env, task_description, vla, processor, action_head, proprio_projector,
                resize_size, initial_state, args.num_steps_wait, max_steps,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            if not args.no_save_video:
                save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description)

            logger.info(f"episode={total_episodes} task_id={task_id} success={success}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        logger.info(f"Task {task_id} ({task_description}) success rate: {task_sr:.4f}")

    final_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    logger.info("Final results:")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Total successes: {total_successes}")
    logger.info(f"Overall success rate: {final_sr:.4f} ({final_sr * 100:.1f}%)")

    return final_sr


if __name__ == "__main__":
    main()

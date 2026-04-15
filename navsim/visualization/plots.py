import io
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene
from navsim.visualization.bev import add_configured_bev_on_ax, add_pose_trajectory_to_bev_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_annotations_to_camera_ax, add_camera_ax, add_lidar_to_camera_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, CAMERAS_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def _extract_target_trajectories(scene: Scene, frame_idx: int) -> Dict[str, np.ndarray]:
    """
    Extract per-target future trajectories in the current ego frame.
    Returns a mapping from track token to an array of poses (x, y, heading).
    Only dynamic foreground targets that are easy to read in the plot are kept.
    """

    current_frame = scene.frames[frame_idx]
    current_ego_pose = StateSE2(*current_frame.ego_status.ego_pose)
    allowed_target_names = {"vehicle", "pedestrian"}

    target_trajectories: Dict[str, List[np.ndarray]] = {
        token: []
        for token, name in zip(current_frame.annotations.track_tokens, current_frame.annotations.names)
        if name in allowed_target_names
    }

    for future_frame in scene.frames[frame_idx:]:
        future_ego_pose = StateSE2(*future_frame.ego_status.ego_pose)
        future_annotation_indices = {
            token: idx
            for idx, (token, name) in enumerate(
                zip(future_frame.annotations.track_tokens, future_frame.annotations.names)
            )
            if name in allowed_target_names
        }

        for token in list(target_trajectories.keys()):
            if token not in future_annotation_indices:
                continue

            annotation_idx = future_annotation_indices[token]
            target_pose_local = future_frame.annotations.boxes[annotation_idx, :3]
            target_pose_global = relative_to_absolute_poses(
                future_ego_pose,
                [StateSE2(*target_pose_local)],
            )[0]
            target_pose_relative = convert_absolute_to_relative_se2_array(
                current_ego_pose,
                np.array([[target_pose_global.x, target_pose_global.y, target_pose_global.heading]], dtype=np.float64),
            )[0]
            target_trajectories[token].append(target_pose_relative)

    return {
        token: np.asarray(trajectory, dtype=np.float32)
        for token, trajectory in target_trajectories.items()
        if len(trajectory) > 1
    }


def _rank_target_tokens_by_ego_trajectory_distance(
    target_trajectories: Dict[str, np.ndarray],
    ego_trajectories: List[np.ndarray],
) -> List[str]:
    """
    Rank target trajectories by the minimum distance between any target point and any ego trajectory point.
    Lower rank means closer to the ego trajectory visualization.
    """

    ego_points = np.concatenate([trajectory[:, :2] for trajectory in ego_trajectories], axis=0)
    ranked_tokens_with_distance: List[Tuple[str, float]] = []

    for token, poses in target_trajectories.items():
        target_points = poses[:, :2]
        distances = np.linalg.norm(target_points[:, None, :] - ego_points[None, :, :], axis=-1)
        ranked_tokens_with_distance.append((token, float(np.min(distances))))

    ranked_tokens_with_distance.sort(key=lambda item: item[1])
    return [token for token, _ in ranked_tokens_with_distance]


def plot_bev_with_agent_and_target_trajectories(
    scene: Scene,
    agent: AbstractAgent,
    max_targets: Optional[int] = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ego/human trajectories together with future trajectories of current annotated targets.
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :param max_targets: optionally limit the number of plotted targets to reduce clutter
    :return: figure and ax object of matplotlib
    """

    frame_idx = scene.scene_metadata.num_history_frames - 1
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    fig, ax = plot_bev_with_agent(scene, agent)
    target_trajectories = _extract_target_trajectories(scene, frame_idx)
    ranked_tokens = _rank_target_tokens_by_ego_trajectory_distance(
        target_trajectories,
        [human_trajectory.poses, agent_trajectory.poses],
    )

    for idx, token in enumerate(ranked_tokens):
        if max_targets is not None and idx >= max_targets:
            break
        poses = target_trajectories[token]
        add_pose_trajectory_to_bev_ax(ax, poses, TRAJECTORY_CONFIG["target"])

    return fig, ax


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)


def concat_scenes_to_gif_with_labels(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scenes: List[Scene],
    frame_indices_list: List[List[int]],
    scene_labels: List[str],
    duration: float = 500,
):
    images: List[Image.Image] = []

    for scene, frame_indices, label in zip(scenes, frame_indices_list, scene_labels):
        for frame_idx in tqdm(frame_indices, desc=f"Rendering {label}"):
            fig, ax = callable_frame_plot(scene, frame_idx)

            # 🔵 Add label to the figure
            fig.text(
                0.1,
                0.95,
                label,
                fontsize=12,
                color="black",
                weight="bold",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6),
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            images.append(Image.open(buf).copy())

            buf.close()
            plt.close(fig)

    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)

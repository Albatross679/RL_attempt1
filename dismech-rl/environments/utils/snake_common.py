"""
Common utilities for snake predator environments.

Contains:
- Cylinder dataclass for target representation
- Observation NamedTuples for approach and coil tasks
- Distance/wrap angle computation utilities
- Visualizer for snake environments
"""
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Union

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import alf

# Type alias for observation spec compatibility
array = Union[np.ndarray, alf.TensorSpec]


@dataclass
class Cylinder:
    """
    Represents a cylindrical target for the snake to approach/coil around.

    Attributes:
        center: (3,) base center position on ground (z=0 for base)
        radius: cylinder radius in meters
        height: cylinder height in meters
    """
    center: np.ndarray
    radius: float
    height: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)
        assert self.center.shape == (3,), f"Center must be (3,), got {self.center.shape}"


class ApproachObservation(NamedTuple):
    """
    Observation for the snake approach task.

    Attributes:
        state_pos: (n_state_points * 3,) node positions
        state_vel: (n_state_points * 3,) node velocities
        curr_kappa_bar: (n_ctrl_points * 2,) current curvature state
        cylinder_rel_pos: (3,) cylinder center relative to snake head
        head_direction: (3,) forward unit vector of snake head
    """
    state_pos: array
    state_vel: array
    curr_kappa_bar: array
    cylinder_rel_pos: array
    head_direction: array


class CoilObservation(NamedTuple):
    """
    Observation for the snake coil task (extends approach).

    Additional attributes:
        node_distances: (n_state_points,) distance from each node to cylinder surface
        wrap_angle: (1,) cumulative wrap angle around cylinder
    """
    state_pos: array
    state_vel: array
    curr_kappa_bar: array
    cylinder_rel_pos: array
    head_direction: array
    node_distances: array
    wrap_angle: array


def compute_node_cylinder_distances(positions: np.ndarray, cylinder: Cylinder) -> np.ndarray:
    """
    Compute distance from each node to the cylinder surface.

    Args:
        positions: (N, 3) node positions
        cylinder: Cylinder target

    Returns:
        (N,) array of signed distances (negative = penetration/inside)
    """
    # Project to XY plane (cylinder axis is Z)
    xy_pos = positions[:, :2]
    xy_center = cylinder.center[:2]

    # Radial distance from cylinder axis
    radial_dist = np.linalg.norm(xy_pos - xy_center, axis=1)

    # Signed distance to surface (negative = inside cylinder)
    surface_dist = radial_dist - cylinder.radius

    return surface_dist.astype(np.float32)


def compute_wrap_angle(positions: np.ndarray, cylinder: Cylinder) -> float:
    """
    Compute cumulative wrap angle of snake around cylinder.

    This measures how much the snake body wraps around the cylinder
    by summing angular differences between consecutive nodes.

    Args:
        positions: (N, 3) node positions
        cylinder: Cylinder target

    Returns:
        Cumulative wrap angle in radians (positive = wrapping)
    """
    # Get positions relative to cylinder center (XY plane)
    rel_pos = positions[:, :2] - cylinder.center[:2]

    # Compute angles for each node
    angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])

    # Compute angular differences between consecutive nodes
    angle_diffs = np.diff(angles)

    # Handle wraparound (-pi to pi)
    angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))

    # Sum absolute angular changes (cumulative wrap)
    total_wrap = np.sum(np.abs(angle_diffs))

    return float(total_wrap)


def compute_head_direction(positions: np.ndarray) -> np.ndarray:
    """
    Compute the forward direction vector of the snake head.

    Args:
        positions: (N, 3) node positions (node 0 is tail, node N-1 is head)

    Returns:
        (3,) unit vector pointing in snake's forward direction
    """
    # Direction from second-to-last to last node
    head_dir = positions[-1] - positions[-2]
    norm = np.linalg.norm(head_dir)
    if norm < 1e-8:
        # Fallback if nodes are coincident
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (head_dir / norm).astype(np.float32)


def compute_forward_velocity(positions: np.ndarray, velocities: np.ndarray) -> float:
    """
    Compute forward velocity component of snake head.

    Args:
        positions: (N, 3) node positions
        velocities: (N, 3) node velocities

    Returns:
        Scalar forward velocity (positive = moving forward)
    """
    head_dir = compute_head_direction(positions)
    head_vel = velocities[-1]
    return float(np.dot(head_vel, head_dir))


class SnakeVisualizer:
    """Matplotlib-based visualizer for snake environments."""

    def __init__(self):
        plt.ion()
        self._fig = plt.figure(figsize=(10, 8))
        self._ax = self._fig.add_subplot(111, projection='3d')

        # Set axis limits
        self._ax.set_xlim(-0.5, 2.0)
        self._ax.set_ylim(-1.0, 1.0)
        self._ax.set_zlim(0, 0.5)
        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')
        self._ax.set_zlabel('Z')

        # Snake body line
        self._snake_line, = self._ax.plot([], [], [], 'b-', linewidth=2, label='Snake')
        self._snake_head, = self._ax.plot([], [], [], 'ro', markersize=8, label='Head')

        # Cylinder surface
        self._cylinder_surface = None

    def render(self,
               positions: np.ndarray,
               cylinder: Cylinder,
               pause_time: float = 0.05):
        """
        Render the current snake state.

        Args:
            positions: (N, 3) node positions
            cylinder: Cylinder target
            pause_time: Time to pause for animation
        """
        # Update snake body
        self._snake_line.set_data(positions[:, 0], positions[:, 1])
        self._snake_line.set_3d_properties(positions[:, 2])

        # Update snake head
        self._snake_head.set_data([positions[-1, 0]], [positions[-1, 1]])
        self._snake_head.set_3d_properties([positions[-1, 2]])

        # Draw cylinder (only once or when changed)
        if self._cylinder_surface is None:
            self._draw_cylinder(cylinder)

        plt.draw()
        plt.pause(pause_time)

    def _draw_cylinder(self, cylinder: Cylinder):
        """Draw cylinder surface."""
        # Create cylinder mesh
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(0, cylinder.height, 10)
        theta, z = np.meshgrid(theta, z)

        x = cylinder.center[0] + cylinder.radius * np.cos(theta)
        y = cylinder.center[1] + cylinder.radius * np.sin(theta)

        self._cylinder_surface = self._ax.plot_surface(
            x, y, z, alpha=0.3, color='green', label='Cylinder')

    def clear_cylinder(self):
        """Clear cylinder surface for reset."""
        if self._cylinder_surface is not None:
            self._cylinder_surface.remove()
            self._cylinder_surface = None

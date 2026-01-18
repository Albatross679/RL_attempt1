"""
Potential functions for Potential-Based Reward Shaping (PBRS).

PBRS adds a shaping term: R' = R + F where F = γΦ(s') - Φ(s)
with Φ(terminal) = 0.

This module provides potential functions for:
- Approach task: distance/alignment-based potentials
- Coil task: wrapping progress-based potentials
"""
import numpy as np
from typing import Optional


# =============================================================================
# Approach Task Potential Functions
# =============================================================================

def approach_simple_distance(cyl_rel_pos: np.ndarray,
                             scale: float = 1.0,
                             d_max: float = 2.0) -> float:
    """
    Simple distance-based potential for approach task.

    Φ(s) = (d_max - dist) / d_max * scale

    Higher potential when closer to cylinder.

    Args:
        cyl_rel_pos: (3,) cylinder position relative to snake head
        scale: scaling factor for potential
        d_max: maximum distance for normalization

    Returns:
        Potential value (0 at d_max, scale at d=0)
    """
    dist = np.linalg.norm(cyl_rel_pos[:2])  # XY distance
    dist = np.clip(dist, 0.0, d_max)
    return (d_max - dist) / d_max * scale


def approach_distance_alignment(cyl_rel_pos: np.ndarray,
                                head_dir: np.ndarray,
                                alpha: float = 1.0,
                                beta: float = 0.5,
                                d_max: float = 2.0) -> float:
    """
    Distance + alignment potential for approach task.

    Φ(s) = α * (d_max - dist) / d_max + β * dot(head_dir, cyl_dir)

    Combines proximity with heading alignment toward cylinder.

    Args:
        cyl_rel_pos: (3,) cylinder position relative to snake head
        head_dir: (3,) unit vector of head direction
        alpha: weight for distance term
        beta: weight for alignment term
        d_max: maximum distance for normalization

    Returns:
        Potential value
    """
    dist = np.linalg.norm(cyl_rel_pos[:2])
    dist = np.clip(dist, 0.0, d_max)

    # Distance term
    distance_term = (d_max - dist) / d_max * alpha

    # Alignment term: dot product of head direction with direction to cylinder
    if dist > 1e-6:
        cyl_dir = cyl_rel_pos / (np.linalg.norm(cyl_rel_pos) + 1e-8)
        alignment = np.dot(head_dir, cyl_dir)
        alignment_term = alignment * beta
    else:
        alignment_term = beta  # Max alignment when at target

    return distance_term + alignment_term


def approach_time_to_goal(cyl_rel_pos: np.ndarray,
                          com_vel: np.ndarray,
                          epsilon: float = 0.01,
                          scale: float = 1.0) -> float:
    """
    Time-to-goal potential based on closing velocity.

    Φ(s) = -dist / max(closing_vel, ε) * scale

    Negative time estimate rewards faster approach.

    Args:
        cyl_rel_pos: (3,) cylinder position relative to snake head
        com_vel: (3,) center-of-mass velocity of snake
        epsilon: minimum closing velocity to avoid division issues
        scale: scaling factor

    Returns:
        Potential value (negative, higher when approaching faster)
    """
    dist = np.linalg.norm(cyl_rel_pos[:2])

    # Closing velocity: component of velocity toward cylinder
    if dist > 1e-6:
        cyl_dir = cyl_rel_pos[:2] / dist
        closing_vel = np.dot(com_vel[:2], cyl_dir)
    else:
        closing_vel = 0.0

    # Clamp closing velocity to positive (approaching)
    closing_vel = max(closing_vel, epsilon)

    # Negative time-to-goal (higher potential = less time)
    return -dist / closing_vel * scale


def approach_exp_distance(cyl_rel_pos: np.ndarray,
                          sigma: float = 0.3,
                          d_success: float = 0.15) -> float:
    """
    Exponential distance potential for approach task.

    Φ(s) = exp(-dist/σ) - exp(-d_success/σ)

    Sharp increase near target, zero at success threshold.

    Args:
        cyl_rel_pos: (3,) cylinder position relative to snake head
        sigma: decay rate (smaller = sharper near target)
        d_success: distance threshold for success (potential = 0 here)

    Returns:
        Potential value (0 at d_success, positive when closer)
    """
    dist = np.linalg.norm(cyl_rel_pos[:2])
    return np.exp(-dist / sigma) - np.exp(-d_success / sigma)


# =============================================================================
# Coil Task Potential Functions
# =============================================================================

def coil_wrap_progress(wrap_angle: float,
                       scale: float = 1.0) -> float:
    """
    Simple wrap progress potential for coil task.

    Φ(s) = (wrap / 2π) * scale

    Linear potential based on wrap angle.

    Args:
        wrap_angle: cumulative wrap angle in radians
        scale: scaling factor

    Returns:
        Potential value (0 at 0 wrap, scale at full wrap)
    """
    return (wrap_angle / (2 * np.pi)) * scale


def coil_wrap_contact(wrap_angle: float,
                      node_dists: np.ndarray,
                      cyl_radius: float,
                      alpha: float = 1.0,
                      beta: float = 0.5,
                      lambda_: float = 0.1) -> float:
    """
    Wrap + contact potential for coil task.

    Φ(s) = α * wrap + β * contact_score - λ * avg_dist

    Combines wrapping progress with contact quality.

    Args:
        wrap_angle: cumulative wrap angle in radians
        node_dists: (N,) signed distances from nodes to cylinder surface
        cyl_radius: cylinder radius
        alpha: weight for wrap term
        beta: weight for contact term
        lambda_: weight for distance penalty

    Returns:
        Potential value
    """
    # Wrap term (normalized to 2π)
    wrap_term = (wrap_angle / (2 * np.pi)) * alpha

    # Contact score: fraction of nodes within threshold of surface
    contact_threshold = cyl_radius * 0.5
    num_close = np.sum(np.abs(node_dists) < contact_threshold)
    contact_score = num_close / len(node_dists)
    contact_term = contact_score * beta

    # Distance penalty: average distance to surface
    avg_dist = np.mean(np.abs(node_dists))
    distance_penalty = avg_dist * lambda_

    return wrap_term + contact_term - distance_penalty


def coil_geometric(wrap_angle: float,
                   node_dists: np.ndarray,
                   d_max: float = 0.5) -> float:
    """
    Geometric potential combining wrap and proximity.

    Φ(s) = wrap * (1 - avg_dist / d_max)

    Wrap progress scaled by closeness to cylinder.

    Args:
        wrap_angle: cumulative wrap angle in radians
        node_dists: (N,) signed distances from nodes to cylinder surface
        d_max: maximum distance for normalization

    Returns:
        Potential value
    """
    avg_dist = np.mean(np.abs(node_dists))
    avg_dist = np.clip(avg_dist, 0.0, d_max)

    proximity_factor = 1.0 - avg_dist / d_max
    return wrap_angle * proximity_factor


# =============================================================================
# Potential Function Registry
# =============================================================================

APPROACH_POTENTIALS = {
    "simple_distance": approach_simple_distance,
    "distance_alignment": approach_distance_alignment,
    "time_to_goal": approach_time_to_goal,
    "exp_distance": approach_exp_distance,
}

COIL_POTENTIALS = {
    "wrap_progress": coil_wrap_progress,
    "wrap_contact": coil_wrap_contact,
    "geometric": coil_geometric,
}


def get_approach_potential(potential_type: str):
    """Get approach potential function by name."""
    if potential_type not in APPROACH_POTENTIALS:
        raise ValueError(f"Unknown approach potential: {potential_type}. "
                        f"Available: {list(APPROACH_POTENTIALS.keys())}")
    return APPROACH_POTENTIALS[potential_type]


def get_coil_potential(potential_type: str):
    """Get coil potential function by name."""
    if potential_type not in COIL_POTENTIALS:
        raise ValueError(f"Unknown coil potential: {potential_type}. "
                        f"Available: {list(COIL_POTENTIALS.keys())}")
    return COIL_POTENTIALS[potential_type]

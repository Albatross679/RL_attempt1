"""
Unit tests for Potential-Based Reward Shaping (PBRS) functions.

Tests each potential function for:
- Correct output values for known inputs
- Terminal state returns 0
- Edge cases (zero distance, zero velocity, etc.)
"""
import numpy as np
import pytest

from environments.utils.potential_functions import (
    # Approach potentials
    approach_simple_distance,
    approach_distance_alignment,
    approach_time_to_goal,
    approach_exp_distance,
    # Coil potentials
    coil_wrap_progress,
    coil_wrap_contact,
    coil_geometric,
    # Registry functions
    get_approach_potential,
    get_coil_potential,
    APPROACH_POTENTIALS,
    COIL_POTENTIALS,
)


# =============================================================================
# Test: approach_simple_distance
# =============================================================================

class TestApproachSimpleDistance:
    def test_at_target(self):
        """At distance=0, potential should equal scale."""
        cyl_rel_pos = np.array([0.0, 0.0, 0.0])
        result = approach_simple_distance(cyl_rel_pos, scale=1.0, d_max=2.0)
        assert np.isclose(result, 1.0)

    def test_at_max_distance(self):
        """At distance=d_max, potential should be 0."""
        cyl_rel_pos = np.array([2.0, 0.0, 0.0])
        result = approach_simple_distance(cyl_rel_pos, scale=1.0, d_max=2.0)
        assert np.isclose(result, 0.0)

    def test_halfway(self):
        """At distance=d_max/2, potential should be scale/2."""
        cyl_rel_pos = np.array([1.0, 0.0, 0.0])
        result = approach_simple_distance(cyl_rel_pos, scale=1.0, d_max=2.0)
        assert np.isclose(result, 0.5)

    def test_beyond_max(self):
        """Beyond d_max, potential should clamp to 0."""
        cyl_rel_pos = np.array([5.0, 0.0, 0.0])
        result = approach_simple_distance(cyl_rel_pos, scale=1.0, d_max=2.0)
        assert np.isclose(result, 0.0)

    def test_xy_only(self):
        """Only XY distance should be used."""
        cyl_rel_pos = np.array([1.0, 0.0, 10.0])  # Large Z shouldn't matter
        result = approach_simple_distance(cyl_rel_pos, scale=1.0, d_max=2.0)
        assert np.isclose(result, 0.5)

    def test_scaling(self):
        """Test scale parameter."""
        cyl_rel_pos = np.array([0.0, 0.0, 0.0])
        result = approach_simple_distance(cyl_rel_pos, scale=5.0, d_max=2.0)
        assert np.isclose(result, 5.0)


# =============================================================================
# Test: approach_distance_alignment
# =============================================================================

class TestApproachDistanceAlignment:
    def test_at_target_aligned(self):
        """At target with perfect alignment, get max potential."""
        cyl_rel_pos = np.array([0.0, 0.0, 0.0])
        head_dir = np.array([1.0, 0.0, 0.0])
        result = approach_distance_alignment(
            cyl_rel_pos, head_dir, alpha=1.0, beta=0.5, d_max=2.0
        )
        # At target: distance_term = 1.0, alignment_term = beta
        assert np.isclose(result, 1.5)

    def test_at_max_aligned(self):
        """At max distance, perfectly aligned toward target."""
        cyl_rel_pos = np.array([2.0, 0.0, 0.0])
        head_dir = np.array([1.0, 0.0, 0.0])
        result = approach_distance_alignment(
            cyl_rel_pos, head_dir, alpha=1.0, beta=0.5, d_max=2.0
        )
        # distance_term = 0, alignment = 1.0 * 0.5 = 0.5
        assert np.isclose(result, 0.5)

    def test_perpendicular(self):
        """Head perpendicular to target direction."""
        cyl_rel_pos = np.array([1.0, 0.0, 0.0])
        head_dir = np.array([0.0, 1.0, 0.0])  # Perpendicular
        result = approach_distance_alignment(
            cyl_rel_pos, head_dir, alpha=1.0, beta=0.5, d_max=2.0
        )
        # distance_term = 0.5, alignment = 0 * 0.5 = 0
        assert np.isclose(result, 0.5)

    def test_facing_away(self):
        """Head facing away from target."""
        cyl_rel_pos = np.array([1.0, 0.0, 0.0])
        head_dir = np.array([-1.0, 0.0, 0.0])  # Facing away
        result = approach_distance_alignment(
            cyl_rel_pos, head_dir, alpha=1.0, beta=0.5, d_max=2.0
        )
        # distance_term = 0.5, alignment = -1.0 * 0.5 = -0.5
        assert np.isclose(result, 0.0)


# =============================================================================
# Test: approach_time_to_goal
# =============================================================================

class TestApproachTimeToGoal:
    def test_approaching(self):
        """Moving toward target should give higher potential."""
        cyl_rel_pos = np.array([1.0, 0.0, 0.0])
        com_vel = np.array([0.5, 0.0, 0.0])  # Moving toward target
        result = approach_time_to_goal(
            cyl_rel_pos, com_vel, epsilon=0.01, scale=1.0
        )
        # time = 1.0 / 0.5 = 2.0, potential = -2.0
        assert np.isclose(result, -2.0)

    def test_zero_velocity(self):
        """Zero velocity should use epsilon."""
        cyl_rel_pos = np.array([1.0, 0.0, 0.0])
        com_vel = np.array([0.0, 0.0, 0.0])
        result = approach_time_to_goal(
            cyl_rel_pos, com_vel, epsilon=0.01, scale=1.0
        )
        # time = 1.0 / 0.01 = 100, potential = -100
        assert np.isclose(result, -100.0)

    def test_moving_away(self):
        """Moving away uses epsilon (negative closing vel clamped)."""
        cyl_rel_pos = np.array([1.0, 0.0, 0.0])
        com_vel = np.array([-0.5, 0.0, 0.0])  # Moving away
        result = approach_time_to_goal(
            cyl_rel_pos, com_vel, epsilon=0.01, scale=1.0
        )
        # Negative closing vel clamped to epsilon
        assert np.isclose(result, -100.0)


# =============================================================================
# Test: approach_exp_distance
# =============================================================================

class TestApproachExpDistance:
    def test_at_success_threshold(self):
        """At success threshold, potential should be 0."""
        d_success = 0.15
        cyl_rel_pos = np.array([d_success, 0.0, 0.0])
        result = approach_exp_distance(cyl_rel_pos, sigma=0.3, d_success=d_success)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_closer_than_threshold(self):
        """Closer than threshold should be positive."""
        cyl_rel_pos = np.array([0.05, 0.0, 0.0])
        result = approach_exp_distance(cyl_rel_pos, sigma=0.3, d_success=0.15)
        assert result > 0

    def test_farther_than_threshold(self):
        """Farther than threshold should be negative."""
        cyl_rel_pos = np.array([0.5, 0.0, 0.0])
        result = approach_exp_distance(cyl_rel_pos, sigma=0.3, d_success=0.15)
        assert result < 0


# =============================================================================
# Test: coil_wrap_progress
# =============================================================================

class TestCoilWrapProgress:
    def test_zero_wrap(self):
        """No wrap should give 0 potential."""
        result = coil_wrap_progress(0.0, scale=1.0)
        assert np.isclose(result, 0.0)

    def test_full_wrap(self):
        """Full wrap (2*pi) should give scale."""
        result = coil_wrap_progress(2 * np.pi, scale=1.0)
        assert np.isclose(result, 1.0)

    def test_half_wrap(self):
        """Half wrap (pi) should give scale/2."""
        result = coil_wrap_progress(np.pi, scale=1.0)
        assert np.isclose(result, 0.5)

    def test_scaling(self):
        """Test scale parameter."""
        result = coil_wrap_progress(2 * np.pi, scale=5.0)
        assert np.isclose(result, 5.0)


# =============================================================================
# Test: coil_wrap_contact
# =============================================================================

class TestCoilWrapContact:
    def test_zero_wrap_far_nodes(self):
        """Zero wrap with far nodes gives penalty."""
        node_dists = np.array([0.5, 0.5, 0.5, 0.5])
        result = coil_wrap_contact(
            0.0, node_dists, cyl_radius=0.08,
            alpha=1.0, beta=0.5, lambda_=0.1
        )
        # wrap_term = 0, contact_term = 0 (no close nodes), dist_penalty = 0.5 * 0.1 = 0.05
        assert result < 0

    def test_full_wrap_close_nodes(self):
        """Full wrap with all close nodes gives high potential."""
        cyl_radius = 0.08
        node_dists = np.array([0.01, 0.01, 0.01, 0.01])  # All within threshold
        result = coil_wrap_contact(
            2 * np.pi, node_dists, cyl_radius=cyl_radius,
            alpha=1.0, beta=0.5, lambda_=0.1
        )
        # wrap_term = 1.0, contact_term = 0.5, dist_penalty = 0.01 * 0.1 = 0.001
        assert result > 1.0


# =============================================================================
# Test: coil_geometric
# =============================================================================

class TestCoilGeometric:
    def test_zero_wrap(self):
        """Zero wrap gives 0 regardless of distance."""
        node_dists = np.array([0.1, 0.1])
        result = coil_geometric(0.0, node_dists, d_max=0.5)
        assert np.isclose(result, 0.0)

    def test_full_wrap_close(self):
        """Full wrap with nodes at surface gives max potential."""
        node_dists = np.array([0.0, 0.0])  # On surface
        result = coil_geometric(2 * np.pi, node_dists, d_max=0.5)
        assert np.isclose(result, 2 * np.pi)

    def test_full_wrap_far(self):
        """Full wrap with max distance nodes gives 0."""
        node_dists = np.array([0.5, 0.5])  # At d_max
        result = coil_geometric(2 * np.pi, node_dists, d_max=0.5)
        assert np.isclose(result, 0.0)


# =============================================================================
# Test: Registry Functions
# =============================================================================

class TestRegistry:
    def test_get_approach_potential(self):
        """Test all approach potentials can be retrieved."""
        for name in APPROACH_POTENTIALS:
            func = get_approach_potential(name)
            assert callable(func)

    def test_get_coil_potential(self):
        """Test all coil potentials can be retrieved."""
        for name in COIL_POTENTIALS:
            func = get_coil_potential(name)
            assert callable(func)

    def test_invalid_approach_potential(self):
        """Invalid name should raise ValueError."""
        with pytest.raises(ValueError):
            get_approach_potential("invalid_name")

    def test_invalid_coil_potential(self):
        """Invalid name should raise ValueError."""
        with pytest.raises(ValueError):
            get_coil_potential("invalid_name")


# =============================================================================
# Test: PBRS Terminal State Property
# =============================================================================

class TestTerminalState:
    """
    PBRS requires Φ(terminal) = 0.
    This is enforced in the env code, not the potential functions.
    These tests document that potential functions themselves don't return 0
    for arbitrary "terminal" inputs - the env must handle this.
    """

    def test_approach_potentials_nonzero_at_goal(self):
        """Approach potentials at goal position are generally non-zero."""
        cyl_rel_pos = np.array([0.0, 0.0, 0.0])
        head_dir = np.array([1.0, 0.0, 0.0])
        com_vel = np.array([0.1, 0.0, 0.0])

        # simple_distance: at target = scale
        assert approach_simple_distance(cyl_rel_pos) == 1.0

        # exp_distance: at target (d=0) is positive
        assert approach_exp_distance(cyl_rel_pos) > 0

    def test_coil_potentials_nonzero_at_full_wrap(self):
        """Coil potentials at full wrap are generally non-zero."""
        wrap = 2 * np.pi
        node_dists = np.array([0.0, 0.0])

        # wrap_progress: full wrap = scale
        assert coil_wrap_progress(wrap) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

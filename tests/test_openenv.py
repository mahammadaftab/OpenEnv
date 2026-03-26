"""
Test Suite for OpenEnv

Comprehensive tests covering:
- Environment initialization
- API compliance
- Physics and dynamics
- Reward computation
- Termination conditions
- Rendering functionality
- Configuration system
"""

import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv import OpenEnv, EnvConfig


class TestEnvInitialization:
    """Test environment initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default config."""
        env = OpenEnv()
        assert env is not None
        assert isinstance(env.config, EnvConfig)
        env.close()
    
    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = EnvConfig(
            episode_length=500,
            verbose=False,
            random_seed=42,
        )
        env = OpenEnv(config=config)
        assert env.config.episode_length == 500
        assert env.config.random_seed == 42
        env.close()
    
    def test_invalid_config(self):
        """Test that invalid config raises error."""
        config = EnvConfig(episode_length=-100)
        with pytest.raises(ValueError):
            env = OpenEnv(config=config)
            env.close()
    
    def test_render_mode_override(self):
        """Test render mode override in constructor."""
        env = OpenEnv(render_mode='rgb_array')
        assert env.config.render_mode == 'rgb_array'
        env.close()


class TestAPICompliance:
    """Test Gymnasium API compliance."""
    
    def test_gymnasium_check(self):
        """Run Gymnasium's environment checker."""
        config = EnvConfig(verbose=False)
        env = OpenEnv(config=config)
        
        # This runs all standard checks
        check_env(env, warn=True, skip_render_check=True)
        
        env.close()
    
    def test_reset_returns_observation_and_info(self):
        """Test reset returns correct format."""
        env = OpenEnv()
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape[0] == env.config.observation_dim
        assert not np.isnan(obs).any()
        
        env.close()
    
    def test_step_returns_correct_format(self):
        """Test step returns correct format."""
        env = OpenEnv()
        obs, _ = env.reset()
        
        action = env.action_space.sample()
        result = env.step(action)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_action_space_bounds(self):
        """Test action space bounds."""
        env = OpenEnv()
        
        # Check action space is bounded
        assert hasattr(env.action_space, 'low')
        assert hasattr(env.action_space, 'high')
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        
        env.close()
    
    def test_observation_space_finite(self):
        """Test observation space is properly bounded."""
        env = OpenEnv()
        obs, _ = env.reset()
        
        # Observations should be finite (not inf or nan)
        assert np.isfinite(obs).all()
        
        env.close()


class TestEnvironmentDynamics:
    """Test physics and environment dynamics."""
    
    def test_position_changes_after_action(self):
        """Test that position changes after taking actions."""
        env = OpenEnv()
        obs_before, _ = env.reset()
        
        # Take a strong action
        action = np.array([1.0, 0.0, 0.0, 0.0])
        obs_after, _, _, _, _ = env.step(action)
        
        # Position should change
        assert not np.array_equal(obs_before[0:2], obs_after[0:2])
        
        env.close()
    
    def test_velocity_changes_after_action(self):
        """Test that velocity changes after taking actions."""
        env = OpenEnv()
        env.reset()
        
        # Initial velocity should be zero
        obs, _, _, _, _ = env.step(np.zeros(4))
        initial_velocity = obs[2:4]
        
        # Apply action
        action = np.array([1.0, 1.0, 0.0, 0.0])
        obs, _, _, _, _ = env.step(action)
        new_velocity = obs[2:4]
        
        # Velocity should change
        assert not np.array_equal(initial_velocity, new_velocity)
        
        env.close()
    
    def test_gravity_affects_velocity(self):
        """Test that gravity affects vertical velocity."""
        config = EnvConfig(gravity=9.81, verbose=False)
        env = OpenEnv(config=config)
        env.reset()
        
        # Take no horizontal action
        action = np.zeros(4)
        obs, _, _, _, _ = env.step(action)
        
        # Vertical velocity should become negative due to gravity
        vy = obs[3]
        assert vy < 0
        
        env.close()
    
    def test_friction_slows_down_agent(self):
        """Test that friction reduces velocity."""
        config = EnvConfig(friction=0.1, verbose=False)
        env = OpenEnv(config=config)
        env.reset()
        
        # Apply strong action
        action = np.array([1.0, 0.0, 0.0, 0.0])
        env.step(action)
        
        # Apply opposite action
        action = np.array([-1.0, 0.0, 0.0, 0.0])
        obs, _, _, _, _ = env.step(action)
        
        # Velocity should be reduced
        vx = obs[2]
        assert abs(vx) < 1.0
        
        env.close()


class TestRewardFunction:
    """Test reward computation."""
    
    def test_reward_is_scalar(self):
        """Test that reward is a scalar."""
        env = OpenEnv()
        env.reset()
        
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        
        env.close()
    
    def test_closer_to_target_higher_reward(self):
        """Test that being closer to target yields better rewards."""
        config = EnvConfig(sparse_rewards=False, verbose=False)
        env = OpenEnv(config=config)
        
        # Reset multiple times and compare rewards
        rewards_close = []
        rewards_far = []
        
        for _ in range(10):
            env.reset()
            
            # Move toward target
            env.position = env.target_position * 0.5  # Close
            obs1, reward1, _, _, _ = env.step(np.zeros(4))
            rewards_close.append(reward1)
            
            env.position = env.target_position * 2.0  # Far
            obs2, reward2, _, _, _ = env.step(np.zeros(4))
            rewards_far.append(reward2)
        
        # On average, closer should have higher rewards
        assert np.mean(rewards_close) > np.mean(rewards_far)
        
        env.close()
    
    def test_sparse_rewards(self):
        """Test sparse reward configuration."""
        config = EnvConfig(sparse_rewards=True, verbose=False)
        env = OpenEnv(config=config)
        env.reset()
        
        # In sparse mode, small movements should give minimal reward
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        
        # Reward should be close to zero (only velocity penalty)
        assert abs(reward) < 1.0
        
        env.close()
    
    def test_reward_scaling(self):
        """Test reward scale parameter."""
        config1 = EnvConfig(reward_scale=1.0, verbose=False)
        config2 = EnvConfig(reward_scale=2.0, verbose=False)
        
        env1 = OpenEnv(config=config1)
        env2 = OpenEnv(config=config2)
        
        env1.reset(seed=42)
        env2.reset(seed=42)
        
        action = np.array([0.5, 0.0, 0.0, 0.0])
        
        _, reward1, _, _, _ = env1.step(action)
        _, reward2, _, _, _ = env2.step(action)
        
        # Scaled reward should be proportionally different
        assert abs(reward2) > abs(reward1)
        
        env1.close()
        env2.close()


class TestTerminationConditions:
    """Test termination and truncation conditions."""
    
    def test_episode_truncation(self):
        """Test that episode truncates at max steps."""
        config = EnvConfig(episode_length=50, verbose=False)
        env = OpenEnv(config=config)
        env.reset()
        
        # Run until truncation
        for i in range(config.episode_length):
            _, _, terminated, truncated, _ = env.step(np.zeros(4))
        
        assert truncated
        assert not terminated
        
        env.close()
    
    def test_boundary_termination(self):
        """Test termination on boundary violation."""
        config = EnvConfig(
            boundary_limit=10.0,
            terminate_on_boundary=True,
            verbose=False,
        )
        env = OpenEnv(config=config)
        env.reset()
        
        # Manually set position beyond boundary
        env.position = np.array([15.0, 0.0])
        
        _, _, terminated, _, _ = env.step(np.zeros(4))
        
        assert terminated
        
        env.close()
    
    def test_max_velocity_termination(self):
        """Test termination on max velocity violation."""
        config = EnvConfig(
            max_velocity=10.0,
            verbose=False,
        )
        env = OpenEnv(config=config)
        env.reset()
        
        # Manually set velocity beyond limit
        env.velocity = np.array([15.0, 0.0])
        
        _, _, terminated, _, _ = env.step(np.zeros(4))
        
        assert terminated
        
        env.close()
    
    def test_no_boundary_termination(self):
        """Test that boundary doesn't terminate when disabled."""
        config = EnvConfig(
            boundary_limit=10.0,
            terminate_on_boundary=False,
            verbose=False,
        )
        env = OpenEnv(config=config)
        env.reset()
        
        # Set position beyond boundary
        env.position = np.array([15.0, 0.0])
        
        _, _, terminated, _, _ = env.step(np.zeros(4))
        
        assert not terminated
        
        env.close()


class TestStateObservation:
    """Test state and observation computation."""
    
    def test_state_shape(self):
        """Test state vector shape."""
        env = OpenEnv()
        env.reset()
        
        state = env.state()
        assert state is not None
        assert state.shape[0] == env.config.observation_dim
        
        env.close()
    
    def test_observation_components(self):
        """Test observation contains expected components."""
        env = OpenEnv()
        env.reset()
        
        obs = env._compute_observation()
        
        # Check all components are present
        assert len(obs) == 8
        
        # Position (0:2)
        assert np.allclose(obs[0:2], env.position)
        
        # Velocity (2:4)
        assert np.allclose(obs[2:4], env.velocity)
        
        # Target (4:6)
        assert np.allclose(obs[4:6], env.target_position)
        
        # Time remaining (6)
        assert 0 <= obs[6] <= 1
        
        # Distance to target (7)
        expected_dist = np.linalg.norm(env.position - env.target_position)
        assert np.isclose(obs[7], expected_dist)
        
        env.close()
    
    def test_distance_computation(self):
        """Test distance to target calculation."""
        env = OpenEnv()
        env.reset()
        
        # Set specific positions
        env.position = np.array([3.0, 4.0])
        env.target_position = np.array([0.0, 0.0])
        
        obs = env._compute_observation()
        expected_distance = 5.0  # 3-4-5 triangle
        
        assert np.isclose(obs[7], expected_distance)
        
        env.close()


class TestReproducibility:
    """Test random seed and reproducibility."""
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        env1 = OpenEnv()
        env2 = OpenEnv()
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Should get identical results
        assert np.array_equal(obs1, obs2)
        
        env1.close()
        env2.close()
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        env1 = OpenEnv()
        env2 = OpenEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=43)
        
        # Should get different results (with high probability)
        assert not np.array_equal(obs1, obs2)
        
        env1.close()
        env2.close()


class TestRendering:
    """Test rendering functionality."""
    
    def test_rgb_array_rendering(self):
        """Test RGB array rendering."""
        config = EnvConfig(render_mode='rgb_array', verbose=False)
        env = OpenEnv(config=config)
        env.reset()
        
        frame = env.render()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # Height, Width, Channels
        assert frame.shape[2] == 3  # RGB channels
        
        env.close()
    
    def test_render_frame_size(self):
        """Test rendered frame size matches config."""
        config = EnvConfig(
            render_mode='rgb_array',
            screen_size=(800, 600),
            verbose=False,
        )
        env = OpenEnv(config=config)
        env.reset()
        
        frame = env.render()
        
        assert frame.shape[0] == 600  # Height
        assert frame.shape[1] == 800  # Width
        
        env.close()


class TestConfiguration:
    """Test configuration system."""
    
    def test_config_save_load(self, tmp_path):
        """Test saving and loading configuration."""
        config = EnvConfig(
            episode_length=500,
            gravity=5.0,
            verbose=False,
        )
        
        # Save to file
        filepath = tmp_path / "config.json"
        config.save(str(filepath))
        
        # Load from file
        loaded_config = EnvConfig.load(str(filepath))
        
        # Verify loaded config
        assert loaded_config.episode_length == 500
        assert loaded_config.gravity == 5.0
        assert loaded_config.verbose == False
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = EnvConfig(episode_length=300, verbose=False)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['episode_length'] == 300
        assert 'gravity' in config_dict
        assert 'friction' in config_dict
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            'episode_length': 400,
            'gravity': 7.5,
            'verbose': False,
        }
        
        config = EnvConfig.from_dict(config_dict)
        
        assert config.episode_length == 400
        assert config.gravity == 7.5
        assert config.verbose == False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_action_outside_bounds(self):
        """Test handling of actions outside valid range."""
        env = OpenEnv()
        env.reset()
        
        # Action outside bounds
        invalid_action = np.array([2.0, -2.0, 2.0, -2.0])
        
        # Should not crash, should clip action
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        # Environment should still function
        assert np.isfinite(obs).all()
        
        env.close()
    
    def test_multiple_resets(self):
        """Test multiple consecutive resets."""
        env = OpenEnv()
        
        for _ in range(5):
            obs, info = env.reset()
            assert obs is not None
            assert np.isfinite(obs).all()
        
        env.close()
    
    def test_many_steps_without_reset(self):
        """Test taking many steps without reset."""
        config = EnvConfig(episode_length=1000, verbose=False)
        env = OpenEnv(config=config)
        env.reset()
        
        # Take more steps than episode length
        for i in range(config.episode_length + 100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Should handle gracefully
            assert np.isfinite(obs).all()
        
        env.close()
    
    def test_zero_action(self):
        """Test zero action behavior."""
        env = OpenEnv()
        env.reset()
        
        # Zero action should still apply gravity
        obs, _, _, _, _ = env.step(np.zeros(4))
        
        # Vertical velocity should be negative (gravity)
        assert obs[3] < 0
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

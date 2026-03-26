"""
Configuration module for OpenEnv environment parameters.

Provides dataclass-based configuration with support for:
- Environment hyperparameters
- Reward function settings
- Termination conditions
- Rendering options
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class EnvConfig:
    """
    Configuration class for OpenEnv environment.
    
    Attributes:
        episode_length: Maximum number of steps per episode
        observation_dim: Dimension of the observation space (12 for 3D)
        action_dim: Dimension of the action space (4 for drone control)
        random_seed: Random seed for reproducibility
        
        # Environment dynamics
        gravity: Gravitational constant
        friction: Friction coefficient
        dt: Time step for physics simulation
        
        # Reward shaping
        reward_scale: Scaling factor for rewards
        sparse_rewards: Whether to use sparse rewards only
        reward_clip: Clip rewards to [-clip, clip]
        
        # Termination conditions
        max_velocity: Maximum allowed velocity before termination
        boundary_limit: Environment boundary limits
        terminate_on_boundary: End episode on boundary violation
        
        # Task difficulty
        task_level: 'easy', 'medium', or 'hard'
        obstacle_count: Number of obstacles
        wind_disturbance: Enable wind effects
        sensor_noise: Noise level in observations
        
        # Rendering
        render_mode: 'human', 'rgb_array', or None
        render_fps: Frames per second for rendering
        screen_size: Window size for rendering
        
        # Logging
        verbose: Enable verbose logging
        log_metrics: Track and log performance metrics
    """
    
    # Core environment settings
    episode_length: int = 500
    observation_dim: int = 12  # 3D navigation (pos:3, vel:3, target:3, time:1, dist:1, obstacle:1)
    action_dim: int = 4  # Drone control (thrust, yaw, pitch, roll)
    random_seed: Optional[int] = None
    
    # Physics parameters
    gravity: float = 9.81
    friction: float = 0.01
    dt: float = 0.02
    
    # Reward configuration
    reward_scale: float = 1.0
    sparse_rewards: bool = False
    reward_clip: Optional[float] = None
    
    # Termination conditions
    max_velocity: float = 100.0
    boundary_limit: float = 50.0
    terminate_on_boundary: bool = True
    
    # Task difficulty
    task_level: str = 'medium'
    obstacle_count: int = 5
    wind_disturbance: bool = False
    sensor_noise: float = 0.05
    target_radius: float = 4.0
    
    # Rendering options
    render_mode: Optional[str] = None
    render_fps: int = 60
    screen_size: Tuple[int, int] = (1024, 768)
    
    # Logging and monitoring
    verbose: bool = True
    log_metrics: bool = True
    
    # Custom parameters (for extensibility)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.render_mode not in [None, 'human', 'rgb_array']:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'episode_length': self.episode_length,
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'random_seed': self.random_seed,
            'gravity': self.gravity,
            'friction': self.friction,
            'dt': self.dt,
            'reward_scale': self.reward_scale,
            'sparse_rewards': self.sparse_rewards,
            'reward_clip': self.reward_clip,
            'max_velocity': self.max_velocity,
            'boundary_limit': self.boundary_limit,
            'terminate_on_boundary': self.terminate_on_boundary,
            'task_level': self.task_level,
            'obstacle_count': self.obstacle_count,
            'wind_disturbance': self.wind_disturbance,
            'sensor_noise': self.sensor_noise,
            'target_radius': self.target_radius,
            'render_mode': self.render_mode,
            'render_fps': self.render_fps,
            'screen_size': self.screen_size,
            'verbose': self.verbose,
            'log_metrics': self.log_metrics,
            'custom_params': self.custom_params,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> 'EnvConfig':
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return EnvConfig.from_dict(config_dict)

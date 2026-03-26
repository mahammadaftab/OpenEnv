"""
OpenEnv - Production-Ready Reinforcement Learning Environment

A Gymnasium-compatible environment implementing the standard step(), reset(), 
and state() API for AI agent training.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import gymnasium as gym
from gymnasium import spaces
import pygame
import logging
import time

from openenv.core.config import EnvConfig


class OpenEnv(gym.Env):
    """
    A production-ready reinforcement learning environment.
    
    This environment implements a generic control task where an agent must
    navigate to a target position while avoiding boundaries and managing velocity.
    
    The environment features:
    - Full Gymnasium API compliance (step, reset, state)
    - Configurable physics and reward parameters
    - Multiple render modes (human, rgb_array)
    - Comprehensive logging and metrics tracking
    - Deterministic behavior with seed support
    
    Observation Space (8-dimensional):
        - Position (x, y): Agent's current position
        - Velocity (vx, vy): Agent's current velocity
        - Target (tx, ty): Target position
        - Time remaining: Normalized time left in episode
        - Distance to target: Euclidean distance to goal
        
    Action Space (4-dimensional continuous):
        - Continuous force vector (fx, fy) to apply
        - Values normalized to [-1, 1]
        
    Reward Function:
        - Dense reward: Negative distance to target
        - Sparse reward: +100 for reaching target
        - Penalty: -50 for boundary violation
        - Shaping: Small penalty for high velocity
        
    Termination Conditions:
        - Episode length exceeded
        - Boundary violation (optional)
        - Maximum velocity exceeded (optional)
        
    Example:
        >>> from openenv import OpenEnv, EnvConfig
        >>> config = EnvConfig(episode_length=500, verbose=True)
        >>> env = OpenEnv(config)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> env.render()
        >>> env.close()
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60,
    }
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the OpenEnv environment.
        
        Args:
            config: Environment configuration. Uses default if None.
            render_mode: Render mode ('human', 'rgb_array', or None)
            
        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__()
        
        # Configuration
        self.config = config if config is not None else EnvConfig()
        self.config.validate()
        
        # Override render mode if provided
        if render_mode is not None:
            self.config.render_mode = render_mode
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            self.seed(self.config.random_seed)
        
        # Logging setup
        self._setup_logging()
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Environment state
        self._state_vector: Optional[np.ndarray] = None
        self.position: np.ndarray = np.zeros(3)  # 3D position (x, y, z)
        self.velocity: np.ndarray = np.zeros(3)  # 3D velocity (vx, vy, vz)
        self.target_position: np.ndarray = np.zeros(3)  # 3D target
        self.time_remaining: float = 1.0
        self.steps_taken: int = 0
        
        # Metrics tracking
        self.metrics: Dict[str, Any] = {}
        self.episode_return: float = 0.0
        
        # Rendering
        self.screen = None
        self.clock = None
        self.render_initialized = False
        
        if self.config.verbose:
            self.logger.info("OpenEnv initialized successfully")
            self.logger.info(f"Configuration: episode_length={self.config.episode_length}, "
                           f"observation_dim={self.config.observation_dim}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger('OpenEnv')
        self.logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation space
        obs_high = np.inf * np.ones(self.config.observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )
        
        # Action space (continuous control)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.action_dim,),
            dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional initialization options
            
        Returns:
            observation: Initial observation
            info: Additional information (empty dict by default)
            
        Example:
            >>> obs, info = env.reset()
        """
        # Handle seeding
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        # Reset state variables
        self.steps_taken = 0
        self.time_remaining = 1.0
        self.episode_return = 0.0
        self.metrics = {
            'steps': 0,
            'return': 0.0,
            'target_reached': False,
            'terminated': False,
            'truncated': False,
        }
        
        # Initialize agent position (random or at origin)
        if options and options.get('random_start', True):
            self.position = self.np_random.uniform(
                low=-self.config.boundary_limit * 0.5,
                high=self.config.boundary_limit * 0.5,
                size=3  # 3D position
            ).astype(np.float32)
        else:
            self.position = np.zeros(3, dtype=np.float32)
        
        # Initialize velocity to zero
        self.velocity = np.zeros(3, dtype=np.float32)
        
        # Set target position
        self.target_position = self.np_random.uniform(
            low=-self.config.boundary_limit * 0.8,
            high=self.config.boundary_limit * 0.8,
            size=3  # 3D target
        ).astype(np.float32)
        
        # Compute initial observation
        observation = self._compute_observation()
        
        # Reset rendering if needed
        if self.config.render_mode == 'human' and self.screen is not None:
            self.render_initialized = False
        
        if self.config.verbose:
            self.logger.info(f"Environment reset. Initial position: {self.position}, "
                           f"Target: {self.target_position}")
        
        return observation, {}
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Action to execute (force vector)
            
        Returns:
            observation: New observation after taking action
            reward: Reward received from taking action
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated (time limit)
            info: Additional information
            
        Example:
            >>> action = env.action_space.sample()
            >>> obs, reward, terminated, truncated, info = env.step(action)
        """
        # Validate action
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            self.logger.warning(f"Action {action} outside action space bounds. Clipping.")
            action = np.clip(action, -1.0, 1.0)
        
        # Apply action and update physics
        self._apply_action(action)
        
        # Update time
        self.steps_taken += 1
        self.time_remaining = 1.0 - (self.steps_taken / self.config.episode_length)
        
        # Compute reward
        reward = self._compute_reward()
        self.episode_return += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self._check_truncation()
        
        # Compute new observation
        observation = self._compute_observation()
        
        # Update metrics
        self.metrics['steps'] = self.steps_taken
        self.metrics['return'] = self.episode_return
        
        # Log if verbose
        if self.config.verbose and self.steps_taken % 100 == 0:
            self.logger.info(f"Step {self.steps_taken}: reward={reward:.3f}, "
                           f"total_return={self.episode_return:.3f}")
        
        return observation, reward, terminated, truncated, self.metrics
    
    def state(self) -> Optional[np.ndarray]:
        """
        Get the current internal state of the environment.
        
        Returns:
            Complete state vector including all internal variables
            
        Note:
            Different from observation - this provides full state access
            for debugging and analysis purposes.
        """
        if self._state_vector is None:
            return None
        
        return self._state_vector.copy()
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode='rgb_array', None if 'human'
            
        Example:
            >>> env.render()  # For human mode
            >>> frame = env.render()  # For rgb_array mode
        """
        if self.config.render_mode is None:
            return None
        
        if not self.render_initialized:
            self._initialize_rendering()
        
        if self.config.render_mode == 'human':
            return self._render_human()
        elif self.config.render_mode == 'rgb_array':
            return self._render_rgb_array()
        
        return None
    
    def close(self) -> None:
        """Clean up resources and close the environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.render_initialized = False
        
        if self.config.verbose:
            self.logger.info("Environment closed")
    
    def seed(self, seed: Optional[int] = None) -> int:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
            
        Returns:
            The seed used
        """
        if seed is None:
            seed = int(time.time() * 1000) % 2**31
        
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.config.random_seed = seed
        
        if self.config.verbose:
            self.logger.info(f"Random seed set to {seed}")
        
        return seed
    
    # ========================================================================
    # Core Physics and Dynamics
    # ========================================================================
    
    def _apply_action(self, action: np.ndarray) -> None:
        """
        Apply action to environment and update physics.
        
        Args:
            action: Force vector to apply (4D: thrust, yaw, pitch, roll)
        """
        # Scale action to physical force
        force_scale = 10.0  # Newtons
        
        # Map 4D action to 3D forces
        # action[0] = thrust (affects z-axis)
        # action[1] = yaw (rotation, not used for translation)
        # action[2] = pitch (affects x-axis)
        # action[3] = roll (affects y-axis)
        force = np.zeros(3)
        force[0] = action[2] * force_scale  # pitch -> x force
        force[1] = action[3] * force_scale  # roll -> y force
        force[2] = action[0] * force_scale  # thrust -> z force
        
        # Compute acceleration (F = ma)
        mass = 1.5  # kg (drone mass)
        acceleration = force / mass
        
        # Apply gravity (only affects z-axis)
        acceleration[2] -= self.config.gravity
        
        # Apply friction (air resistance)
        friction_force = -self.config.friction * self.velocity
        acceleration += friction_force / mass
        
        # Update velocity (Euler integration)
        self.velocity += acceleration * self.config.dt
        
        # Clip maximum velocity
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > self.config.max_velocity:
            self.velocity = (self.velocity / velocity_norm) * self.config.max_velocity
        
        # Update position
        self.position += self.velocity * self.config.dt
    
    def _compute_observation(self) -> np.ndarray:
        """
        Compute observation vector from current state.
        
        Returns:
            Observation array
        """
        # Distance to target
        distance_to_target = self.target_position - self.position
        distance_norm = np.linalg.norm(distance_to_target)
        
        # Normalize time
        time_normalized = self.time_remaining
        
        # Construct observation vector (12D for 3D navigation)
        observation = np.concatenate([
            self.position,           # x, y, z (3)
            self.velocity,           # vx, vy, vz (3)
            self.target_position,    # tx, ty, tz (3)
            [time_normalized],       # time remaining (1)
            [distance_norm],         # distance to target (1)
            [0.0],                   # placeholder for obstacle info (1)
        ])
        
        # Store full state
        self._state_vector = observation.copy()
        
        return observation.astype(np.float32)
    
    def _compute_reward(self) -> float:
        """
        Compute reward for current state.
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Distance to target (dense reward)
        distance = np.linalg.norm(self.position - self.target_position)
        
        if not self.config.sparse_rewards:
            # Negative distance as reward (closer = better)
            reward -= distance * 0.1
        
        # Reward shaping: penalize high velocity
        velocity_penalty = -0.01 * np.linalg.norm(self.velocity)
        reward += velocity_penalty
        
        # Reward for making progress toward target
        if hasattr(self, '_prev_distance'):
            progress = self._prev_distance - distance
            reward += progress * 0.5  # Reward improvement
        self._prev_distance = distance
        
        # Sparse reward for reaching target
        if distance < 1.0:  # Success threshold
            reward += 100.0 * self.config.reward_scale
            self.metrics['target_reached'] = True
        
        # Apply reward scaling
        reward *= self.config.reward_scale
        
        # Clip rewards if configured
        if self.config.reward_clip is not None:
            reward = np.clip(reward, -self.config.reward_clip, self.config.reward_clip)
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if termination conditions are met
        """
        terminated = False
        
        # Check boundary violation
        if self.config.terminate_on_boundary:
            position_norm = np.linalg.norm(self.position)
            if position_norm > self.config.boundary_limit:
                terminated = True
                self.metrics['terminated'] = True
                if self.config.verbose:
                    self.logger.info(f"Episode terminated: boundary violated "
                                   f"(position={position_norm:.2f})")
        
        # Check velocity violation
        velocity_norm = np.linalg.norm(self.velocity)
        if velocity_norm > self.config.max_velocity:
            terminated = True
            self.metrics['terminated'] = True
            if self.config.verbose:
                self.logger.info(f"Episode terminated: max velocity exceeded "
                               f"(velocity={velocity_norm:.2f})")
        
        return terminated
    
    def _check_truncation(self) -> bool:
        """
        Check if episode should be truncated (time limit).
        
        Returns:
            True if truncation conditions are met
        """
        truncated = self.steps_taken >= self.config.episode_length
        
        if truncated:
            self.metrics['truncated'] = True
            if self.config.verbose:
                self.logger.info(f"Episode truncated: max steps reached "
                               f"({self.config.episode_length})")
        
        return truncated
    
    # ========================================================================
    # Rendering Methods
    # ========================================================================
    
    def _initialize_rendering(self) -> None:
        """Initialize Pygame rendering system."""
        if pygame.get_init() is None:
            pygame.init()
        
        # Initialize font system separately (required for text rendering)
        if pygame.font.get_init() is None:
            pygame.font.init()
        
        width, height = self.config.screen_size
        
        if self.config.render_mode == 'human':
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('OpenEnv')
        elif self.config.render_mode == 'rgb_array':
            self.screen = pygame.Surface((width, height))
        
        # Initialize clock for frame rate control (compatible with all Pygame versions)
        try:
            self.clock = pygame.time.Clock()
        except AttributeError:
            # Fallback for very old Pygame versions
            self.clock = None
        
        self.render_initialized = True
        
        if self.config.verbose:
            self.logger.info(f"Rendering initialized: mode={self.config.render_mode}")
    
    def _render_human(self) -> None:
        """Render environment for human viewing."""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Clear screen
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw environment elements
        self._draw_environment()
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate (if clock is available)
        if self.clock is not None:
            self.clock.tick(self.config.render_fps)
    
    def _render_rgb_array(self) -> np.ndarray:
        """
        Render environment as RGB array.
        
        Returns:
            RGB array of the rendered frame
        """
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw environment elements
        self._draw_environment()
        
        # Convert to RGB array
        surface = self.screen
        raw_data = pygame.image.tostring(surface, "RGB")
        width, height = self.config.screen_size
        img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        
        return img_array
    
    def _draw_environment(self) -> None:
        """Draw all environment elements on screen."""
        if self.screen is None:
            return
        
        width, height = self.config.screen_size
        
        # Coordinate transformation (center origin)
        scale = min(width, height) / (2 * self.config.boundary_limit)
        center_x, center_y = width // 2, height // 2
        
        def to_screen_coords(pos):
            x = center_x + pos[0] * scale
            y = center_y - pos[1] * scale  # Y is inverted in Pygame
            return (int(x), int(y))
        
        # Draw boundary circle
        boundary_radius = int(self.config.boundary_limit * scale)
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), 
                          boundary_radius, 2)
        
        # Draw target
        target_pos = to_screen_coords(self.target_position)
        pygame.draw.circle(self.screen, (0, 255, 0), target_pos, 10)
        
        # Draw agent
        agent_pos = to_screen_coords(self.position)
        pygame.draw.circle(self.screen, (255, 0, 0), agent_pos, 8)
        
        # Draw velocity vector
        velocity_end = to_screen_coords(self.position + self.velocity * 0.5)
        pygame.draw.line(self.screen, (0, 0, 255), agent_pos, velocity_end, 2)
        
        # Draw info text with error handling
        try:
            font = pygame.font.Font(None, 24)  # Default font, size 24
        except Exception:
            # Fallback: try to use a basic font
            try:
                font = pygame.font.SysFont('arial', 20)
            except Exception:
                # Last resort: skip text rendering
                font = None
        
        if font is not None:
            info_text = [
                f"Steps: {self.steps_taken}/{self.config.episode_length}",
                f"Return: {self.episode_return:.2f}",
                f"Velocity: {np.linalg.norm(self.velocity):.2f}",
            ]
            
            for i, text in enumerate(info_text):
                try:
                    text_surface = font.render(text, True, (0, 0, 0))
                    self.screen.blit(text_surface, (10, 10 + i * 20))
                except Exception as e:
                    # Skip individual text rendering errors
                    print(f"Text render error (non-fatal): {e}")

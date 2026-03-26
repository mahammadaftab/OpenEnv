"""
Basic OpenEnv Usage Example

This script demonstrates the fundamental API usage of OpenEnv
without any external RL libraries. Perfect for understanding the basics.

Usage:
    python examples/basic_usage.py
"""

import numpy as np
from openenv import OpenEnv, EnvConfig


def random_agent_example():
    """Run environment with random actions."""
    print("=" * 60)
    print("OpenEnv - Random Agent Example")
    print("=" * 60)
    
    # Create environment with default config
    config = EnvConfig(
        episode_length=200,
        verbose=True,
        render_mode=None,  # Set to 'human' to visualize
    )
    
    env = OpenEnv(config=config)
    
    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {observation.shape}")
    print(f"Initial observation: {observation}")
    
    # Run episode with random actions
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward
        total_reward += reward
        step_count += 1
        
        # Check if episode ended
        done = terminated or truncated
        
        if step_count % 50 == 0:
            print(f"Step {step_count}: reward={reward:.3f}, "
                  f"total_reward={total_reward:.3f}")
    
    print(f"\nEpisode Statistics:")
    print(f"  Total Steps: {step_count}")
    print(f"  Total Reward: {total_reward:.3f}")
    print(f"  Final Info: {info}")
    
    # Close environment
    env.close()
    
    return total_reward


def custom_config_example():
    """Demonstrate custom configuration options."""
    print("\n" + "=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)
    
    # Create custom configuration
    config = EnvConfig(
        episode_length=300,
        observation_dim=8,
        action_dim=4,
        gravity=9.81,
        friction=0.02,
        dt=0.02,
        reward_scale=1.5,
        sparse_rewards=False,
        max_velocity=80.0,
        boundary_limit=40.0,
        terminate_on_boundary=True,
        verbose=True,
        log_metrics=True,
        random_seed=123,
    )
    
    env = OpenEnv(config=config)
    
    # Reset and run a few steps
    obs, info = env.reset()
    
    print(f"\nEnvironment Configuration:")
    print(f"  Episode Length: {config.episode_length}")
    print(f"  Action Space: {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    
    # Take 10 random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, position={obs[0:2]}")
    
    env.close()


def state_inspection_example():
    """Demonstrate state inspection capabilities."""
    print("\n" + "=" * 60)
    print("State Inspection Example")
    print("=" * 60)
    
    config = EnvConfig(verbose=False)
    env = OpenEnv(config=config)
    
    env.reset(seed=42)
    
    # Get full internal state
    state = env.state()
    print(f"\nFull State Vector: {state}")
    print(f"State Shape: {state.shape if state is not None else None}")
    
    # Access individual components
    if state is not None:
        print("\nState Components:")
        print(f"  Position (x, y): {state[0:2]}")
        print(f"  Velocity (vx, vy): {state[2:4]}")
        print(f"  Target (tx, ty): {state[4:6]}")
        print(f"  Time Remaining: {state[6]:.3f}")
        print(f"  Distance to Target: {state[7]:.3f}")
    
    env.close()


def multiple_episodes_example():
    """Run multiple episodes and collect statistics."""
    print("\n" + "=" * 60)
    print("Multiple Episodes Example")
    print("=" * 60)
    
    config = EnvConfig(episode_length=100, verbose=False)
    env = OpenEnv(config=config)
    
    n_episodes = 5
    episode_returns = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        # Reset environment
        obs, info = env.reset(seed=ep)
        
        # Run episode
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Store statistics
        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Return={total_reward:.3f}, Steps={steps}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean Return: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min Return: {np.min(episode_returns):.3f}")
    print(f"  Max Return: {np.max(episode_returns):.3f}")
    
    env.close()
    
    return episode_returns, episode_lengths


def save_load_config_example():
    """Demonstrate saving and loading configuration."""
    print("\n" + "=" * 60)
    print("Configuration Save/Load Example")
    print("=" * 60)
    
    # Create and save configuration
    config = EnvConfig(
        episode_length=500,
        gravity=5.0,
        friction=0.05,
        verbose=True,
    )
    
    # Save to file
    config.save("env_config.json")
    print("Configuration saved to env_config.json")
    
    # Load from file
    loaded_config = EnvConfig.load("env_config.json")
    print("Configuration loaded successfully")
    
    # Verify loaded config
    print(f"\nLoaded Configuration:")
    print(f"  Episode Length: {loaded_config.episode_length}")
    print(f"  Gravity: {loaded_config.gravity}")
    print(f"  Friction: {loaded_config.friction}")
    print(f"  Verbose: {loaded_config.verbose}")
    
    # Clean up
    import os
    if os.path.exists("env_config.json"):
        os.remove("env_config.json")
        print("\nCleaned up temporary config file")


def main():
    """Run all examples."""
    print("\n🚀 OpenEnv Basic Usage Examples\n")
    
    # Example 1: Random agent
    random_agent_example()
    
    # Example 2: Custom configuration
    custom_config_example()
    
    # Example 3: State inspection
    state_inspection_example()
    
    # Example 4: Multiple episodes
    multiple_episodes_example()
    
    # Example 5: Save/load config
    save_load_config_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try train_openenv.py for RL training examples")
    print("  - Read README.md for detailed documentation")
    print("  - Explore tests/ for more usage patterns")


if __name__ == "__main__":
    main()

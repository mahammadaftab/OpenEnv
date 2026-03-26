"""
Example Training Script for OpenEnv using Stable Baselines3

This script demonstrates how to train an RL agent on OpenEnv using PPO.
It includes training, evaluation, and visualization components.

Usage:
    python examples/train_openenv.py --total_timesteps 100000
    
Requirements:
    pip install stable-baselines3 matplotlib
"""

import argparse
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from openenv import OpenEnv, EnvConfig


class TrainingCallback(BaseCallback):
    """
    Custom callback for logging during training.
    
    This callback prints progress updates and tracks metrics.
    """
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log every 1000 steps
        if self.n_calls % 1000 == 0 and self.verbose > 0:
            print(f"Step {self.n_calls:,} / {self.model.n_timesteps:,}")
        return True
    
    def _on_rollout_end(self) -> None:
        # Collect rollout statistics
        if len(self.model.ep_info_buffer) > 0:
            infos = list(self.model.ep_info_buffer)
            returns = [info['r'] for info in infos]
            lengths = [info['l'] for info in infos]
            
            self.episode_returns.extend(returns)
            self.episode_lengths.extend(lengths)
            
            if self.verbose > 0:
                print(f"Rollout complete - Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}, "
                      f"Mean Length: {np.mean(lengths):.1f}")


def make_env(env_config: EnvConfig, rank: int, seed: int = 0):
    """
    Environment factory function for vectorized environments.
    
    Args:
        env_config: Environment configuration
        rank: Environment index (for seeding)
        seed: Base random seed
        
    Returns:
        Callable that creates a monitored environment
    """
    def _init():
        env = OpenEnv(config=env_config)
        env.seed(seed + rank)
        env = Monitor(env)  # Track episode returns and lengths
        return env
    
    return _init


def create_environment(
    config: EnvConfig,
    n_envs: int = 1,
    parallel: bool = False,
    seed: int = 42,
) -> DummyVecEnv | SubprocVecEnv:
    """
    Create vectorized environment for training.
    
    Args:
        config: Environment configuration
        n_envs: Number of parallel environments
        parallel: Use multiprocessing (SubprocVecEnv)
        seed: Random seed
        
    Returns:
        Vectorized environment wrapper
    """
    if n_envs == 1:
        env = DummyVecEnv([make_env(config, 0, seed)])
    else:
        if parallel:
            env = SubprocVecEnv([make_env(config, i, seed) for i in range(n_envs)])
        else:
            env = DummyVecEnv([make_env(config, i, seed) for i in range(n_envs)])
    
    return env


def train_ppo(
    env_config: EnvConfig,
    total_timesteps: int = 100000,
    n_envs: int = 1,
    parallel_envs: bool = False,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    log_dir: str = "./logs",
    eval_freq: int = 10000,
    save_freq: int = 50000,
    verbose: int = 1,
) -> tuple[PPO, dict]:
    """
    Train a PPO agent on OpenEnv.
    
    Args:
        env_config: Environment configuration
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        parallel_envs: Use SubprocVecEnv instead of DummyVecEnv
        learning_rate: Learning rate for optimizer
        n_steps: Steps per rollout per environment
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of epochs when updating
        gamma: Discount factor
        gae_lambda: Factor for GAE advantage estimation
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        seed: Random seed
        log_dir: Directory for logs
        eval_freq: Evaluation frequency
        save_freq: Model saving frequency
        verbose: Verbosity level
        
    Returns:
        Trained model and training information dictionary
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = create_environment(env_config, n_envs, parallel_envs, seed)
    
    # Create callback for logging
    training_callback = TrainingCallback(verbose=verbose)
    
    # Create evaluation callback
    eval_env = create_environment(env_config, seed=seed + 1000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    
    # Initialize PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=log_dir,
        seed=seed,
        verbose=verbose,
    )
    
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Environment: {n_envs} parallel environment(s)")
    print(f"Model architecture: {model.policy}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[training_callback, eval_callback],
    )
    
    # Save final model
    model.save(os.path.join(log_dir, "ppo_openenv_final"))
    
    # Close environments
    env.close()
    eval_env.close()
    
    training_info = {
        'total_timesteps': total_timesteps,
        'episode_returns': training_callback.episode_returns,
        'episode_lengths': training_callback.episode_lengths,
    }
    
    print(f"Training complete! Model saved to {log_dir}")
    
    return model, training_info


def evaluate_agent(
    model: PPO,
    env_config: EnvConfig,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Evaluate trained agent.
    
    Args:
        model: Trained RL model
        env_config: Environment configuration
        n_eval_episodes: Number of episodes for evaluation
        deterministic: Use deterministic actions
        render: Render episodes
        seed: Random seed
        
    Returns:
        Mean reward and standard deviation
    """
    env_config.render_mode = 'human' if render else None
    env = OpenEnv(config=env_config)
    env.seed(seed)
    
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
    )
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Episodes: {n_eval_episodes}")
    
    env.close()
    
    return mean_reward, std_reward


def plot_training_results(
    training_info: dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training progress.
    
    Args:
        training_info: Dictionary with training data
        save_path: Path to save plot
        show: Display plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot episode returns
    returns = training_info['episode_returns']
    if len(returns) > 0:
        x_axis = range(len(returns))
        axes[0].plot(x_axis, returns, alpha=0.7, label='Episode Return')
        
        # Moving average
        window_size = min(10, len(returns) // 5)
        if window_size > 0:
            ma_returns = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
            ma_x = range(window_size - 1, len(returns))
            axes[0].plot(ma_x, ma_returns, 'r-', linewidth=2, label=f'{window_size}-ep MA')
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Return')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    lengths = training_info['episode_lengths']
    if len(lengths) > 0:
        x_axis = range(len(lengths))
        axes[1].plot(x_axis, lengths, alpha=0.7, color='green', label='Episode Length')
        
        # Moving average
        window_size = min(10, len(lengths) // 5)
        if window_size > 0:
            ma_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
            ma_x = range(window_size - 1, len(lengths))
            axes[1].plot(ma_x, ma_lengths, 'r-', linewidth=2, label=f'{window_size}-ep MA')
        
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Episode Duration')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train RL agent on OpenEnv')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                       help='Total training timesteps (default: 100000)')
    parser.add_argument('--n_envs', type=int, default=1,
                       help='Number of parallel environments (default: 1)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use multiprocessing for environments')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--log_dir', type=str, default='./logs/openenv',
                       help='Log directory (default: ./logs/openenv)')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency (default: 10000)')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Model saving frequency (default: 50000)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model after training')
    parser.add_argument('--render', action='store_true',
                       help='Render evaluation episodes')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Configure environment
    env_config = EnvConfig(
        episode_length=500,
        verbose=args.verbose > 0,
        log_metrics=True,
        random_seed=args.seed,
    )
    
    print("=" * 60)
    print("OpenEnv Training Script")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Total Timesteps: {args.total_timesteps:,}")
    print(f"  Parallel Environments: {args.n_envs}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Log Directory: {args.log_dir}")
    print("=" * 60)
    
    # Train agent
    model, training_info = train_ppo(
        env_config=env_config,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        parallel_envs=args.parallel,
        seed=args.seed,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        verbose=args.verbose,
    )
    
    # Evaluate agent
    if args.evaluate:
        print("\n" + "=" * 60)
        print("Evaluating Trained Agent")
        print("=" * 60)
        evaluate_agent(
            model=model,
            env_config=env_config,
            n_eval_episodes=10,
            deterministic=True,
            render=args.render,
            seed=args.seed,
        )
    
    # Plot results
    if args.plot:
        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)
        plot_training_results(
            training_info=training_info,
            save_path=os.path.join(args.log_dir, "training_results.png"),
            show=False,
        )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

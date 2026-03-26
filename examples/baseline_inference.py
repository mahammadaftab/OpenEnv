"""
Baseline Inference Script for OpenEnv

Runs reproducible evaluation with deterministic scoring across all difficulty levels.

Usage:
    python examples/baseline_inference.py --task_level medium --n_episodes 10
    python examples/baseline_inference.py --all_tasks
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv import OpenEnv, EnvConfig
from openenv.core.grader import create_grader


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not installed. Using default configuration.")
        return get_default_config()
    except FileNotFoundError:
        print(f"Warning: {yaml_path} not found. Using default configuration.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'tasks': {
            'easy': {
                'config': {
                    'episode_length': 300,
                    'boundary_limit': 80.0,
                    'max_velocity': 60.0,
                    'gravity': 5.0,
                    'friction': 0.02,
                    'obstacle_count': 0,
                    'wind_disturbance': False,
                    'sensor_noise': 0.0,
                },
                'grader': {
                    'success_threshold': 0.7,
                    'criteria': [
                        {'name': 'reached_target', 'weight': 0.6},
                        {'name': 'time_efficiency', 'weight': 0.2},
                        {'name': 'energy_efficiency', 'weight': 0.2},
                    ]
                }
            },
            'medium': {
                'config': {
                    'episode_length': 500,
                    'boundary_limit': 60.0,
                    'max_velocity': 50.0,
                    'gravity': 7.0,
                    'friction': 0.03,
                    'obstacle_count': 5,
                    'wind_disturbance': False,
                    'sensor_noise': 0.05,
                },
                'grader': {
                    'success_threshold': 0.75,
                    'criteria': [
                        {'name': 'reached_target', 'weight': 0.5},
                        {'name': 'collision_avoidance', 'weight': 0.25},
                        {'name': 'time_efficiency', 'weight': 0.15},
                        {'name': 'energy_efficiency', 'weight': 0.1},
                    ]
                }
            },
            'hard': {
                'config': {
                    'episode_length': 700,
                    'boundary_limit': 50.0,
                    'max_velocity': 40.0,
                    'gravity': 9.0,
                    'friction': 0.05,
                    'obstacle_count': 10,
                    'wind_disturbance': True,
                    'sensor_noise': 0.1,
                },
                'grader': {
                    'success_threshold': 0.8,
                    'criteria': [
                        {'name': 'reached_target', 'weight': 0.45},
                        {'name': 'collision_avoidance', 'weight': 0.25},
                        {'name': 'wind_compensation', 'weight': 0.15},
                        {'name': 'time_efficiency', 'weight': 0.1},
                        {'name': 'energy_efficiency', 'weight': 0.05},
                    ]
                }
            },
        }
    }


def run_episode(
    env: OpenEnv,
    grader,
    seed: int,
    render: bool = False,
) -> Dict[str, Any]:
    """
    Run single episode and collect metrics.
    
    Args:
        env: Environment instance
        grader: Task grader instance
        seed: Random seed
        render: Whether to render
        
    Returns:
        Episode results dictionary
    """
    # Reset environment and grader
    obs, info = env.reset(seed=seed)
    grader.reset()
    
    done = False
    total_reward = 0.0
    steps = 0
    
    prev_position = env.position.copy()
    optimal_distance = np.linalg.norm(env.target_position - env.position)
    grader.episode_data['optimal_distance'] = optimal_distance
    
    while not done:
        # Get action (random policy for baseline)
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update grader with metrics
        current_position = env.position.copy()
        distance_delta = np.linalg.norm(current_position - prev_position)
        
        grader.update(
            steps=1,
            distance_traveled=distance_delta,
            energy_consumed=np.sum(np.abs(action)) * 0.5,
        )
        
        # Check for collisions (if obstacles exist)
        if hasattr(env, 'check_collision') and env.check_collision():
            grader.update(collisions=1)
        
        # Track wind deviation
        if env.config.wind_disturbance and hasattr(env, 'wind_deviation'):
            grader.update(max_wind_deviation=max(
                grader.episode_data['max_wind_deviation'],
                env.wind_deviation
            ))
        
        # Update position
        prev_position = current_position.copy()
        
        # Accumulate reward
        total_reward += reward
        steps += 1
        
        # Render if requested
        if render:
            env.render()
        
        # Check termination
        done = terminated or truncated
    
    # Final updates to grader
    final_distance = np.linalg.norm(env.position - env.target_position)
    grader.update(
        target_reached=final_distance < getattr(env, 'target_radius', 5.0),
        final_distance_to_target=final_distance,
        time_to_complete=steps,
    )
    
    # Get grade report
    grade_report = grader.get_grade_report()
    
    # Compile results
    results = {
        'seed': seed,
        'steps': steps,
        'total_reward': total_reward,
        'final_score': grade_report['final_score'],
        'passed': grade_report['passed'],
        'criteria_scores': grade_report['criteria_scores'],
        'episode_data': grade_report['episode_data'],
        'feedback': grade_report['feedback'],
    }
    
    return results


def evaluate_task(
    task_level: str,
    config: Dict[str, Any],
    n_episodes: int = 10,
    seed: int = 42,
    render: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate agent on specific task level.
    
    Args:
        task_level: Difficulty level
        config: Task configuration
        n_episodes: Number of episodes
        seed: Base random seed
        render: Render episodes
        verbose: Print progress
        
    Returns:
        Aggregated evaluation results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating {task_level.upper()} task")
        print(f"{'='*60}")
        print(f"Configuration:")
        for key, value in config['config'].items():
            print(f"  {key}: {value}")
        print(f"Grading criteria:")
        for criterion in config['grader']['criteria']:
            print(f"  - {criterion['name']}: {criterion['weight']*100:.0f}%")
        print(f"{'='*60}\n")
    
    # Create environment
    env_config = EnvConfig(
        **config['config'],
        task_level=task_level,
        verbose=False,
    )
    env = OpenEnv(config=env_config)
    
    # Create grader
    grader = create_grader(task_level, config['grader'])
    
    # Run episodes
    episode_results = []
    for ep in range(n_episodes):
        episode_seed = seed + ep
        result = run_episode(env, grader, episode_seed, render=render)
        episode_results.append(result)
        
        if verbose:
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            print(f"Episode {ep+1}/{n_episodes} (seed={episode_seed}): "
                  f"Score={result['final_score']:.3f} {status}")
    
    env.close()
    
    # Aggregate results
    scores = [r['final_score'] for r in episode_results]
    rewards = [r['total_reward'] for r in episode_results]
    steps = [r['steps'] for r in episode_results]
    passed_count = sum(1 for r in episode_results if r['passed'])
    
    aggregated = {
        'task_level': task_level,
        'n_episodes': n_episodes,
        'base_seed': seed,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'min_score': float(np.min(scores)),
        'max_score': float(np.max(scores)),
        'pass_rate': passed_count / n_episodes,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_steps': float(np.mean(steps)),
        'episode_results': episode_results,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results Summary - {task_level.upper()}")
        print(f"{'='*60}")
        print(f"Mean Score: {aggregated['mean_score']:.3f} ± {aggregated['std_score']:.3f}")
        print(f"Score Range: [{aggregated['min_score']:.3f}, {aggregated['max_score']:.3f}]")
        print(f"Pass Rate: {aggregated['pass_rate']*100:.1f}% ({passed_count}/{n_episodes})")
        print(f"Mean Reward: {aggregated['mean_reward']:.2f} ± {aggregated['std_reward']:.2f}")
        print(f"Mean Steps: {aggregated['mean_steps']:.1f}")
        print(f"{'='*60}\n")
    
    return aggregated


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description='Baseline Inference for OpenEnv')
    parser.add_argument('--task_level', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'],
                       help='Task difficulty level')
    parser.add_argument('--all_tasks', action='store_true',
                       help='Evaluate on all difficulty levels')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    parser.add_argument('--config', type=str, default='openenv.yaml',
                       help='Path to configuration file')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OpenEnv Baseline Inference")
    print("="*60)
    
    # Load configuration
    yaml_config = load_config_from_yaml(args.config)
    
    # Determine which tasks to evaluate
    if args.all_tasks:
        task_levels = ['easy', 'medium', 'hard']
    else:
        task_levels = [args.task_level]
    
    all_results = {}
    
    # Evaluate each task level
    for task_level in task_levels:
        task_config = yaml_config['tasks'][task_level]
        results = evaluate_task(
            task_level=task_level,
            config=task_config,
            n_episodes=args.n_episodes,
            seed=args.seed,
            render=args.render,
            verbose=not args.quiet,
        )
        all_results[task_level] = results
    
    # Save results
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print overall summary
    if len(task_levels) > 1:
        print("\n" + "="*60)
        print("Overall Performance Summary")
        print("="*60)
        for task_level in task_levels:
            results = all_results[task_level]
            print(f"{task_level.upper():10s}: Score={results['mean_score']:.3f} ± "
                  f"{results['std_score']:.3f}, Pass Rate={results['pass_rate']*100:.1f}%")
        print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()

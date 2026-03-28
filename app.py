"""
OpenEnv Drone Navigation - Hugging Face Spaces Demo

Interactive web interface for testing the drone navigation environment.

Features:
- Select difficulty level (easy/medium/hard)
- Watch agent attempt navigation task
- View real-time metrics and scoring
- Compare performance across difficulty levels
"""

import gradio as gr
import numpy as np
from pathlib import Path
import json

from openenv import OpenEnv, EnvConfig
from openenv.core.grader import create_grader


# Load configuration
CONFIG_PATH = Path(__file__).parent / "openenv.yaml"

def load_yaml_config():
    """Load configuration from YAML."""
    try:
        import yaml
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None


def get_task_config(task_level: str) -> dict:
    """Get configuration for specific task level."""
    yaml_config = load_yaml_config()
    
    if yaml_config and 'tasks' in yaml_config:
        return yaml_config['tasks'][task_level]
    
    # Fallback defaults
    defaults = {
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
    return defaults.get(task_level, defaults['medium'])


def run_demo_episode(
    task_level: str,
    seed: int = 42,
    render_mode: str = "rgb_array",
):
    """
    Run single demo episode and return results.
    
    Args:
        task_level: Difficulty level
        seed: Random seed
        render_mode: Rendering mode
        
    Returns:
        Tuple of (screenshot, metrics_text, grade_text)
    """
    # Get configuration
    task_config = get_task_config(task_level)
    
    # Create environment
    env_config = EnvConfig(
        **task_config['config'],
        task_level=task_level,
        render_mode=render_mode,
        verbose=False,
    )
    
    try:
        env = OpenEnv(config=env_config)
    except Exception as e:
        import traceback
        error_msg = f"Failed to create environment: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        # Return placeholder image and error message
        placeholder = np.zeros((768, 1024, 3), dtype=np.uint8)
        return placeholder, "Error initializing environment", error_msg
    
    # Create grader
    grader = create_grader(task_level, task_config['grader'])
    
    # Reset
    obs, info = env.reset(seed=seed)
    grader.reset()
    
    # Run episode
    frames = []
    total_reward = 0.0
    steps = 0
    max_steps = 200  # Limit for demo
    
    prev_position = env.position.copy()
    optimal_distance = np.linalg.norm(env.target_position - env.position)
    grader.episode_data['optimal_distance'] = optimal_distance
    
    for step in range(max_steps):
        # Random action for demo (in real use, this would be your agent)
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update grader
        current_position = env.position.copy()
        distance_delta = np.linalg.norm(current_position - prev_position)
        
        grader.update(
            steps=1,
            distance_traveled=distance_delta,
            energy_consumed=np.sum(np.abs(action)) * 0.5,
        )
        
        # Check collisions
        if hasattr(env, 'check_collision') and env.check_collision():
            grader.update(collisions=1)
        
        # Track wind deviation
        if env.config.wind_disturbance and hasattr(env, 'wind_deviation'):
            grader.update(max_wind_deviation=max(
                grader.episode_data['max_wind_deviation'],
                env.wind_deviation
            ))
        
        prev_position = current_position.copy()
        total_reward += reward
        steps += 1
        
        # Render frame
        if render_mode == "rgb_array":
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                print(f"Rendering error (non-fatal): {e}")
                # Continue without rendering
                pass
        
        # Check termination
        if terminated or truncated:
            break
    
    # Final updates
    final_distance = np.linalg.norm(env.position - env.target_position)
    target_radius = getattr(env, 'target_radius', 5.0)
    
    grader.update(
        target_reached=final_distance < target_radius,
        final_distance_to_target=final_distance,
        time_to_complete=steps,
    )
    
    # Get grade report
    grade_report = grader.get_grade_report()
    
    # Generate metrics text
    metrics_text = f"""
**Episode Statistics:**
- Steps: {steps}
- Total Reward: {total_reward:.2f}
- Final Distance: {final_distance:.2f}
- Target Reached: {'Yes ✓' if grade_report['episode_data']['target_reached'] else 'No ✗'}
- Collisions: {grade_report['episode_data']['collisions']}
    """.strip()
    
    # Generate grade text
    grade_text = f"""
**Performance Grade: {grade_report['final_score']:.2f} / 1.00**

{grade_report['feedback']}

**Criteria Scores:**
    """
    
    for criterion_name, score in grade_report['criteria_scores'].items():
        grade_text += f"\n- {criterion_name.replace('_', ' ').title()}: {score:.2f}"
    
    grade_text += f"\n\n**Status:** {'✓ PASSED' if grade_report['passed'] else '✗ FAILED'}"
    grade_text += f"\nThreshold: {grade_report['success_threshold']:.2f}"
    
    env.close()
    
    # Return last frame (or create composite if multiple frames)
    if len(frames) > 0:
        # Use middle frame as representative
        screenshot = frames[len(frames) // 2]
    else:
        # Create placeholder
        screenshot = np.zeros((768, 1024, 3), dtype=np.uint8)
    
    return screenshot, metrics_text, grade_text


def compare_all_levels(seed: int = 42):
    """
    Run comparison across all difficulty levels.
    
    Args:
        seed: Random seed
        
    Returns:
        Comparison table text
    """
    results = []
    
    for level in ['easy', 'medium', 'hard']:
        task_config = get_task_config(level)
        
        env_config = EnvConfig(
            **task_config['config'],
            task_level=level,
            verbose=False,
        )
        
        env = OpenEnv(config=env_config)
        grader_instance = create_grader(level, task_config['grader'])
        
        obs, _ = env.reset(seed=seed)
        grader_instance.reset()
        
        # Run episode
        done = False
        steps = 0
        while not done and steps < 300:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            grader_instance.update(steps=1)
            done = terminated or truncated
            steps += 1
        
        # Final evaluation
        final_distance = np.linalg.norm(env.position - env.target_position)
        grader_instance.update(
            target_reached=final_distance < 5.0,
            final_distance_to_target=final_distance,
        )
        
        grade_report = grader_instance.get_grade_report()
        
        results.append({
            'level': level.upper(),
            'score': grade_report['final_score'],
            'passed': '✓' if grade_report['passed'] else '✗',
            'steps': steps,
        })
        
        env.close()
    
    # Create comparison table
    table = "| Difficulty | Score | Status | Steps |\n"
    table += "|------------|-------|--------|-------|\n"
    
    for result in results:
        table += f"| {result['level']:10s} | {result['score']:.2f} | {result['passed']:6s} | {result['steps']:5d} |\n"
    
    return table


def create_demo():
    """Create Gradio interface."""
    
    with gr.Blocks(title="OpenEnv Drone Navigation") as demo:
        gr.Markdown("""
        # 🚁 OpenEnv: Autonomous Drone Navigation
        
        **Real-world RL environment for warehouse inventory inspection**
        
        Test our AI agent's ability to navigate drones through warehouses for automated inventory inspection.
        Choose a difficulty level and watch the agent attempt to reach the target while avoiding obstacles!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🎮 Control Panel")
                
                task_level_dropdown = gr.Dropdown(
                    choices=['easy', 'medium', 'hard'],
                    value='medium',
                    label="Difficulty Level",
                    info="Select task difficulty"
                )
                
                seed_slider = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=42,
                    step=1,
                    label="Random Seed",
                    info="For reproducible runs"
                )
                
                run_button = gr.Button("🚀 Run Episode", variant="primary")
                
                compare_button = gr.Button("📊 Compare All Levels")
            
            with gr.Column(scale=3):
                gr.Markdown("### 📺 Environment View")
                
                output_image = gr.Image(
                    label="Drone Navigation",
                    type="numpy",
                    height=500,
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Metrics")
                metrics_output = gr.Textbox(
                    label="Episode Statistics",
                    lines=8,
                )
            
            with gr.Column():
                gr.Markdown("### 🎯 Performance Grade")
                grade_output = gr.Textbox(
                    label="Grade Report",
                    lines=10,
                )
        
        with gr.Row():
            gr.Markdown("### 📋 Level Comparison")
            comparison_output = gr.Textbox(
                label="Performance Across Difficulty Levels",
                lines=8,
            )
        
        # Event handlers
        run_button.click(
            fn=run_demo_episode,
            inputs=[task_level_dropdown, seed_slider],
            outputs=[output_image, metrics_output, grade_output],
        )
        
        compare_button.click(
            fn=compare_all_levels,
            inputs=[seed_slider],
            outputs=[comparison_output],
        )
        
        # Auto-run on load
        demo.load(
            fn=run_demo_episode,
            inputs=[task_level_dropdown, seed_slider],
            outputs=[output_image, metrics_output, grade_output],
        )
        
        gr.Markdown("""
        ---
        **About:** This is a production-ready RL environment for training autonomous drones.
        
        **Task:** Navigate to the green target while managing velocity and avoiding obstacles.
        
        **Scoring:** Agents are graded on target acquisition, collision avoidance, time efficiency, and energy management.
        
        [View on GitHub](https://github.com/yourusername/OpenEnv) | [Documentation](https://github.com/yourusername/OpenEnv#readme)
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())

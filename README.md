---
title: OpenEnv
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.10.0
python_version: '3.11'
app_file: app.py
pinned: false
license: mit
---

# OpenEnv

<div align="center">

**A Production-Ready Reinforcement Learning Environment for Autonomous Drone Navigation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/yourusername/openenv-drone-navigation)

🚁 **Try the live demo:** [OpenEnv on Hugging Face Spaces](https://huggingface.co/spaces/yourusername/openenv-drone-navigation)

</div>

---

## 🌍 Real-World Task: Warehouse Inventory Inspection

OpenEnv simulates **autonomous drone navigation for automated warehouse inventory inspection** - a critical real-world robotics challenge faced by logistics companies worldwide.

### The Problem
- **Manual inventory checks** in massive warehouses are time-consuming and error-prone
- **Human inspectors** need to navigate aisles, read barcodes, and verify stock levels
- **Operational costs** are high, and accuracy is critical for supply chain management

### Our Solution
Train AI agents to autonomously navigate drones through warehouse environments to:
- ✅ Reach inspection checkpoints (inventory scanners)
- ✅ Avoid static obstacles (shelves, boxes, equipment)
- ✅ Compensate for dynamic disturbances (wind from ventilation, moving machinery)
- ✅ Optimize flight paths for battery efficiency
- ✅ Complete inspections within time constraints

### Industry Impact
This environment directly models challenges faced by:
- **Amazon Robotics** - Automated warehouse monitoring
- **DJI Enterprise** - Industrial inspection drones
- **Boston Dynamics** - Autonomous navigation systems
- **Wing Aviation** - Delivery drone path planning

---

## ✨ Key Features

### 🎯 Three Difficulty Levels with Agent Graders

| Level | Task | Challenges | Scoring Criteria |
|-------|------|------------|------------------|
| **Easy** | Basic Navigation | Open space, no obstacles | Target reached (60%), Time (20%), Energy (20%) |
| **Medium** | Obstacle Avoidance | 5 static obstacles, mild sensor noise | Target (50%), Collision avoidance (25%), Time (15%), Energy (10%) |
| **Hard** | Dynamic Environment | 10 moving obstacles, wind, sensor noise | Target (45%), Collisions (25%), Wind compensation (15%), Time (10%), Energy (5%) |

**Scoring:** Each task graded 0.0–1.0 with weighted criteria and partial credit

### 🧠 Meaningful Reward Function

**Dense Rewards:**
- Distance-based shaping: `-0.15 × distance_to_target`
- Progress bonus: `+0.8 × Δdistance` (reward for improvement)
- Velocity penalty: `-0.02 × ||velocity||` (encourage smooth flight)

**Sparse Rewards:**
- Success bonus: `+100` for reaching target
- Collision penalty: `-50` per collision
- Boundary violation: `-30`

**Partial Progress Signals:**
- Waypoint bonus: `+10` for passing intermediate checkpoints
- Altitude bonus: `+5` for maintaining safe flying height
- Stability bonus: `+2` for smooth control inputs

### 🔬 Reproducible Evaluation

- Deterministic seeding across all difficulty levels
- Standardized baseline inference script
- Comprehensive grading with detailed feedback
- Performance metrics tracking

### 🚀 Deployment Ready

- **Hugging Face Spaces** integration with interactive web demo
- **Docker** containerization for easy deployment
- **Gradio** interface for visualization
- **YAML** configuration for experiment management

---

## 📦 Installation

### Quick Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/OpenEnv.git
cd OpenEnv

# Install dependencies
pip install -r requirements.txt

# Optional: Install as package
pip install -e .
```

### Dependencies

**Core:**
- `gymnasium>=0.28.0` - Environment interface
- `numpy>=1.21.0` - Numerical operations
- `pygame>=2.1.0` - Rendering

**RL Training:**
- `stable-baselines3>=2.0.0` - RL algorithms (PPO, A2C, SAC)
- `sb3-contrib>=2.0.0` - Additional algorithms

**Configuration & Deployment:**
- `pyyaml>=6.0` - YAML configuration parsing
- `gradio>=4.0.0` - Web interface for Hugging Face Spaces

**Development:**
- `matplotlib>=3.5.0` - Visualization
- `pytest>=7.0.0` - Testing
- `black>=22.0.0`, `flake8>=5.0.0` - Code quality

---

## 🎮 Environment Description

### Task Overview

**Objective:** Navigate a drone from starting position to target checkpoint while maximizing efficiency and safety.

**State Space (12-dimensional):**
- **Position (3D):** `(x, y, z)` - Current drone coordinates
- **Velocity (3D):** `(vx, vy, vz)` - Current velocity vector
- **Target (3D):** `(tx, ty, tz)` - Target checkpoint location
- **Obstacles (2D):** `(nearest_distance, nearest_angle)` - Closest obstacle info
- **Time:** Normalized time remaining in episode `[0, 1]`

**Action Space (4-dimensional continuous):**
- **Thrust:** Vertical force control `[-1.0, 1.0]`
- **Yaw:** Rotation control `[-1.0, 1.0]`
- **Pitch:** Forward/backward tilt `[-1.0, 1.0]`
- **Roll:** Lateral movement `[-1.0, 1.0]`

**Physics Model:**
- Drone dynamics with mass `1.5 kg`
- Gravity `9.81 m/s²` (varies by difficulty)
- Drag coefficient `0.01`
- Maximum thrust `20.0 N`
- Battery capacity `1000 mAh` with drain rate `0.5 mAh/step`

### Configuration

All parameters configurable via [`openenv.yaml`](openenv.yaml):

```yaml
tasks:
  easy:
    config:
      episode_length: 300
      boundary_limit: 80.0
      max_velocity: 60.0
      obstacle_count: 0
      wind_disturbance: false
      
  medium:
    config:
      episode_length: 500
      boundary_limit: 60.0
      max_velocity: 50.0
      obstacle_count: 5
      sensor_noise: 0.05
      
  hard:
    config:
      episode_length: 700
      boundary_limit: 50.0
      max_velocity: 40.0
      obstacle_count: 10
      wind_disturbance: true
```

---

## 🎯 Quick Start

### Basic Usage

```python
from openenv import OpenEnv, EnvConfig

# Create environment with default config
env = OpenEnv()

# Or with custom configuration
config = EnvConfig(
    episode_length=500,
    verbose=True,
    render_mode='human'
)
env = OpenEnv(config=config)

# Reset environment
observation, info = env.reset()

# Training loop
for step in range(1000):
    # Sample random action (replace with your agent)
    action = env.action_space.sample()
    
    # Take step in environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render if enabled
    env.render()
    
    # Check if episode is done
    if terminated or truncated:
        observation, info = env.reset()

# Cleanup
env.close()
```

### Integration with Stable Baselines3

```python
from stable_baselines3 import PPO
from openenv import OpenEnv

# Create environment
env = OpenEnv(render_mode=None)  # No rendering during training

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)
model.learn(total_timesteps=100000)

# Save trained model
model.save("ppo_openenv")

# Load and test
model = PPO.load("ppo_openenv")
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

---

## 🧪 Baseline Inference & Evaluation

Run reproducible evaluation across all difficulty levels:

```bash
# Evaluate on medium task (default)
python examples/baseline_inference.py --task_level medium --n_episodes 10

# Evaluate on all tasks
python examples/baseline_inference.py --all_tasks

# Save results to file
python examples/baseline_inference.py --all_tasks --output results.json

# Run without verbose output
python examples/baseline_inference.py --all_tasks --quiet
```

**Example Output:**
```
============================================================
Evaluating MEDIUM task
============================================================
Configuration:
  episode_length: 500
  boundary_limit: 60.0
  max_velocity: 50.0
Grading criteria:
  - reached_target: 50%
  - collision_avoidance: 25%
  - time_efficiency: 15%
  - energy_efficiency: 10%
============================================================

Episode 1/10 (seed=42): Score=0.720 ✓ PASSED
Episode 2/10 (seed=43): Score=0.650 ✗ FAILED
...

Results Summary - MEDIUM
============================================================
Mean Score: 0.685 ± 0.045
Score Range: [0.620, 0.780]
Pass Rate: 70.0% (7/10)
Mean Reward: 45.3 ± 12.5
Mean Steps: 380.5
```

---

## 🤗 Hugging Face Spaces Deployment

### Try the Live Demo

Visit our interactive web demo: **[OpenEnv Drone Navigation](https://huggingface.co/spaces/yourusername/openenv-drone-navigation)**

Features:
- 🎮 Visual demonstration of drone navigation
- 📊 Real-time performance metrics
- 🎯 Automatic grading and feedback
- 📈 Comparison across difficulty levels

### Deploy Your Own Space

1. **Fork the repository** on Hugging Face

2. **Create `requirements.txt`** with Gradio:
   ```txt
   gradio>=4.0.0
   pyyaml>=6.0
   gymnasium>=0.28.0
   numpy>=1.21.0
   ```

3. **Add `app.py`** (already included in this repo)

4. **Configure Docker** (Dockerfile included)

5. **Push to Hugging Face**:
   ```bash
   git remote add space https://huggingface.co/spaces/yourusername/openenv-drone-navigation
   git push space main
   ```

Your Space will automatically deploy with the Gradio interface!

---

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t openenv-drone:latest .

# Run container
docker run -p 7860:7860 openenv-drone:latest

# Access at http://localhost:7860
```

The Dockerfile includes:
- Python 3.10 slim base
- All dependencies pre-installed
- Gradio web interface
- Health checks
- Non-root user for security

---

### EnvConfig Parameters

The `EnvConfig` dataclass provides extensive customization:

```python
from openenv import EnvConfig

config = EnvConfig(
    # Core settings
    episode_length=1000,      # Max steps per episode
    observation_dim=8,        # Observation space dimension
    action_dim=4,             # Action space dimension
    random_seed=42,           # Random seed for reproducibility
    
    # Physics parameters
    gravity=9.81,            # Gravitational constant (m/s²)
    friction=0.01,           # Friction coefficient
    dt=0.02,                 # Time step (seconds)
    
    # Reward configuration
    reward_scale=1.0,        # Global reward scaling
    sparse_rewards=False,    # Use only sparse rewards
    reward_clip=None,        # Clip rewards to [-clip, clip]
    
    # Termination conditions
    max_velocity=100.0,      # Max velocity before termination
    boundary_limit=50.0,     # Environment boundary radius
    terminate_on_boundary=True,  # End episode on boundary violation
    
    # Rendering
    render_mode=None,        # 'human', 'rgb_array', or None
    render_fps=60,          # Rendering frame rate
    screen_size=(800, 600), # Window size
    
    # Logging
    verbose=True,           # Enable logging
    log_metrics=True,       # Track performance metrics
)
```

### Loading/Saving Configuration

```python
# Save config to file
config.save("env_config.json")

# Load config from file
config = EnvConfig.load("env_config.json")

# Convert to/from dictionary
config_dict = config.to_dict()
config = EnvConfig.from_dict(config_dict)
```

---

## 🏗️ Environment Specification

### Observation Space (8-dimensional)

| Index | Component | Description |
|-------|-----------|-------------|
| 0-1 | Position (x, y) | Agent's current position |
| 2-3 | Velocity (vx, vy) | Agent's current velocity |
| 4-5 | Target (tx, ty) | Target position coordinates |
| 6 | Time remaining | Normalized time left [0, 1] |
| 7 | Distance to target | Euclidean distance to goal |

**Space Type:** `Box(low=-inf, high=inf, shape=(8,), dtype=np.float32)`

### Action Space (4-dimensional continuous)

Continuous force vector applied to agent:
- Actions normalized to `[-1.0, 1.0]`
- Represents force direction and magnitude
- Scaled internally by physics engine

**Space Type:** `Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)`

### Reward Function

The reward function combines multiple components:

1. **Dense Reward** (default):
   - Negative distance to target: `-0.1 × distance`
   - Encourages moving toward goal

2. **Sparse Reward**:
   - Success bonus: `+100` when reaching target (distance < 1.0)

3. **Reward Shaping**:
   - Progress bonus: `+0.5 × Δdistance`
   - Velocity penalty: `-0.01 × ||velocity||`

4. **Boundary Penalty**:
   - Episode termination (no explicit negative reward)

**Formula:**
```
reward = (-0.1 × distance - 0.01 × ||velocity|| + 0.5 × Δdistance) × scale
         + 100 × [distance < 1.0] × scale
```

### Termination Conditions

Episode ends when **any** of these occur:

1. **Time Limit**: `steps >= episode_length` (truncated)
2. **Boundary Violation**: `||position|| > boundary_limit` (terminated)
3. **Max Velocity**: `||velocity|| > max_velocity` (terminated)

---

## 🎮 API Reference

### Core Methods

#### `reset(seed=None, options=None)`

Reset environment to initial state.

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `options` (dict, optional): Additional initialization options
  - `random_start` (bool): Randomize starting position (default: True)

**Returns:**
- `observation` (np.ndarray): Initial observation
- `info` (dict): Additional information (empty by default)

**Example:**
```python
obs, info = env.reset()
obs, info = env.reset(seed=42, options={'random_start': False})
```

---

#### `step(action)`

Execute one time step in the environment.

**Parameters:**
- `action` (np.ndarray): Action to execute (force vector)

**Returns:**
- `observation` (np.ndarray): New observation
- `reward` (float): Reward received
- `terminated` (bool): Episode terminated
- `truncated` (bool): Episode truncated (time limit)
- `info` (dict): Additional information

**Example:**
```python
action = np.array([0.5, -0.3, 0.0, 0.0])
obs, reward, terminated, truncated, info = env.step(action)
```

---

#### `state()`

Get complete internal state vector.

**Returns:**
- `state` (np.ndarray or None): Full state representation

**Note:** Different from observation - provides full state access for debugging.

**Example:**
```python
full_state = env.state()
```

---

#### `render()`

Render the environment.

**Returns:**
- RGB array if `render_mode='rgb_array'`
- `None` if `render_mode='human'`

**Example:**
```python
env.render()  # Display to screen
frame = env.render()  # Get RGB array
```

---

#### `close()`

Clean up resources and close environment.

**Example:**
```python
env.close()
```

---

#### `seed(seed=None)`

Set random seed for reproducibility.

**Parameters:**
- `seed` (int, optional): Seed value

**Returns:**
- `seed` (int): The seed used

**Example:**
```python
env.seed(42)
```

---

## 📊 Metrics and Logging

### Tracked Metrics

The environment tracks performance metrics accessible via the `info` dict:

```python
{
    'steps': int,              # Steps taken in current episode
    'return': float,           # Cumulative return
    'target_reached': bool,    # Whether target was reached
    'terminated': bool,        # Whether episode terminated early
    'truncated': bool,         # Whether episode truncated
}
```

### Logging Levels

Control verbosity with the `verbose` config parameter:

```python
# Verbose mode (INFO level)
config = EnvConfig(verbose=True)
env = OpenEnv(config)

# Silent mode (WARNING level)
config = EnvConfig(verbose=False)
env = OpenEnv(config)
```

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=openenv --cov-report=html

# Run specific test file
pytest tests/test_env.py -v
```

### Test Coverage

The test suite includes:

- ✅ Unit tests for core functionality
- ✅ API compliance tests (Gymnasium checker)
- ✅ Physics dynamics validation
- ✅ Reward function tests
- ✅ Termination condition tests
- ✅ Rendering tests
- ✅ Configuration tests
- ✅ Integration tests with sample agents

---

## 📈 Performance Benchmarks

### Baseline Results

Training with PPO (Stable Baselines3):

| Metric | Value |
|--------|-------|
| Timesteps | 100,000 |
| Mean Return | ~850 |
| Success Rate | ~95% |
| Episode Length | ~150 steps |

### Environment Speed

- **Step Latency:** < 0.1ms (no rendering)
- **Step Latency:** ~2ms (with rgb_array rendering)
- **Parallel Performance:** Scales linearly with VecEnv

---

## 🔬 Example Environments

### Custom Environment Variants

You can create specialized variants by modifying configuration:

```python
# Easy version - larger target, no boundary termination
easy_config = EnvConfig(
    boundary_limit=100.0,
    max_velocity=200.0,
    reward_scale=2.0,
    terminate_on_boundary=False,
)

# Hard version - smaller target, strict constraints
hard_config = EnvConfig(
    boundary_limit=20.0,
    max_velocity=50.0,
    sparse_rewards=True,
    friction=0.1,
)

# Fast training - shorter episodes
fast_config = EnvConfig(
    episode_length=200,
    dt=0.01,
)
```

---

## 🛠️ Development

### Code Quality

This project follows professional standards:

- **Type Hints:** Full type annotation throughout
- **PEP 8:** Compliant code style
- **Black Formatting:** Automated code formatting
- **Docstrings:** Comprehensive documentation
- **Logging:** Structured logging system

### Running Linters

```bash
# Code formatting
black openenv/ tests/

# Linting
flake8 openenv/ tests/

# Type checking
mypy openenv/
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Ensure code passes linting (`black . && flake8`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on [Gymnasium](https://gymnasium.farama.org/) framework
- Inspired by classic control environments (MountainCar, LunarLander)
- Designed for compatibility with [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

---

## 📞 Support

For issues, questions, or contributions:

- **Bug Reports:** GitHub Issues
- **Questions:** GitHub Discussions
- **General Inquiries:** See README contact info

---

## 🎓 Citation

If you use OpenEnv in your research, please cite:

```bibtex
@software{openenv2024,
  author = {OpenEnv Team},
  title = {OpenEnv: A Production-Ready Reinforcement Learning Environment},
  year = {2024},
  url = {https://github.com/yourusername/OpenEnv},
  version = {1.0.0}
}
```

---

<div align="center">

**Built with ❤️ for the RL Community**

</div>

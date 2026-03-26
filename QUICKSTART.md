# Quick Start Guide - OpenEnv

Get up and running with OpenEnv in minutes!

## 📦 Installation (5 minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/OpenEnv.git
cd OpenEnv
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Package (Optional)
```bash
pip install -e .
```

## 🚀 Your First Environment (2 minutes)

### Minimal Example
```python
from openenv import OpenEnv

# Create environment
env = OpenEnv()

# Reset
obs, info = env.reset()

# Take random actions
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**That's it!** You've just run your first RL environment.

## 🎮 Try the Examples

### Basic Usage Demo
```bash
python examples/basic_usage.py
```

This runs through all basic features:
- Random agent
- Custom configuration
- State inspection
- Multiple episodes
- Config save/load

### Training with PPO
```bash
python examples/train_openenv.py --total_timesteps 50000
```

Watch the agent learn to navigate to the target!

## ⚙️ Common Configurations

### Easy Mode (Beginner-Friendly)
```python
from openenv import EnvConfig, OpenEnv

config = EnvConfig(
    episode_length=300,      # Shorter episodes
    boundary_limit=100.0,    # Larger play area
    max_velocity=150.0,      # More forgiving
    verbose=True,
)

env = OpenEnv(config=config)
```

### Hard Mode (Challenge)
```python
config = EnvConfig(
    episode_length=200,      # Shorter time
    boundary_limit=20.0,     # Smaller area
    max_velocity=30.0,       # Strict limits
    sparse_rewards=True,     # Only goal reward
    friction=0.1,           # More drag
)

env = OpenEnv(config=config)
```

### Visual Mode (Watch It Run)
```python
config = EnvConfig(
    render_mode='human',     # Show window
    render_fps=60,
    screen_size=(800, 600),
)

env = OpenEnv(config=config)

# In your loop
env.render()  # Shows the environment
```

## 🏋️ Train Your First Agent (10 minutes)

### Using Stable Baselines3

```python
from stable_baselines3 import PPO
from openenv import OpenEnv

# Create environment
env = OpenEnv(render_mode=None)

# Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train for 50,000 steps
model.learn(total_timesteps=50000)

# Save model
model.save("my_first_agent")

print("Training complete!")
```

### Load and Test
```python
from stable_baselines3 import PPO
from openenv import OpenEnv

# Load trained model
model = PPO.load("my_first_agent")

# Create environment for testing
env = OpenEnv(render_mode='human')

# Run trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

env.close()
```

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:** Make sure you're in the OpenEnv directory and installed dependencies:
```bash
cd OpenEnv
pip install -r requirements.txt
```

### Issue: "No module named 'openenv'"
**Solution:** Install the package in development mode:
```bash
pip install -e .
```

### Issue: Pygame errors on Windows
**Solution:** Reinstall pygame:
```bash
pip uninstall pygame
pip install pygame --no-cache-dir
```

### Issue: Slow performance
**Solution:** Disable rendering during training:
```python
env = OpenEnv(render_mode=None)  # No rendering
```

## 📚 What's Next?

Now that you have the basics:

1. **Read the full documentation** - See README.md for complete API reference
2. **Explore examples/** - More complex use cases and patterns
3. **Run the tests** - `pytest tests/` to verify everything works
4. **Start your project** - Apply OpenEnv to your RL research!

## 💡 Pro Tips

### Tip 1: Use Vectorized Environments
Train faster with parallel environments:
```python
from stable_baselines3.common.vec_env import DummyVecEnv
from openenv import OpenEnv, EnvConfig

config = EnvConfig()
env = DummyVecEnv([lambda: OpenEnv(config) for _ in range(4)])
```

### Tip 2: Monitor Training
Use TensorBoard for visualization:
```bash
tensorboard --logdir=./logs/openenv
```

### Tip 3: Reproducibility
Always set seeds for reproducible results:
```python
env = OpenEnv()
env.seed(42)
obs, _ = env.reset(seed=42)
```

## 🤝 Need Help?

- **Documentation:** README.md (full API reference)
- **Examples:** examples/ directory
- **Tests:** tests/ for usage patterns
- **Issues:** GitHub Issues for bugs

---

**Congratulations!** You're ready to start training RL agents with OpenEnv! 🎉

Happy learning! 🚀

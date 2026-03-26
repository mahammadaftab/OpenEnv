# 🚀 How to Run OpenEnv - Step-by-Step Guide

## ⚡ Quick Start (Choose Your Path)

### Option 1: Test Installation First (Recommended)
```bash
# Just test if it works without installing everything
python -c "from openenv import OpenEnv; env = OpenEnv(); print('✅ Works!')"
```

### Option 2: Full Setup with Training
Follow the complete guide below.

---

## 📦 Step 1: Install Dependencies

### A. Basic Installation (Minimum Requirements)
```bash
cd c:\Users\mdaft\OneDrive\Desktop\OpenEnv
pip install gymnasium numpy pygame
```

### B. Full Installation (Recommended for RL Training)
```bash
pip install -r requirements.txt
```

This installs:
- Core: `gymnasium`, `numpy`, `pygame`
- RL: `stable-baselines3`, `sb3-contrib`
- Config: `pyyaml`
- Web: `gradio` (for Hugging Face demo)
- Testing: `pytest`

### C. If You Encounter Errors

**Pygame installation issue on Windows:**
```bash
pip install pygame --no-cache-dir
```

**Permission issues:**
```bash
pip install --user -r requirements.txt
```

---

## 🧪 Step 2: Verify Installation

Run this simple test:
```bash
python -c "from openenv import OpenEnv, EnvConfig; env = OpenEnv(); obs, info = env.reset(); print(f'Observation shape: {obs.shape}'); print('✅ Installation successful!')"
```

Expected output:
```
Observation shape: (12,)
✅ Installation successful!
```

---

## 🎮 Step 3: Run Examples

### Example 1: Basic Usage (No RL Agent)
```bash
python examples/basic_usage.py
```

What happens:
- Creates environment
- Runs random actions
- Shows statistics
- Takes ~10 seconds

Expected output:
```
============================================================
OpenEnv - Random Agent Example
============================================================
...
Episode Statistics:
  Total Steps: 200
  Total Reward: -45.678
```

### Example 2: Baseline Evaluation (All Difficulty Levels)
```bash
python examples/baseline_inference.py --all_tasks --n_episodes 5
```

What happens:
- Evaluates on easy, medium, hard tasks
- Runs 5 episodes each (15 total)
- Calculates scores (0.0–1.0)
- Shows pass/fail rates
- Takes ~30 seconds

Expected output:
```
============================================================
Evaluating EASY task
============================================================
Episode 1/5 (seed=42): Score=0.720 ✓ PASSED
Episode 2/5 (seed=43): Score=0.680 ✗ FAILED
...
Mean Score: 0.700 ± 0.050
Pass Rate: 80.0% (4/5)
```

### Example 3: Train RL Agent (PPO Algorithm)
```bash
python examples/train_openenv.py --total_timesteps 50000
```

What happens:
- Trains PPO agent for 50k steps
- Saves model to `logs/openenv/`
- Shows training progress
- Takes ~5-10 minutes

Expected output:
```
============================================================
OpenEnv Training Script
============================================================
Starting training for 50,000 timesteps...
-------------------------------------------
| rollout/ep_len_mean | 250               |
| rollout/ep_rew_mean | 45.3              |
| time/fps            | 1200              |
-------------------------------------------
Training complete!
```

---

## 🌐 Step 4: Launch Web Demo (Hugging Face Style)

### Run Gradio Interface
```bash
python app.py
```

What happens:
- Starts web server at `http://localhost:7860`
- Opens interactive demo in browser
- Shows drone navigation visualization
- Real-time grading display

Expected output:
```
* Running on local URL: http://localhost:7860
* To create a public link, set share=True in app.launch()
```

Then open your browser to: **http://localhost:7860**

**What you can do in the demo:**
1. Select difficulty (easy/medium/hard)
2. Adjust random seed slider
3. Click "🚀 Run Episode" to see agent perform
4. Click "📊 Compare All Levels" to see comparison table
5. View real-time metrics and grades

---

## 🐳 Step 5: Docker Deployment (Optional)

If you want to deploy as a containerized service:

### Build Docker Image
```bash
docker build -t openenv-drone:latest .
```

### Run Container
```bash
docker run -p 7860:7860 openenv-drone:latest
```

Access at: **http://localhost:7860**

---

## 🧪 Step 6: Run Tests (Verify Everything Works)

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_openenv.py::TestEnvInitialization::test_default_initialization PASSED
tests/test_openenv.py::TestAPIC ompliance::test_gymnasium_check PASSED
...
==================== 40 passed in 15.23s ====================
```

With coverage:
```bash
pytest tests/ --cov=openenv --cov-report=html
```

Then open `htmlcov/index.html` to see detailed coverage report.

---

## 🎯 Common Scenarios

### Scenario 1: "I just want to see it work quickly"
```bash
# Run the simplest example
python examples/basic_usage.py
```

### Scenario 2: "I want to train an RL agent"
```bash
# Train PPO agent
python examples/train_openenv.py --total_timesteps 50000 --verbose 1
```

### Scenario 3: "I want to evaluate performance"
```bash
# Run baseline evaluation on all tasks
python examples/baseline_inference.py --all_tasks --n_episodes 10 --output results.json
```

### Scenario 4: "I want to see the web interface"
```bash
# Launch Gradio demo
python app.py
# Then open http://localhost:7860
```

### Scenario 5: "I want to customize the environment"
Edit `openenv.yaml` then run:
```bash
python examples/baseline_inference.py --task_level medium --config openenv.yaml
```

---

## 🔧 Troubleshooting

### Issue: "Module not found: openenv"
**Solution:** Add project to Python path
```bash
# Windows (Git Bash)
export PYTHONPATH="$PWD:$PYTHONPATH"
python examples/basic_usage.py

# Or install in development mode
pip install -e .
```

### Issue: "Pygame font not initialized"
**Solution:** Disable rendering or initialize fonts
```python
# In your code, set render_mode=None
env = OpenEnv(render_mode=None)  # No rendering
```

### Issue: "Gradio not launching"
**Solution:** Check port availability
```bash
# Kill process on port 7860 (Windows)
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Or use different port
python app.py --port 7861
```

### Issue: "Training is slow"
**Solution:** Reduce complexity
```bash
# Train for fewer steps
python examples/train_openenv.py --total_timesteps 10000

# Use fewer parallel environments
python examples/train_openenv.py --n_envs 1
```

---

## 📊 What Each Command Does

| Command | Time | Output | Purpose |
|---------|------|--------|---------|
| `basic_usage.py` | 10s | Console text | Test API |
| `baseline_inference.py` | 30s | JSON file | Evaluate performance |
| `train_openenv.py` | 5-10 min | Saved model | Train RL agent |
| `app.py` | Instant | Web UI | Interactive demo |
| `pytest tests/` | 15s | Test report | Verify correctness |

---

## 🎓 Learning Path

### Day 1: Understand the Basics
1. Run `examples/basic_usage.py`
2. Read the code to understand API
3. Modify parameters in `openenv.yaml`

### Day 2: Evaluate Performance
1. Run `examples/baseline_inference.py --all_tasks`
2. Analyze results in `results.json`
3. Compare difficulty levels

### Day 3: Train RL Agent
1. Run `examples/train_openenv.py --total_timesteps 50000`
2. Watch training progress
3. Test trained model

### Day 4: Deploy
1. Run `python app.py`
2. Share with team
3. Consider Docker deployment

---

## ✅ Success Checklist

- [ ] Installation completed
- [ ] Basic usage example runs successfully
- [ ] Baseline inference produces scores
- [ ] (Optional) RL agent trained
- [ ] (Optional) Web demo launches
- [ ] (Optional) Tests pass

---

## 🆘 Need Help?

If you encounter issues:

1. **Check error message carefully** - Most issues are dependency-related
2. **Try minimal installation first** - Just `gymnasium` and `numpy`
3. **Disable optional features** - Set `render_mode=None`, skip Gradio
4. **Check Python version** - Requires Python 3.8+
5. **Read full documentation** - See `README.md` for details

---

## 🎉 You're Ready!

Pick a starting point based on your goal:

- **Just testing?** → Run `examples/basic_usage.py`
- **Research?** → Run `examples/baseline_inference.py --all_tasks`
- **Training agents?** → Run `examples/train_openenv.py`
- **Demo for others?** → Run `python app.py`

**Good luck with your reinforcement learning experiments!** 🚀

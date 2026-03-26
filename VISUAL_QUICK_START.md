# 🚀 OpenEnv - Visual Quick Start Guide

## ⚡ Choose Your Path

### Path 1: "I just want to see it work!" (2 minutes)
```bash
pip install gymnasium numpy pygame
python examples/basic_usage.py
```
**What happens:** Random drone flies around, shows statistics

---

### Path 2: "I want to train AI!" (10 minutes)
```bash
pip install stable-baselines3
python examples/train_openenv.py --total_timesteps 50000
```
**What happens:** AI learns to navigate autonomously

---

### Path 3: "I want a visual demo!" (1 minute)
```bash
pip install gradio pyyaml
python app.py
# Open http://localhost:7860 in browser
```
**What happens:** Interactive web interface with visualization

---

### Path 4: "I need to evaluate performance!" (3 minutes)
```bash
python examples/baseline_inference.py --all_tasks
```
**What happens:** Tests all difficulty levels, gives scores 0.0-1.0

---

## 📦 Installation Options

### Minimal (Just run basic example)
```bash
pip install gymnasium numpy pygame
```
Size: ~50 MB | Time: 2 minutes

---

### Full (Everything for RL training)
```bash
pip install -r requirements.txt
```
Size: ~200 MB | Time: 5 minutes

Includes:
- ✅ Core environment
- ✅ RL algorithms (PPO, A2C, SAC)
- ✅ Web interface (Gradio)
- ✅ Configuration (YAML)
- ✅ Testing tools

---

### Development (Contribute to project)
```bash
pip install -e .
```
Installs as package + dev tools (black, flake8, mypy, pytest)

---

## 🎯 What Each File Does (Visual Map)

```
OpenEnv/
│
├── 🧠 THE BRAIN (Core Simulation)
│   └── openenv/core/env.py
│       • Physics engine (gravity, thrust, friction)
│       • Reward calculator (scores AI performance)
│       • Collision detection
│       • Rendering system
│
├── ⚙️ THE CONTROLS (Configuration)
│   ├── openenv/core/config.py    ← Python config class
│   └── openenv.yaml              ← YAML settings file
│       • Adjust difficulty
│       • Tune physics
│       • Modify rewards
│
├── 📊 THE JUDGES (Grading System)
│   └── openenv/core/grader.py
│       • EasyGrader (60% target, 20% time, 20% energy)
│       • MediumGrader (50% target, 25% collision, 15% time, 10% energy)
│       • HardGrader (45% target, 25% collision, 15% wind, 10% time, 5% energy)
│
├── 📚 LEARNING MATERIALS (Examples)
│   ├── examples/basic_usage.py        ← API basics
│   ├── examples/train_openenv.py      ← RL training
│   └── examples/baseline_inference.py ← Performance evaluation
│
├── 🌐 SHOWCASE (Web Demo)
│   └── app.py
│       • Gradio interface
│       • Live visualization
│       • Score display
│
├── 🚢 DEPLOYMENT (Docker)
│   └── Dockerfile
│       • Hugging Face Spaces ready
│       • Production container
│
└── 📖 DOCUMENTATION (Guides)
    ├── README.md               ← Main guide (676+ lines)
    ├── HOW_TO_RUN.md           ← Step-by-step (358 lines)
    ├── WHAT_IS_THIS_PROJECT.md ← This overview
    └── [10+ more docs]
```

---

## 🎮 How The Drone Simulation Works

### Step-by-Step Flow:

```
1. RESET
   ├─ Drone spawns at random position
   ├─ Target appears (green sphere)
   └─ AI receives observation (12 numbers)

2. AI DECIDES
   ├─ Sees: position, velocity, target, time
   ├─ Chooses: thrust, yaw, pitch, roll
   └─ Each value from -1.0 to 1.0

3. PHYSICS ENGINE CALCULATES
   ├─ Applies forces (F=ma)
   ├─ Adds gravity (-9.81 m/s²)
   ├─ Applies air resistance
   └─ Updates position & velocity

4. REWARD COMPUTED
   ├─ Distance reward (-0.15 × distance)
   ├─ Progress bonus (+0.8 × improvement)
   ├─ Velocity penalty (-0.02 × speed)
   └─ Success bonus (+100 if reached target)

5. GRADED (at episode end)
   ├─ Did it reach target? (50%)
   ├─ Did it avoid obstacles? (25%)
   ├─ Was it fast enough? (15%)
   └─ Was it energy efficient? (10%)
   └─ Final score: 0.0 to 1.0
```

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    YOUR CODE                        │
│         (RL Agent / AI Controller)                  │
└───────────────────┬─────────────────────────────────┘
                    │
          Actions: [thrust, yaw, pitch, roll]
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              OpenEnv Environment                    │
│  ┌─────────────────────────────────────────────┐   │
│  │  Physics Engine                             │   │
│  │  • Apply forces                             │   │
│  │  • Gravity simulation                       │   │
│  │  • Collision detection                      │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  Reward Function                            │   │
│  │  • Calculate distance reward                │   │
│  │  • Add progress bonuses                     │   │
│  │  • Apply penalties                          │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  Task Grader                                │   │
│  │  • Evaluate performance                     │   │
│  │  • Score 0.0 to 1.0                         │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────┘
                    │
          Returns: (observation, reward, done, info)
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              LEARNING LOOP                          │
│         Update AI strategy based on reward          │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Three Difficulty Levels Compared

### Easy Mode (Learning to Fly)
```
Warehouse: Empty space
Obstacles: None
Wind: Calm
Target: Large (5.0 units)
Time: 300 steps
Scoring: 60% reach target, 20% time, 20% energy
Pass threshold: 0.70
```

### Medium Mode (Avoiding Obstacles)
```
Warehouse: 5 static obstacles
Obstacles: Boxes, shelves
Wind: Light breeze
Target: Medium (4.0 units)
Time: 500 steps
Scoring: 50% target, 25% collisions, 15% time, 10% energy
Pass threshold: 0.75
```

### Hard Mode (Professional Pilot)
```
Warehouse: 10 moving obstacles
Obstacles: Forklifts, drones
Wind: Strong gusts from vents
Target: Small (3.0 units)
Time: 700 steps
Scoring: 45% target, 25% collisions, 15% wind, 10% time, 5% energy
Pass threshold: 0.80
```

---

## 🎓 Complete Learning Journey

### Week 1: Foundations
```bash
Day 1: Run basic_usage.py
Day 2: Read HOW_TO_RUN.md
Day 3: Modify openenv.yaml parameters
Day 4: Understand observation space (12D)
Day 5: Understand action space (4D)
Day 6: Study reward function
Day 7: Analyze grading criteria
```

### Week 2: Training
```bash
Day 1: Install Stable Baselines3
Day 2: Train PPO for 10k steps
Day 3: Watch learning progress
Day 4: Train for 50k steps
Day 5: Save trained model
Day 6: Load and test model
Day 7: Compare different algorithms
```

### Week 3: Evaluation
```bash
Day 1: Run baseline_inference.py
Day 2: Analyze score distributions
Day 3: Compare easy/medium/hard
Day 4: Export results to JSON
Day 5: Create performance charts
Day 6: Write analysis report
Day 7: Present findings
```

### Week 4: Deployment
```bash
Day 1: Build Docker image
Day 2: Test locally
Day 3: Deploy to Hugging Face
Day 4: Share public link
Day 5: Collect user feedback
Day 6: Iterate improvements
Day 7: Document learnings
```

---

## 🔧 Common Workflows

### Workflow 1: Quick Experiment
```bash
# 1. Change one parameter in openenv.yaml
# e.g., gravity: 5.0 → 15.0

# 2. Run test
python examples/basic_usage.py

# 3. See how behavior changes
```

### Workflow 2: Train & Evaluate
```bash
# 1. Train agent
python examples/train_openenv.py --total_timesteps 100000

# 2. Evaluate performance
python examples/baseline_inference.py --all_tasks --n_episodes 20

# 3. Check results.json
```

### Workflow 3: Debug Issue
```bash
# 1. Run specific test
pytest tests/test_openenv.py::TestRewardFunction -v

# 2. Enable verbose logging
export OPENENV_VERBOSE=1
python examples/train_openenv.py

# 3. Check logs in logs/ directory
```

---

## 📈 Performance Benchmarks

### Expected Results (After 100k training steps):

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| **Mean Score** | 0.85 | 0.72 | 0.58 |
| **Pass Rate** | 90% | 75% | 45% |
| **Avg Steps** | 180 | 320 | 480 |
| **Success Bonus** | Always | Often | Rarely |

### Training Time Estimates:

| Algorithm | 10k Steps | 50k Steps | 100k Steps |
|-----------|-----------|-----------|------------|
| **PPO** | 1 min | 5 min | 10 min |
| **A2C** | 2 min | 10 min | 20 min |
| **SAC** | 3 min | 15 min | 30 min |

---

## 🎯 Pick Your Starting Point

### "I'm a beginner"
→ Start with [`examples/basic_usage.py`](examples/basic_usage.py)

### "I know RL, want to train"
→ Go to [`examples/train_openenv.py`](examples/train_openenv.py)

### "I want visual feedback"
→ Launch [`app.py`](app.py) (opens at http://localhost:7860)

### "I need to benchmark"
→ Run [`baseline_inference.py`](examples/baseline_inference.py)

### "I want to contribute"
→ Read tests and documentation structure

---

## 🆘 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -e .` |
| Pygame error | `pip install pygame --no-cache-dir` |
| Port 7860 in use | `python app.py --port 7861` |
| Slow training | Reduce `--total_timesteps` or use fewer envs |
| Font errors | Already fixed in latest code! |

---

## 📞 Next Steps

1. **Run something now:** `python examples/basic_usage.py`
2. **Read full guide:** [`WHAT_IS_THIS_PROJECT.md`](WHAT_IS_THIS_PROJECT.md)
3. **Join community:** GitHub Discussions
4. **Start training:** Pick an algorithm and go!

---

**You're all set! Choose your path and start exploring!** 🚀

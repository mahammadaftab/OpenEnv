"""
Quick test to verify the 3D drone navigation environment works correctly.
"""

from openenv import OpenEnv, EnvConfig

print("="*60)
print("Testing OpenEnv - 3D Drone Navigation")
print("="*60)

# Create environment
config = EnvConfig(
    task_level='medium',
    verbose=True,
    render_mode=None
)
env = OpenEnv(config=config)

# Test reset
print("\n1. Testing reset()...")
obs, info = env.reset(seed=42)
print(f"   ✓ Observation shape: {obs.shape}")
print(f"   ✓ Observation dtype: {obs.dtype}")
assert obs.shape == (12,), f"Expected shape (12,), got {obs.shape}"

# Test step
print("\n2. Testing step()...")
action = env.action_space.sample()
print(f"   ✓ Action shape: {action.shape}")
obs, reward, terminated, truncated, info = env.step(action)
print(f"   ✓ New observation shape: {obs.shape}")
print(f"   ✓ Reward: {reward:.3f}")
print(f"   ✓ Terminated: {terminated}, Truncated: {truncated}")

# Test multiple steps
print("\n3. Testing multiple steps...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"   Episode ended at step {i+1}")
        obs, info = env.reset()

print(f"   ✓ Completed 10 steps successfully")

# Test state
print("\n4. Testing state()...")
state = env.state()
print(f"   ✓ State shape: {state.shape if state is not None else None}")

# Close
env.close()
print("\n✓ All tests passed!")
print("="*60)

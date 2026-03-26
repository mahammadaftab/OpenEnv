"""
Test rendering with fonts.
"""

from openenv import OpenEnv, EnvConfig
import numpy as np

print("="*60)
print("Testing Rendering with Fonts")
print("="*60)

# Create environment with rendering enabled
config = EnvConfig(
    task_level='medium',
    verbose=False,
    render_mode='rgb_array'
)

env = OpenEnv(config=config)

# Reset
obs, info = env.reset(seed=42)
print("\n✓ Environment created and reset")

# Run a few steps and render
print("\nRunning episode with rendering...")
for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Try to render
    try:
        frame = env.render()
        if frame is not None:
            print(f"Step {step+1}: ✓ Frame rendered, shape: {frame.shape}")
        else:
            print(f"Step {step+1}: ⚠ Render returned None")
    except Exception as e:
        print(f"Step {step+1}: ✗ Render error: {e}")
    
    if terminated or truncated:
        break

env.close()

print("\n" + "="*60)
print("Rendering test completed!")
print("="*60)

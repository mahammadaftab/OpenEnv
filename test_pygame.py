"""
Test Pygame compatibility and rendering.
"""

import pygame
import numpy as np

print("="*60)
print("Testing Pygame Compatibility")
print("="*60)

# Check Pygame version
print(f"\nPygame version: {pygame.version.ver}")

# Initialize Pygame
if pygame.get_init() is None:
    print("Initializing Pygame...")
    pygame.init()

# Test Clock availability
print("\n1. Testing pygame.time.Clock()...")
try:
    clock = pygame.time.Clock()
    print(f"   ✓ pygame.time.Clock() works!")
    print(f"   ✓ Clock object: {clock}")
except AttributeError as e:
    print(f"   ✗ pygame.time.Clock() failed: {e}")

# Test surface creation
print("\n2. Testing surface creation...")
try:
    screen = pygame.Surface((800, 600))
    print(f"   ✓ Surface created: {screen.get_size()}")
except Exception as e:
    print(f"   ✗ Surface creation failed: {e}")

# Test drawing
print("\n3. Testing basic drawing...")
try:
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (255, 0, 0), (400, 300), 50)
    print(f"   ✓ Drawing works!")
except Exception as e:
    print(f"   ✗ Drawing failed: {e}")

# Test RGB conversion
print("\n4. Testing RGB array conversion...")
try:
    raw_data = pygame.image.tostring(screen, "RGB")
    img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((600, 800, 3))
    print(f"   ✓ RGB conversion works! Shape: {img_array.shape}")
except Exception as e:
    print(f"   ✗ RGB conversion failed: {e}")

print("\n" + "="*60)
print("All Pygame tests completed!")
print("="*60)

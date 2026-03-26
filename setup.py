"""
Setup script for OpenEnv package installation.

Usage:
    pip install -e .          # Development mode
    python setup.py install   # Standard installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="openenv",
    version="1.0.0",
    author="OpenEnv Team",
    author_email="openenv@example.com",
    description="A Production-Ready Reinforcement Learning Environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/OpenEnv",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "rl": [
            "stable-baselines3>=2.0.0",
            "sb3-contrib>=2.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
        ],
    },
    include_package_data=True,
    package_data={
        "openenv": ["py.typed"],
    },
    entry_points={
        "console_scripts": [
            # Add command-line scripts here if needed
            # e.g., "openenv-train=openenv.scripts.train:main"
        ],
    },
    keywords=[
        "reinforcement-learning",
        "gymnasium",
        "machine-learning",
        "ai",
        "agent",
        "environment",
        "gym",
        "openai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/OpenEnv/issues",
        "Source": "https://github.com/yourusername/OpenEnv",
        "Documentation": "https://github.com/yourusername/OpenEnv#readme",
    },
)

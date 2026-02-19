"""Setup script for preimage_sampling, models, and training packages."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="preimage_sampling",
    version="0.2.0",
    author="Gabriele Pintus",
    description="Certified Polyhedral Projection for Robust Counterfactual Explanations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielepintus/PreimageCounterfactualSampling",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "auto-LiRPA>=0.4.0",
        "shapely>=2.0.0",
        "cvxpy>=1.4.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "lightning>=2.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

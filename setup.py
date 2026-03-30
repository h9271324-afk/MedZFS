"""
MedZFS: Hallucinating Anatomical Prototypes for Unified Zero-to-Few-Shot
Medical Image Segmentation.

This package implements the method proposed in the accompanying research paper.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="medzfs",
    version="1.0.0",
    description=(
        "Hallucinating Anatomical Prototypes for Unified "
        "Zero-to-Few-Shot Medical Image Segmentation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/MedZFS",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "medzfs-train=training.train:main",
            "medzfs-eval=evaluation.evaluate:main",
            "medzfs-predict=inference.predict:main",
        ],
    },
)

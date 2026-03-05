#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="grass-mil",
    version="0.0.1",
    description="Geometric deep learning framework for spatial omics data",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
            "loocv_command = src.loocv:main",
            "interpretability_report_command = src.interpretability.report_cli:main",
        ]
    },
)

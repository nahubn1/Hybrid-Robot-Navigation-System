# Project Folder Structure

This document provides an overview of the main folders in the repository and their purposes.

```
.
├── .devcontainer/       # Development container configuration
├── .github/             # GitHub settings and workflows
├── assets/              # URDF models and other simulation assets
├── configs/             # Configuration files for models, planners and simulation
│   ├── data_generation/ # Parameters for dataset creation scripts
│   ├── dataset/         # Dataset description files
│   ├── dnn/             # DNN specific settings
│   ├── prm/             # PRM sampling and graph parameters
│   ├── dwa/             # DWA tuning parameters
│   └── simulation/      # Environment and scenario definitions
├── copilot/             # Minimal example module with its own tests
│   └── tests/
├── data/                # Datasets used for training and evaluation
│   ├── raw/             # Generated environments and initial data
│   ├── processed/       # Data processed for DNN training
│   └── training_samples/ # Cached training samples
├── docs/                # Documentation and research notes
├── environment/         # Dependency lists and environment setup files
│   ├── requirements.txt
│   └── setup.sh
├── models/              # Saved machine learning models
│   └── dnn_guidance/    # Checkpoints for the guidance network
├── notebooks/           # notebooks for experimentation and analysis
│   ├── data_exploration/
│   ├── model_prototyping/
│   └── results_analysis/
├── results/             # Logs, metrics and generated plots
│   ├── logs/
│   ├── metrics/
│   └── visualizations/
├── scripts/             # Standalone utility and execution scripts
│   ├── data_generation/
│   ├── model_training/
│   ├── evaluation/
│   ├── simulation/
│   └── utils/
├── src/                 # Source code of the hybrid navigation system
│   ├── data_generation/
│   ├── dnn_guidance/
│   ├── dwa_planner/
│   ├── hybrid_system/
│   ├── mapping/
│   ├── planning_algorithms/
│   ├── prm_planner/
│   ├── simulation/
│   └── utils/
└── tests/               # Unit and integration tests
    ├── unit/
    └── integration/
```

Each folder contains subdirectories or files that contribute to different stages of data preparation, training, and evaluation of the hybrid navigation system.

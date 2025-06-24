# Project Folder Structure

This document provides an overview of the main folders in the repository and their purposes.

```
.
├── configs/             # Configuration files for models, planners and simulation
│   ├── dnn/             # DNN specific settings
│   ├── prm/             # PRM sampling and graph parameters
│   ├── dwa/             # DWA tuning parameters
│   └── simulation/      # Environment and scenario definitions
├── data/                # Datasets used for training and evaluation
│   ├── raw/             # Generated environments and initial data
│   └── processed/       # Data processed for DNN training
├── docs/                # Documentation and research notes
├── environment/         # Dependency lists and environment setup files
├── models/              # Saved machine learning models
│   └── dnn_guidance/    # Checkpoints for the guidance network
├── notebooks/           # Jupyter notebooks for experimentation and analysis
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
│   └── deployment/
├── src/                 # Source code of the hybrid navigation system
│   ├── data_generation/
│   ├── dnn_guidance/
│   ├── dwa_planner/
│   ├── hybrid_system/
│   ├── prm_planner/
│   ├── simulation/
│   └── utils/
└── tests/               # Unit and integration tests
    ├── unit/
    └── integration/
```

Each folder contains subdirectories or files that contribute to different stages of data preparation, training, and evaluation of the hybrid navigation system.

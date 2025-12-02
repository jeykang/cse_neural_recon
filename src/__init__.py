"""
Neural 3D Reconstruction for MobileX Poles Surveillance Robot.

This package provides a complete pipeline for converting multi-camera video 
feeds into high-fidelity, watertight 3D point clouds for surveillance 
coverage analysis.

Modules:
    - data: Data loading and preprocessing (CSE dataset support)
    - models: Neural network architectures (Neural SDF, encodings, etc.)
    - losses: Loss functions (SDF, planar, regularization)
    - training: Training infrastructure (trainer, schedulers, checkpoints)
    - refinement: Post-processing (mesh extraction, statistical filtering)
    - utils: Utilities (visualization, I/O, metrics)
"""

__version__ = '0.1.0'

# Data
from .data import (
    CSEDataset,
    CSEMultiCameraDataset,
    MultiCameraSynchronizer,
    CameraRig,
    DataTransform,
    DepthProcessor
)

# Models
from .models import (
    NeuralSDF,
    NeuralSDFWithPlanar,
    HashGridEncoding,
    MultiResolutionHashGrid,
    PositionalEncoding,
    FourierFeatures,
    PlanarAttention,
    IterativePointFilter,
    ScoreNetwork,
    ScoreBasedDenoiser
)

# Losses
from .losses import (
    SDFLoss,
    SurfaceLoss,
    FrespaceLoss,
    EikonalLoss,
    PlanarConsistencyLoss,
    ManhattanLoss,
    SmoothnessLoss,
    LaplacianLoss,
    TotalVariationLoss
)

# Training
from .training import (
    Trainer,
    TrainingConfig,
    CheckpointManager,
    get_scheduler,
    get_sampler
)

# Refinement
from .refinement import (
    extract_mesh_from_sdf,
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
    voxel_downsample,
    NeuralRefiner,
    PointCompletionNetwork
)

# Utils
from .utils import (
    visualize_point_cloud,
    visualize_mesh,
    visualize_depth,
    load_point_cloud,
    save_point_cloud,
    chamfer_distance,
    f_score
)

__all__ = [
    # Version
    '__version__',
    # Data
    'CSEDataset',
    'CSEMultiCameraDataset', 
    'MultiCameraSynchronizer',
    'CameraRig',
    'DataTransform',
    'DepthProcessor',
    # Models
    'NeuralSDF',
    'NeuralSDFWithPlanar',
    'HashGridEncoding',
    'MultiResolutionHashGrid',
    'PositionalEncoding',
    'FourierFeatures',
    'PlanarAttention',
    'IterativePointFilter',
    'ScoreNetwork',
    'ScoreBasedDenoiser',
    # Losses
    'SDFLoss',
    'SurfaceLoss',
    'FrespaceLoss',
    'EikonalLoss',
    'PlanarConsistencyLoss',
    'ManhattanLoss',
    'SmoothnessLoss',
    'LaplacianLoss',
    'TotalVariationLoss',
    # Training
    'Trainer',
    'TrainingConfig',
    'CheckpointManager',
    'get_scheduler',
    'get_sampler',
    # Refinement
    'extract_mesh_from_sdf',
    'StatisticalOutlierRemoval',
    'RadiusOutlierRemoval',
    'voxel_downsample',
    'NeuralRefiner',
    'PointCompletionNetwork',
    # Utils
    'visualize_point_cloud',
    'visualize_mesh',
    'visualize_depth',
    'load_point_cloud',
    'save_point_cloud',
    'chamfer_distance',
    'f_score'
]

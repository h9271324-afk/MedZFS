"""
Data loading and preprocessing modules for MedZFS.

Provides episodic medical image datasets, anatomical knowledge graphs,
preprocessing pipelines, and domain-specific augmentations.
"""

# Lazy imports to avoid requiring nibabel for non-data tasks
__all__ = [
    "EpisodicMedicalDataset",
    "MedicalVolumeDataset",
    "AnatomicalGraphBuilder",
    "MedicalPreprocessor",
    "MedicalTransforms",
]


def __getattr__(name):
    if name == "EpisodicMedicalDataset":
        from data.dataset import EpisodicMedicalDataset
        return EpisodicMedicalDataset
    elif name == "MedicalVolumeDataset":
        from data.dataset import MedicalVolumeDataset
        return MedicalVolumeDataset
    elif name == "AnatomicalGraphBuilder":
        from data.anatomical_graphs import AnatomicalGraphBuilder
        return AnatomicalGraphBuilder
    elif name == "MedicalPreprocessor":
        from data.preprocessing import MedicalPreprocessor
        return MedicalPreprocessor
    elif name == "MedicalTransforms":
        from data.transforms import MedicalTransforms
        return MedicalTransforms
    raise AttributeError(f"module 'data' has no attribute {name}")

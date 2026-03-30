"""
MedZFS Model Architecture Modules.

Implements the complete MedZFS framework for unified zero-to-few-shot
medical image segmentation via hallucinated anatomical prototypes.
"""

from models.medzfs import MedZFS
from models.visual_encoder import VisualEncoder
from models.text_encoder import TextEncoder
from models.hallucination import PrototypeHallucinator
from models.graph_network import HeterogeneousGraphNetwork
from models.fusion import GraphConstrainedFusion
from models.prototype_mining import HardPrototypeMiner
from models.loss_functions import MedZFSLoss

__all__ = [
    "MedZFS",
    "VisualEncoder",
    "TextEncoder",
    "PrototypeHallucinator",
    "HeterogeneousGraphNetwork",
    "GraphConstrainedFusion",
    "HardPrototypeMiner",
    "MedZFSLoss",
]

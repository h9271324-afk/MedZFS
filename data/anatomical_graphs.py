"""
Anatomical Knowledge Graph Definitions for MedZFS.

Defines heterogeneous anatomical knowledge graphs for each dataset:
  - Abdomen (MRI/CT): Liver, Right Kidney, Left Kidney, Spleen
  - Cardiac MRI: LV Myocardium, LV Blood Pool, Right Ventricle

Each graph encodes four relation types:
  - spatial: Positional relations (e.g., superior-to, anterior-to)
  - hierarchical: Part-whole relations (e.g., contains, part-of)
  - functional: Physiological relations (e.g., connected-via-blood-supply)
  - pathological: Disease co-occurrence relations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class AnatomicalNode:
    """Represents an anatomical structure node in the knowledge graph."""
    name: str
    label: int
    description: str
    parent: str = ""


@dataclass
class AnatomicalEdge:
    """Represents a relation edge in the knowledge graph."""
    source: str
    target: str
    relation: str        # One of: spatial, hierarchical, functional, pathological
    description: str     # Human-readable description of the relation


@dataclass
class AnatomicalGraph:
    """Complete anatomical knowledge graph."""
    name: str
    nodes: List[AnatomicalNode]
    edges: List[AnatomicalEdge]
    relation_types: List[str] = field(
        default_factory=lambda: ["spatial", "hierarchical", "functional", "pathological"]
    )

    def get_node_names(self) -> List[str]:
        """Return list of node names."""
        return [n.name for n in self.nodes]

    def get_node_descriptions(self) -> List[str]:
        """Return list of node textual descriptions."""
        return [n.description for n in self.nodes]

    def get_edge_index_by_relation(self) -> Dict[str, torch.LongTensor]:
        """Get edge indices grouped by relation type.

        Returns:
            Dictionary mapping relation type to (2, num_edges) tensor.
        """
        name_to_idx = {n.name: i for i, n in enumerate(self.nodes)}
        edge_dict: Dict[str, List[Tuple[int, int]]] = {r: [] for r in self.relation_types}

        for edge in self.edges:
            if edge.source in name_to_idx and edge.target in name_to_idx:
                src = name_to_idx[edge.source]
                tgt = name_to_idx[edge.target]
                edge_dict[edge.relation].append((src, tgt))

        result = {}
        for rel, edges in edge_dict.items():
            if edges:
                result[rel] = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                result[rel] = torch.zeros(2, 0, dtype=torch.long)

        return result

    def get_full_edge_index(self) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Get combined edge index and edge type tensor.

        Returns:
            (edge_index, edge_type) where edge_index is (2, E) and
            edge_type is (E,) with integer relation type IDs.
        """
        name_to_idx = {n.name: i for i, n in enumerate(self.nodes)}
        rel_to_idx = {r: i for i, r in enumerate(self.relation_types)}

        sources, targets, types = [], [], []
        for edge in self.edges:
            if edge.source in name_to_idx and edge.target in name_to_idx:
                sources.append(name_to_idx[edge.source])
                targets.append(name_to_idx[edge.target])
                types.append(rel_to_idx[edge.relation])

        if sources:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            edge_type = torch.tensor(types, dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        return edge_index, edge_type


class AnatomicalGraphBuilder:
    """Factory class for constructing anatomical knowledge graphs.

    Supports Abdomen (MRI/CT) and Cardiac MRI graphs, encoding spatial,
    hierarchical, functional, and pathological relations between organs.
    """

    @staticmethod
    def build_abdomen_graph() -> AnatomicalGraph:
        """Build the abdominal organ knowledge graph.

        Covers: Liver, Right Kidney, Left Kidney, Spleen, plus
        contextual structures (Stomach, Colon, Pancreas, Aorta).
        """
        nodes = [
            AnatomicalNode(
                name="liver", label=1,
                description="A large, wedge-shaped organ in the upper right abdomen beneath the diaphragm with a smooth capsular surface and homogeneous parenchyma.",
                parent="abdomen",
            ),
            AnatomicalNode(
                name="right_kidney", label=2,
                description="A bean-shaped retroperitoneal organ on the right side, slightly lower than the left kidney, with distinct cortex and medulla.",
                parent="abdomen",
            ),
            AnatomicalNode(
                name="left_kidney", label=3,
                description="A bean-shaped retroperitoneal organ on the left side near the spleen, with distinct cortex and medulla.",
                parent="abdomen",
            ),
            AnatomicalNode(
                name="spleen", label=4,
                description="An ovoid organ in the left upper quadrant, posterior to the stomach and superior to the left kidney, with homogeneous signal.",
                parent="abdomen",
            ),
            AnatomicalNode(
                name="stomach", label=0,
                description="A hollow muscular organ in the left upper abdomen, anterior to the spleen and inferior to the diaphragm.",
                parent="abdomen",
            ),
            AnatomicalNode(
                name="aorta", label=0,
                description="The main arterial trunk descending along the vertebral column, providing blood supply to abdominal organs.",
                parent="abdomen",
            ),
            AnatomicalNode(
                name="pancreas", label=0,
                description="An elongated gland posterior to the stomach, extending from the duodenum to the spleen.",
                parent="abdomen",
            ),
        ]

        edges = [
            # --- Spatial Relations ---
            AnatomicalEdge("liver", "right_kidney", "spatial", "Liver is superior to right kidney"),
            AnatomicalEdge("liver", "stomach", "spatial", "Liver is right-lateral to stomach"),
            AnatomicalEdge("spleen", "left_kidney", "spatial", "Spleen is superior to left kidney"),
            AnatomicalEdge("spleen", "stomach", "spatial", "Spleen is posterior to stomach"),
            AnatomicalEdge("right_kidney", "left_kidney", "spatial", "Right kidney is contralateral to left kidney"),
            AnatomicalEdge("liver", "spleen", "spatial", "Liver is contralateral to spleen"),
            AnatomicalEdge("pancreas", "stomach", "spatial", "Pancreas is posterior to stomach"),
            AnatomicalEdge("aorta", "left_kidney", "spatial", "Aorta is medial to left kidney"),
            AnatomicalEdge("aorta", "right_kidney", "spatial", "Aorta is medial to right kidney"),

            # --- Hierarchical Relations ---
            AnatomicalEdge("liver", "right_kidney", "hierarchical", "Both are abdominal organs"),
            AnatomicalEdge("left_kidney", "right_kidney", "hierarchical", "Bilateral kidney pair"),
            AnatomicalEdge("spleen", "stomach", "hierarchical", "Both in left upper quadrant"),

            # --- Functional Relations ---
            AnatomicalEdge("liver", "aorta", "functional", "Liver receives arterial blood from hepatic artery via aorta"),
            AnatomicalEdge("spleen", "aorta", "functional", "Spleen receives blood from splenic artery via aorta"),
            AnatomicalEdge("right_kidney", "aorta", "functional", "Right kidney receives blood from renal artery"),
            AnatomicalEdge("left_kidney", "aorta", "functional", "Left kidney receives blood from renal artery"),
            AnatomicalEdge("liver", "pancreas", "functional", "Liver and pancreas share biliary drainage"),

            # --- Pathological Relations ---
            AnatomicalEdge("liver", "spleen", "pathological", "Portal hypertension affects both liver and spleen"),
            AnatomicalEdge("right_kidney", "left_kidney", "pathological", "Bilateral renal disease co-occurrence"),
            AnatomicalEdge("liver", "pancreas", "pathological", "Hepatobiliary-pancreatic disease association"),
        ]

        return AnatomicalGraph(name="abdomen", nodes=nodes, edges=edges)

    @staticmethod
    def build_cardiac_graph() -> AnatomicalGraph:
        """Build the cardiac structure knowledge graph.

        Covers: LV Myocardium, LV Blood Pool, Right Ventricle, plus
        contextual structures (Interventricular Septum, Pericardium).
        """
        nodes = [
            AnatomicalNode(
                name="lv_myo", label=1,
                description="The left ventricular myocardium, a thick muscular wall forming a ring around the LV cavity with intermediate signal intensity on cardiac MRI.",
                parent="heart",
            ),
            AnatomicalNode(
                name="lv_bp", label=2,
                description="The left ventricular blood pool, the bright cavity enclosed by the LV myocardium, showing high signal due to flowing blood.",
                parent="heart",
            ),
            AnatomicalNode(
                name="rv", label=3,
                description="The right ventricle, a crescent-shaped chamber with a thinner wall than the LV, wrapping around the interventricular septum.",
                parent="heart",
            ),
            AnatomicalNode(
                name="ivs", label=0,
                description="The interventricular septum, the muscular wall separating left and right ventricles.",
                parent="heart",
            ),
            AnatomicalNode(
                name="pericardium", label=0,
                description="The fibrous sac surrounding the heart, visible as a thin line on cardiac MRI.",
                parent="heart",
            ),
        ]

        edges = [
            # --- Spatial Relations ---
            AnatomicalEdge("lv_myo", "lv_bp", "spatial", "LV myocardium surrounds LV blood pool"),
            AnatomicalEdge("rv", "lv_myo", "spatial", "RV is anterior and rightward to LV"),
            AnatomicalEdge("ivs", "lv_bp", "spatial", "IVS borders LV blood pool medially"),
            AnatomicalEdge("ivs", "rv", "spatial", "IVS separates LV from RV"),
            AnatomicalEdge("pericardium", "lv_myo", "spatial", "Pericardium encloses the heart externally"),
            AnatomicalEdge("pericardium", "rv", "spatial", "Pericardium encloses RV externally"),

            # --- Hierarchical Relations ---
            AnatomicalEdge("lv_myo", "lv_bp", "hierarchical", "LV myocardium contains LV blood pool"),
            AnatomicalEdge("pericardium", "lv_myo", "hierarchical", "Pericardium contains heart structures"),
            AnatomicalEdge("pericardium", "rv", "hierarchical", "Pericardium contains RV"),

            # --- Functional Relations ---
            AnatomicalEdge("lv_myo", "lv_bp", "functional", "LV myocardium contracts to eject blood from LV cavity"),
            AnatomicalEdge("rv", "lv_bp", "functional", "RV and LV maintain serial circulation"),
            AnatomicalEdge("ivs", "lv_myo", "functional", "IVS contributes to LV contraction"),

            # --- Pathological Relations ---
            AnatomicalEdge("lv_myo", "rv", "pathological", "Cardiomyopathy can affect both ventricles"),
            AnatomicalEdge("lv_myo", "ivs", "pathological", "Hypertrophic cardiomyopathy thickens IVS and LV wall"),
            AnatomicalEdge("rv", "lv_bp", "pathological", "Heart failure affects biventricular function"),
        ]

        return AnatomicalGraph(name="cardiac", nodes=nodes, edges=edges)

    @staticmethod
    def build_graph(dataset_name: str) -> AnatomicalGraph:
        """Build the appropriate anatomical graph based on dataset name.

        Args:
            dataset_name: One of "abd_mri", "abd_ct", "cmr".

        Returns:
            The anatomical knowledge graph for the dataset.
        """
        if dataset_name in ("abd_mri", "abd_ct"):
            return AnatomicalGraphBuilder.build_abdomen_graph()
        elif dataset_name == "cmr":
            return AnatomicalGraphBuilder.build_cardiac_graph()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Expected: abd_mri, abd_ct, cmr")

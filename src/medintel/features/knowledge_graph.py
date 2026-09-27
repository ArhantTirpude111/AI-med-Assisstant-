"""Medical knowledge-graph construction & visualization.

Builds a graph of clinical relationships from extracted entities:

    Diagnosis ── has symptom ──▶ Symptom
    Diagnosis ── treated with ─▶ Medication
    Diagnosis ── evidenced by ─▶ Lab finding
    Diagnosis ── involves ─────▶ Anatomy

Relationships are inferred by **co-occurrence on the same page** (a pragmatic,
explainable heuristic), falling back to document-level co-occurrence. The graph
renders three ways: NetworkX (programmatic), PyVis (interactive HTML, optional),
and Plotly (inline in Streamlit, no extra dependency).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from medintel.logging_config import get_logger
from medintel.models.schemas import EntityType, MedicalEntities

logger = get_logger(__name__)

_NODE_COLORS = {
    EntityType.DIAGNOSIS.value: "#e74c3c",
    EntityType.SYMPTOM.value: "#f39c12",
    EntityType.MEDICATION.value: "#3498db",
    EntityType.LAB.value: "#9b59b6",
    EntityType.ANATOMY.value: "#16a085",
}


class KnowledgeGraphService:
    """Construct and render a clinical knowledge graph."""

    def build(self, entities: MedicalEntities) -> nx.Graph:
        graph = nx.Graph()
        # page -> {entity_type: set(labels)}
        page_index: dict[Any, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        def _add_entity(label: str, etype: str, page: Any) -> None:
            label = label.strip()
            if not label:
                return
            node_id = f"{etype}:{label.lower()}"
            if node_id not in graph:
                graph.add_node(node_id, label=label, type=etype, color=_NODE_COLORS.get(etype, "#888"))
            page_index[page or "doc"][etype].add(node_id)

        for e in entities.diagnoses:
            _add_entity(e.text, EntityType.DIAGNOSIS.value, e.page_number)
        for e in entities.symptoms:
            _add_entity(e.text, EntityType.SYMPTOM.value, e.page_number)
        for e in entities.medications:
            _add_entity(e.text, EntityType.MEDICATION.value, e.page_number)
        for e in entities.anatomy:
            _add_entity(e.text, EntityType.ANATOMY.value, e.page_number)
        for lab in entities.lab_results:
            label = f"{lab.name} ({lab.flag.value})" if lab.flag.value != "unknown" else lab.name
            _add_entity(label, EntityType.LAB.value, lab.page_number)

        # Link diagnoses to co-occurring entities on each page.
        relation = {
            EntityType.SYMPTOM.value: "has symptom",
            EntityType.MEDICATION.value: "treated with",
            EntityType.LAB.value: "evidenced by",
            EntityType.ANATOMY.value: "involves",
        }
        for buckets in page_index.values():
            diagnoses = buckets.get(EntityType.DIAGNOSIS.value, set())
            for dx in diagnoses:
                for etype, rel in relation.items():
                    for target in buckets.get(etype, set()):
                        graph.add_edge(dx, target, relation=rel)
            # If no diagnosis on the page, still connect symptoms↔labs for context.
            if not diagnoses:
                symptoms = buckets.get(EntityType.SYMPTOM.value, set())
                for s in symptoms:
                    for lab in buckets.get(EntityType.LAB.value, set()):
                        graph.add_edge(s, lab, relation="associated with")

        logger.info("Knowledge graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    # -- stats --------------------------------------------------------------
    @staticmethod
    def stats(graph: nx.Graph) -> dict[str, int]:
        by_type: dict[str, int] = defaultdict(int)
        for _, data in graph.nodes(data=True):
            by_type[data.get("type", "unknown")] += 1
        return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), **by_type}

    # -- PyVis (interactive HTML) ------------------------------------------
    def to_pyvis_html(self, graph: nx.Graph, output_path: str | Path | None = None) -> str | None:
        """Render to a standalone interactive HTML file (requires pyvis)."""
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed; skipping interactive HTML render.")
            return None
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#222")
        net.barnes_hut()
        for node, data in graph.nodes(data=True):
            net.add_node(node, label=data.get("label", node), color=data.get("color", "#888"),
                         title=f"{data.get('type', '')}: {data.get('label', '')}")
        for u, v, data in graph.edges(data=True):
            net.add_edge(u, v, title=data.get("relation", ""))
        if output_path:
            net.save_graph(str(output_path))
            return str(output_path)
        return net.generate_html()

    # -- Plotly (inline) ----------------------------------------------------
    def to_plotly_figure(self, graph: nx.Graph) -> Any:
        """Return a Plotly Figure for inline display in Streamlit."""
        import plotly.graph_objects as go

        if graph.number_of_nodes() == 0:
            return go.Figure()

        pos = nx.spring_layout(graph, seed=42, k=0.6)
        edge_x, edge_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line={"width": 1, "color": "#cbd5e1"}, hoverinfo="none",
        )
        node_x, node_y, text, colors, hover = [], [], [], [], []
        for node, data in graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(data.get("label", node))
            colors.append(data.get("color", "#888"))
            hover.append(f"{data.get('type', '')}: {data.get('label', '')}")
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center",
            hovertext=hover, hoverinfo="text",
            marker={"size": 18, "color": colors, "line": {"width": 2, "color": "#fff"}},
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False, margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis={"visible": False}, yaxis={"visible": False}, height=550,
            plot_bgcolor="white",
        )
        return fig

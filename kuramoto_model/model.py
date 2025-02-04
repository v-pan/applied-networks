from typing import List, Tuple
from raphtory import Graph
import raphtory

import matplotlib.pyplot as plt
import numpy as np


def phase_dt(vertex_i: raphtory.Node):
    phase_i = vertex_i.properties["phase"]

    total_phase = 0.0
    for edge in vertex_i.edges:
        phase_j = edge.dst.properties["phase"]
        total_phase += edge.properties["weight"] * np.sin(phase_j - phase_i)

    # normalise neighbouring additions
    total_phase /= vertex_i.in_degree()

    total_phase += vertex_i.properties["natural_frequency"]

    return total_phase


def coupling_dt(edge: raphtory.Edge, timescale_separation: float):
    phase_i = edge.src.properties["phase"]
    phase_j = edge.dst.properties["phase"]

    cur_weight = edge.properties["weight"]

    return timescale_separation * (cur_weight + np.sin(phase_j - phase_i))


NODES = 50
TIMESTEPS = 50
TIME_SEP = 1


# Set up nodes and edges between them
def densely_connected_graph(nodes=10) -> Tuple[raphtory.Graph, List[raphtory.Edge]]:
    graph = Graph()
    edges = []

    for n in range(nodes):
        graph.add_node(0, n, properties={"natural_frequency": 1.0, "phase": 0.0 + n})

    for node in graph.nodes:
        for other_node in graph.nodes:
            if other_node.id == node.id:
                continue

            edge = graph.add_edge(0, node.id, other_node.id, properties={"weight": 1.0})
            edges.append(edge)

    return graph, edges


graph, edges = densely_connected_graph(NODES)

for i in range(1, TIMESTEPS):
    # Update vertices
    for vertex in graph.nodes:
        nat_freq = vertex.properties["natural_frequency"]
        new_phase = phase_dt(vertex)

        graph.add_node(
            i, vertex.id, properties={"natural_frequency": nat_freq, "phase": new_phase}
        )

    # Update edges
    for edge in edges:
        src_id = edge.src.id
        dst_id = edge.dst.id

        new_weight = coupling_dt(edge, TIME_SEP)

        graph.add_edge(i, src_id, dst_id, properties={"weight": new_weight})


for node in graph.nodes:
    phase = []
    for timestep in node.expanding(1):
        phase.append(timestep.properties["phase"])

    plt.plot(list(range(0, TIMESTEPS)), phase)

plt.show()

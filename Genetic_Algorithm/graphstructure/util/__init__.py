"""
    In this module, some different general utility functions are defined.
"""

import numpy as np

from ..graph3d import SubstructureInterface

def sub_to_plotly_data(sub: SubstructureInterface):
    """Converts edge and node data to a format used by plotly.

    It is better to do this on the python side, as it somehow takes ages on the
    JS side.

    Args:
        sub: Substructure object

    Returns:
        List of nodes as 1xn array; list of coordinate lists of the edges.
    """
    nodes = np.array([n.as_array() for n in sub.nodes])

    # collect edge data so that plotly can take it in right away
    x, y, z = [], [], []
    for edge in sub.edges:
        a, b = edge.as_tuple()
        # this is the weird way, the plotly takes in the edge values...
        x += [nodes[a, 0], nodes[b, 0], None]
        y += [nodes[a, 1], nodes[b, 1], None]
        z += [nodes[a, 2], nodes[b, 2], None]

    return nodes.T.tolist(), [x, y, z]



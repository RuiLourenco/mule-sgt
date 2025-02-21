import numpy as np
import networkx as nx
from scipy.sparse import csgraph

def build_connected_graph(image_shape, w_h=100, w_d=10):
    """Constructs a fully connected graph with balanced horizontal and diagonal edges."""
    h, w = image_shape
    G = nx.grid_2d_graph(h, w)

    # Iterate over the grid to add edges
    for i in range(h - 1):
        for j in range(w - 1):
            # Horizontal edges (stronger connection)
            G.add_edge((i, j), (i, j + 1), weight=w_h)
            G.add_edge((i, j + 1), (i, j), weight=w_h)  # Ensure bidirectional edge

            # Diagonal edges (weaker connection)
            G.add_edge((i, j), (i + 1, j + 1), weight=w_d)
            G.add_edge((i + 1, j + 1), (i, j), weight=w_d)  # Ensure bidirectional edge

    # Ensure the graph is connected
    if not nx.is_connected(G):
        raise ValueError("Graph is disconnected!")

    # Compute the unnormalized Laplacian
    L = nx.laplacian_matrix(G, weight='weight')
    
    return L

# Example usage:
image_shape = (16, 16)
L = build_connected_graph(image_shape)

# Convert the sparse Laplacian to a dense format for eigenvalue computation
L_dense = L.todense()

# Compute the eigenvalues and eigenvectors of the unnormalized Laplacian
eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

# The smallest eigenvalue should be 0, and its corresponding eigenvector should be constant
print("Smallest Eigenvalue: ", eigenvalues[0])
print("Eigenvector corresponding to the smallest eigenvalue: ", eigenvectors[:, 0])

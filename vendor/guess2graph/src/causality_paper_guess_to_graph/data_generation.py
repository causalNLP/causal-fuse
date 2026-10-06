"""
Linear DGP Data Generation Module

Generates synthetic causal data using linear structural equation models.
"""

import numpy as np
from typing import Tuple, Optional


def generate_er_dag_adjacency_matrix(
    num_nodes: int, 
    edge_probability: float,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate an Erdős-Rényi random DAG adjacency matrix.
    
    Args:
        num_nodes: Number of nodes in the DAG
        edge_probability: Probability of edge existence (controls sparsity)
        random_seed: Random seed for reproducibility
    
    Returns:
        Binary adjacency matrix representing the DAG
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random adjacency matrix
    adjacency_matrix = np.random.rand(num_nodes, num_nodes)
    
    # Apply edge probability threshold
    adjacency_matrix = (adjacency_matrix < edge_probability).astype(int)
    
    # Ensure DAG property by making it lower triangular (no self-loops, no cycles)
    adjacency_matrix = np.tril(adjacency_matrix, k=-1)
    
    return adjacency_matrix


def generate_weighted_adjacency_matrix(
    binary_adjacency: np.ndarray,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Convert binary adjacency matrix to weighted adjacency matrix.
    
    Args:
        binary_adjacency: Binary adjacency matrix
        random_seed: Random seed for reproducibility
    
    Returns:
        Weighted adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    weighted_adjacency = binary_adjacency.astype(float)
    
    # Find edges (non-zero entries)
    edges = np.nonzero(weighted_adjacency)
    
    if len(edges[0]) > 0:
        # Generate weights avoiding values close to 0 to prevent faithfulness violations
        # Sample from [-2.5, -1.5] or [1.5, 2.5] uniformly
        signs = np.random.choice([-1, 1], size=len(edges[0]))
        weights = np.random.uniform(0.1, 2, size=len(edges[0]))
        weights = weights * signs
        
        # Assign weights to edges
        weighted_adjacency[edges] = weights
    
    return weighted_adjacency


def generate_linear_gaussian_data(
    weighted_adjacency: np.ndarray,
    num_samples: int,
    noise_variance: float = 1.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate data from a linear Gaussian structural equation model.
    
    Args:
        weighted_adjacency: Weighted adjacency matrix of the DAG
        num_samples: Number of samples to generate
        noise_variance: Variance of Gaussian noise
        random_seed: Random seed for reproducibility
    
    Returns:
        Data matrix of shape (num_samples, num_nodes)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_nodes = weighted_adjacency.shape[0]
    
    # Initialize data matrix
    data = np.zeros((num_samples, num_nodes))
    
    # Generate data following topological ordering (lower triangular structure)
    for i in range(num_nodes):
        # Generate noise for this variable
        noise = np.random.normal(0, np.sqrt(noise_variance), num_samples)
        
        # Compute linear combination of parents
        parent_contribution = np.zeros(num_samples)
        for j in range(i):  # Only consider previous nodes (DAG property)
            if weighted_adjacency[i, j] != 0:
                parent_contribution += weighted_adjacency[i, j] * data[:, j]
        
        # Final value is linear combination of parents plus noise
        data[:, i] = parent_contribution + noise
    
    return data


def generate_linear_data(n_nodes, edge_probability, sample_size):
    """
    Generate synthetic linear causal data.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes (variables) in the graph
    edge_probability : float
        Probability of edge existence (0 to 1)
    sample_size : int
        Number of samples to generate
        
    Returns:
    --------
    tuple
        (G, data) where G is the weighted adjacency matrix and data is the sample matrix
    """
    # Basic validation
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    
    # # Convert sparsity to edge probability
    # # sparsity is expected edges per node, so total expected edges = sparsity * n_nodes
    # # For lower triangular matrix, max possible edges = n_nodes * (n_nodes - 1) / 2
    # max_edges = n_nodes * (n_nodes - 1) // 2
    # if max_edges == 0:
    #     edge_probability = 0
    # else:
    #     expected_edges = min(sparsity * n_nodes, max_edges)
    #     edge_probability = expected_edges / max_edges
    
    # Generate binary DAG
    binary_adj = generate_er_dag_adjacency_matrix(n_nodes, edge_probability)
    
    # Generate weighted adjacency matrix
    weighted_adj = generate_weighted_adjacency_matrix(binary_adj)
    
    # Generate data
    data = generate_linear_gaussian_data(weighted_adj, sample_size)
    
    # Standardize data (zero mean, unit variance)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    return weighted_adj, data
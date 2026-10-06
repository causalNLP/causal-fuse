"""
Metrics Module

Calculate accuracy metrics between predicted and true graphs.
"""

import numpy as np


def evaluate_skeleton_recovery(
    true_skeleton: np.ndarray,
    estimated_skeleton: np.ndarray
) -> dict:
    """
    Evaluate skeleton recovery performance using precision, recall, and F1 score.
    
    Args:
        true_skeleton: True skeleton adjacency matrix (binary, symmetric)
        estimated_skeleton: Estimated skeleton adjacency matrix (binary, symmetric)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Check if inputs are actually skeletons (symmetric)
    if not np.array_equal(true_skeleton, true_skeleton.T):
        raise ValueError("True graph is not symmetric - expected skeleton, got DAG")
    if not np.array_equal(estimated_skeleton, estimated_skeleton.T):
        raise ValueError("Estimated graph is not symmetric - expected skeleton, got DAG")
    
    # Ensure matrices are binary
    true_skeleton = (true_skeleton != 0).astype(int)
    estimated_skeleton = (estimated_skeleton != 0).astype(int)
    
    # Due to symmetry, take upper triangle to avoid double counting
    true_upper = np.triu(true_skeleton, k=1)
    estimated_upper = np.triu(estimated_skeleton, k=1)
    
    # Calculate confusion matrix components
    true_positives = np.sum((true_upper == 1) & (estimated_upper == 1))
    false_positives = np.sum((true_upper == 0) & (estimated_upper == 1))
    false_negatives = np.sum((true_upper == 1) & (estimated_upper == 0))
    true_negatives = np.sum((true_upper == 0) & (estimated_upper == 0))
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'true_negatives': int(true_negatives)
    }


def calculate_metrics(
    true_graph: np.ndarray,
    predicted_graph: np.ndarray,
    metrics_to_compute: list = ['f1', 'precision', 'recall']
) -> dict:
    """
    Calculate specified metrics between true and predicted graphs
    
    Args:
        true_graph: True adjacency matrix
        predicted_graph: Predicted adjacency matrix
        metrics_to_compute: List of metrics to calculate
        
    Returns:
        Dictionary with computed metrics
    """
    # Validate input shapes
    if true_graph.shape != predicted_graph.shape:
        raise ValueError(f"Graph shapes don't match: {true_graph.shape} vs {predicted_graph.shape}")
    
    # Calculate all metrics using custom evaluation function
    all_metrics = evaluate_skeleton_recovery(true_graph, predicted_graph)
    
    # Extract requested metrics
    result = {}
    for metric in metrics_to_compute:
        if metric in all_metrics:
            result[metric] = all_metrics[metric]
        else:
            available_metrics = list(all_metrics.keys())
            raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available_metrics}")
    
    return result
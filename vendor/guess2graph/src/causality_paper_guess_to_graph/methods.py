"""
Causal Discovery Methods Module

Implements PC and Stable PC algorithms using causal-learn.
"""

import numpy as np
import time
import os
import networkx as nx
from itertools import combinations
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, CIT
# from networkx.algorithms.d_separation import is_d_separator


def run_pc(data: np.ndarray, alpha: float = 0.01, ci_test: str = "fisherz") -> np.ndarray:
    """
    Run standard PC algorithm
    cd te
    Args:
        data: Empirical sample matrix (n_samples, n_features)
        alpha: Significance level for independence tests
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Generate fresh random permutation for each call
    seed = int.from_bytes(os.urandom(4), 'little')
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n_vars)
    
    # Permute data columns
    permuted_data = data[:, permutation]
    
    # Run PC algorithm on permuted data
    cg = pc(permuted_data, alpha=alpha, indep_test=ci_test, stable=False)
    
    # Extract graph
    graph = cg.G.graph
    
    # Create inverse permutation
    inverse_perm = np.empty(n_vars, dtype=int)
    inverse_perm[permutation] = np.arange(n_vars)
    
    # Apply inverse permutation to graph (both rows and columns)
    graph = graph[np.ix_(inverse_perm, inverse_perm)]
    
    # Convert to undirected skeleton (symmetric adjacency matrix)
    skeleton = np.abs(graph)
    skeleton = np.maximum(skeleton, skeleton.T)
    skeleton = (skeleton > 0).astype(int)
    
    return skeleton


def run_stable_pc(data: np.ndarray, alpha: float = 0.01, ci_test: str = "fisherz") -> np.ndarray:
    """
    Run Stable PC algorithm
    
    Args:
        data: Empirical sample matrix
        alpha: Significance level
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    # Run Stable PC algorithm with specified CI test
    cg = pc(data, alpha=alpha, indep_test=ci_test, stable=True)
    
    # Extract graph and convert to skeleton
    graph = cg.G.graph
    
    # Convert to undirected skeleton (symmetric adjacency matrix)
    skeleton = np.abs(graph)
    skeleton = np.maximum(skeleton, skeleton.T)
    skeleton = (skeleton > 0).astype(int)
    
    return skeleton


def run_pc_guess_dag(data: np.ndarray, guessed_dag: np.ndarray, alpha: float = 0.01, ci_test: str = "fisherz") -> np.ndarray:
    """
    Run PC algorithm with expert-guided ordering based on guessed DAG
    
    Args:
        data: Data matrix (n_samples, n_features)
        guessed_dag: Guessed DAG adjacency matrix (n_features, n_features)
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    cit = CIT(data, ci_test)
    
    guessed_skeleton = ((guessed_dag != 0) | (guessed_dag.T != 0)).astype(int)
    topo_order = _topological_sort(guessed_dag)
    
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    level = -1
    
    while True:
        level += 1
        testable_edges = set(_get_testable_edges(current_graph, level))
        if not testable_edges:
            break
        
        ordered_edges = _expert_edge_ordering(testable_edges, guessed_skeleton)
        processed_pairs = set()
        
        for i, j in ordered_edges:
            if current_graph[i, j] == 0:
                continue
            
            pair_key = (min(i, j), max(i, j))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
            neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
            
            # Choose smaller neighborhood
            if len(neighbors_i) <= len(neighbors_j):
                if len(neighbors_i) >= level:
                    test_neighbors = neighbors_i
                    test_i, test_j = i, j
                elif len(neighbors_j) >= level:
                    test_neighbors = neighbors_j
                    test_i, test_j = j, i
                else:
                    continue
            else:
                if len(neighbors_j) >= level:
                    test_neighbors = neighbors_j
                    test_i, test_j = j, i
                elif len(neighbors_i) >= level:
                    test_neighbors = neighbors_i
                    test_i, test_j = i, j
                else:
                    continue
            
            conditioning_sets = list(combinations(test_neighbors, level))
            ordered_sets = _expert_subset_ordering(conditioning_sets, test_i, test_j, topo_order)
            
            for cond_set in ordered_sets:
                p_value = cit(test_i, test_j, list(cond_set))
                if p_value > alpha:
                    current_graph[pair_key[0], pair_key[1]] = 0
                    current_graph[pair_key[1], pair_key[0]] = 0
                    sepsets[pair_key] = sepsets[(pair_key[1], pair_key[0])] = cond_set
                    break
    
    return current_graph.astype(int)


def _topological_sort(dag: np.ndarray) -> list:
    """Extract topological ordering from DAG. Falls back to random if cycles exist."""
    n = dag.shape[0]
    in_degree = np.sum(dag, axis=0)
    topo_order = []
    queue = [i for i in range(n) if in_degree[i] == 0]
    
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        
        for neighbor in range(n):
            if dag[node, neighbor] != 0:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    # If cycle detected, return random ordering
    if len(topo_order) != n:
        return list(np.random.permutation(n))
    
    return topo_order


def _get_testable_edges(graph: np.ndarray, level: int) -> list:
    """Get ORDERED pairs of edges that can be tested at given level."""
    n = graph.shape[0]
    testable = []
    
    for i in range(n):
        for j in range(n):
            if i != j and graph[i, j] == 1:  # Edge exists
                neighbors_i = np.sum(graph[i, :]) - 1  # i's neighbors excluding j
                if neighbors_i >= level:
                    testable.append((i, j))  # Can test from i's perspective
    
    return testable


def _expert_edge_ordering(edges: list, guessed_skeleton: np.ndarray) -> list:
    """Order edges: predicted non-edges first, then predicted edges.
    Keep (i,j) and (j,i) together as units."""
    
    # Group edges by unordered pair
    edge_pairs = {}
    for i, j in edges:
        key = (min(i, j), max(i, j))
        if key not in edge_pairs:
            edge_pairs[key] = []
        edge_pairs[key].append((i, j))
    
    false_pairs = []
    true_pairs = []
    
    for (i, j), directions in edge_pairs.items():
        if guessed_skeleton[i, j] == 0:  # Predicted non-edge
            false_pairs.append(directions)
        else:  # Predicted edge
            true_pairs.append(directions)
    
    # Shuffle pair groups
    np.random.shuffle(false_pairs)
    np.random.shuffle(true_pairs)
    
    # Flatten, with random order within each pair
    result = []
    for directions in false_pairs + true_pairs:
        np.random.shuffle(directions)  # Randomize (i,j) vs (j,i) order
        result.extend(directions)
    
    return result


def _expert_subset_ordering(conditioning_sets: list, i: int, j: int, topo_order: list) -> list:
    """Order conditioning sets based on topological ordering."""
    pos_i = topo_order.index(i) if i in topo_order else len(topo_order)
    pos_j = topo_order.index(j) if j in topo_order else len(topo_order)
    max_pos = max(pos_i, pos_j)
    
    b1_sets = []  # No variable after both i,j in ordering
    b2_sets = []  # Has variable after both i,j
    
    for cond_set in conditioning_sets:
        has_descendant = False
        for var in cond_set:
            var_pos = topo_order.index(var) if var in topo_order else len(topo_order)
            if var_pos > max_pos:
                has_descendant = True
                break
        
        if has_descendant:
            b2_sets.append(cond_set)
        else:
            b1_sets.append(cond_set)
    
    # Random permutation of each group
    np.random.shuffle(b1_sets)
    np.random.shuffle(b2_sets)
    
    return b1_sets + b2_sets


def run_pc_guess_expert(
    data: np.ndarray,
    true_dag: np.ndarray,
    p_acc_edge_true: float,
    p_acc_edge_false: float,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run PC algorithm with expert guidance using single-direction testing
    
    Args:
        data: Data matrix (n_samples, n_features)
        true_dag: True DAG (hidden from algorithm, only used for oracle)
        p_acc_edge_true: Probability of correctly identifying true edges
        p_acc_edge_false: Probability of correctly identifying false edges
        p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
        p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Convert true DAG to NetworkX graph for d-separation queries
    dag_graph = nx.from_numpy_array(true_dag, create_using=nx.DiGraph)
    
    # Create probabilistic skeleton based on true DAG
    true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
    probabilistic_skeleton = _create_probabilistic_skeleton(
        true_skeleton, p_acc_edge_true, p_acc_edge_false
    )
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    level = -1

    while True:
        level += 1
        testable_edges = set(_get_testable_edges(current_graph, level))
        
        if not testable_edges:
            break
        
        ordered_edges = _expert_edge_ordering(testable_edges, probabilistic_skeleton)
        processed_pairs = set()
        
        for i, j in ordered_edges:
            # Skip if edge already removed
            if current_graph[i, j] == 0:
                continue
            
            # Check if we've already processed this unordered pair
            pair_key = (min(i, j), max(i, j))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Get current neighborhood sizes
            neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
            neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
            
            # Determine which direction to test from (prefer smaller neighborhood)
            if len(neighbors_i) <= len(neighbors_j):
                # Try i first (smaller or equal)
                if len(neighbors_i) >= level:
                    test_neighbors = neighbors_i
                    test_i, test_j = i, j
                elif len(neighbors_j) >= level:
                    # Fallback to j
                    test_neighbors = neighbors_j
                    test_i, test_j = j, i
                else:
                    # Neither has enough neighbors
                    continue
            else:
                # Try j first (smaller)
                if len(neighbors_j) >= level:
                    test_neighbors = neighbors_j
                    test_i, test_j = j, i
                elif len(neighbors_i) >= level:
                    # Fallback to i
                    test_neighbors = neighbors_i
                    test_i, test_j = i, j
                else:
                    # Neither has enough neighbors
                    continue
            
            # Test using selected neighborhood
            conditioning_sets = list(combinations(test_neighbors, level))
            
            # Apply probabilistic subset ordering
            ordered_sets = _probabilistic_subset_ordering(
                conditioning_sets, test_i, test_j, dag_graph, p_acc_subset_MD, p_acc_subset_NMD
            )
            
            for cond_set in ordered_sets:
                p_value = cit(test_i, test_j, list(cond_set))
                if p_value > alpha:
                    # Remove edge using original pair_key for consistency
                    current_graph[pair_key[0], pair_key[1]] = 0
                    current_graph[pair_key[1], pair_key[0]] = 0
                    sepsets[pair_key] = sepsets[(pair_key[1], pair_key[0])] = cond_set
                    break
    
    return current_graph.astype(int)


def run_pc_guess_expert_twice(
    data: np.ndarray,
    true_dag: np.ndarray,
    p_acc_edge_true: float,
    p_acc_edge_false: float,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run PC algorithm with expert guidance, testing both neighbor sets
    
    Args:
        data: Data matrix (n_samples, n_features)
        true_dag: True DAG (hidden from algorithm, only used for oracle)
        p_acc_edge_true: Probability of correctly identifying true edges
        p_acc_edge_false: Probability of correctly identifying false edges
        p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
        p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Create probabilistic skeleton based on true DAG
    true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
    probabilistic_skeleton = _create_probabilistic_skeleton(
        true_skeleton, p_acc_edge_true, p_acc_edge_false
    )
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    level = -1

    while True:
        level += 1
        testable_edges = set(_get_testable_edges(current_graph, level))
        
        if not testable_edges:
            break
        
        ordered_edges = _expert_edge_ordering(testable_edges, probabilistic_skeleton)
        processed_pairs = set()
        
        for i, j in ordered_edges:
            # Skip if edge already removed
            if current_graph[i, j] == 0:
                continue
            
            # Check if we've already processed this unordered pair
            pair_key = (min(i, j), max(i, j))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Get current neighborhood sizes
            neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
            neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
            
            # Determine which neighbor sets are large enough
            valid_neighbors = []
            if len(neighbors_i) >= level:
                valid_neighbors.append((neighbors_i, i, j))
            if len(neighbors_j) >= level:
                valid_neighbors.append((neighbors_j, j, i))
            
            # If no valid neighbor sets, skip this edge
            if not valid_neighbors:
                continue
            
            # Randomly shuffle the order of valid neighbor sets
            np.random.shuffle(valid_neighbors)
            
            # Test each valid neighbor set until independence is found
            edge_removed = False
            for test_neighbors, test_i, test_j in valid_neighbors:
                if edge_removed:  # Break if edge already removed
                    break
                
                # Test using selected neighborhood
                conditioning_sets = list(combinations(test_neighbors, level))
                
                # Apply probabilistic subset ordering
                ordered_sets = _probabilistic_subset_ordering(
                    conditioning_sets, test_i, test_j, true_dag, p_acc_subset_MD, p_acc_subset_NMD
                )
                
                for cond_set in ordered_sets:
                    p_value = cit(test_i, test_j, list(cond_set))
                    if p_value > alpha:
                        # Remove edge using original pair_key for consistency
                        current_graph[pair_key[0], pair_key[1]] = 0
                        current_graph[pair_key[1], pair_key[0]] = 0
                        sepsets[pair_key] = sepsets[(pair_key[1], pair_key[0])] = cond_set
                        edge_removed = True
                        break  # Break from conditioning sets loop
    
    return current_graph.astype(int)


# def run_pc_guess_expert(
#     data: np.ndarray,
#     true_dag: np.ndarray,
#     p_acc_edge_true: float,
#     p_acc_edge_false: float,
#     p_acc_subset_MD: float,
#     p_acc_subset_NMD: float,
#     alpha: float = 0.01,
#     ci_test: str = "fisherz"
# ) -> np.ndarray:
#     """
#     Run PC algorithm with expert guidance
    
#     Args:
#         data: Data matrix (n_samples, n_features)
#         true_dag: True DAG (hidden from algorithm, only used for oracle)
#         p_acc_edge_true: Probability of correctly identifying true edges
#         p_acc_edge_false: Probability of correctly identifying false edges
#         p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
#         p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
#         alpha: Significance level for independence tests
#         ci_test: Independence test method
    
#     Returns:
#         Skeleton adjacency matrix (undirected)
#     """
#     n_vars = data.shape[1]
    
#     # Initialize CIT object for independence testing
#     cit = CIT(data, ci_test)
    
#     # Create probabilistic skeleton based on true DAG
#     true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
#     probabilistic_skeleton = _create_probabilistic_skeleton(
#         true_skeleton, p_acc_edge_true, p_acc_edge_false
#     )
    
#     # Initialize complete undirected graph
#     current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
#     sepsets = {}
    
#     # Set level
#     level = -1

#     # Ahh I should really use a set to store all testable edges, remove edges from here each time, and just check if set is empty.
#     # And I should update that set itself, not recall get_testable_edges everytime.

#     while True:
#         # Update level
#         level += 1

#         # Get testable edges at this level
#         testable_edges = set(_get_testable_edges(current_graph, level))

#         # If no more testable_edges, terminate algorithm
#         if not testable_edges:
#             break
        
#         # Apply expert-based edge ordering using probabilistic skeleton
#         ordered_edges = _expert_edge_ordering(testable_edges, probabilistic_skeleton)
        
#         for i, j in ordered_edges:
#             # If edge already removed, no need to check
#             if current_graph[i, j] == 0:  
#                 continue

#             # Get conditioning sets of size 'level' from i's neighbors
#             neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]

#             # If edge is no longer testable due to earlier tests, skip
#             if len(neighbors_i) < level:
#                 continue
                
#             conditioning_sets = list(combinations(neighbors_i, level))
            
#             # Apply probabilistic subset ordering
#             ordered_sets = _probabilistic_subset_ordering(
#                 conditioning_sets, i, j, true_dag, p_acc_subset_MD, p_acc_subset_NMD
#             )
            
#             # Loop through subsets
#             for cond_set in ordered_sets:
#                 p_value = cit(i, j, list(cond_set))
#                 # If CIT conditioned on subset fails to reject independence, remove edge and stop looping
#                 if p_value > alpha:
#                     current_graph[i, j] = current_graph[j, i] = 0
#                     sepsets[(i, j)] = sepsets[(j, i)] = cond_set
#                     break
    
    
#     return current_graph.astype(int)

# def run_pc_guess_expert(
#     data: np.ndarray,
#     true_dag: np.ndarray,
#     p_acc_edge_true: float,
#     p_acc_edge_false: float,
#     p_acc_subset_MD: float,
#     p_acc_subset_NMD: float,
#     alpha: float = 0.01
# ) -> np.ndarray:
#     """
#     Run PC algorithm with expert guidance
    
#     Args:
#         data: Data matrix (n_samples, n_features)
#         true_dag: True DAG (hidden from algorithm, only used for oracle)
#         p_acc_edge_true: Probability of correctly identifying true edges
#         p_acc_edge_false: Probability of correctly identifying false edges
#         p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
#         p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
#         alpha: Significance level for independence tests
    
#     Returns:
#         Skeleton adjacency matrix (undirected)
#     """
#     n_vars = data.shape[1]
    
#     # Initialize CIT object for independence testing
#     cit = CIT(data, "fisherz")
    
#     # Create probabilistic skeleton based on true DAG
#     true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
#     probabilistic_skeleton = _create_probabilistic_skeleton(
#         true_skeleton, p_acc_edge_true, p_acc_edge_false
#     )
    
#     # Initialize complete undirected graph
#     current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
#     sepsets = {}
    
#     # Set level
#     level = 0

#     # Set a Flag to check if there are testable edges left.
#     testable_edges_left = True

#     # Ahh I should really use a set to store all testable edges, remove edges from here each time, and just check if set is empty.
#     # And I should update that set itself, not recall get_testable_edges everytime.

    
#     while True:
#         # Get testable edges at this level
#         testable_edges = _get_testable_edges(current_graph, level)
#         # Set Flag according to whether there are testable edges left
#         if not testable_edges:
#             testable_edges_left = False
#         # Terminate algorithm if no testable edges left
#         if not testable_edges_left:
#             break
        
#         # Apply expert-based edge ordering using probabilistic skeleton
#         ordered_edges = _expert_edge_ordering(testable_edges, probabilistic_skeleton)
        
#         for i, j in ordered_edges:
#             # check if edge already removed
#             if current_graph[i, j] == 0:  # Edge already removed
#                 continue
#             # Get list of testable edges at this level
#             testable_edges = _get_testable_edges(current_graph, level)
#             # Check if testable_edges is empty, halt current loop 
#             if not testable_edges:
#                 testable_edges_left = False
#                 break
#             # Check if edge i,j, is in testable edges, if not continue to next edge
#             if (i,j) not in testable_edges:
#                 continue

#             # Get conditioning sets of size 'level' from i's neighbors
#             neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
            
#             #
#             if len(neighbors_i) < level:
#                 continue
                
#             conditioning_sets = list(combinations(neighbors_i, level))
            
#             # Apply probabilistic subset ordering
#             ordered_sets = _probabilistic_subset_ordering(
#                 conditioning_sets, i, j, true_dag, p_acc_subset_MD, p_acc_subset_NMD
#             )
            
#             for cond_set in ordered_sets:
#                 p_value = cit(i, j, list(cond_set))
#                 if p_value > alpha:
#                     # Remove edge
#                     current_graph[i, j] = current_graph[j, i] = 0
#                     sepsets[(i, j)] = sepsets[(j, i)] = cond_set
#                     break
        
#         level += 1
    
#     return current_graph.astype(int)


def _sample_hard_constraints(
    probabilistic_skeleton: np.ndarray,
    constraint_fraction: float
) -> tuple:
    """
    Sample edges to use as hard constraints.
    
    Args:
        probabilistic_skeleton: Predicted skeleton from expert
        constraint_fraction: Fraction of edges to constrain (0-1)
    
    Returns:
        Tuple of (constrained_true_edges, constrained_false_edges)
        Each is a set of unordered pairs (i,j) where i < j
    """
    n = probabilistic_skeleton.shape[0]
    
    # Collect all unordered pairs
    predicted_edges = []
    predicted_non_edges = []
    
    for i in range(n):
        for j in range(i+1, n):  # Upper triangle only
            if probabilistic_skeleton[i, j] == 1:
                predicted_edges.append((i, j))
            else:
                predicted_non_edges.append((i, j))
    
    # Sample constraint_fraction from each category
    n_constrained_edges = int(len(predicted_edges) * constraint_fraction)
    n_constrained_non_edges = int(len(predicted_non_edges) * constraint_fraction)
    
    # Randomly sample
    constrained_true_edges = set()
    if n_constrained_edges > 0 and len(predicted_edges) > 0:
        sampled_indices = np.random.choice(len(predicted_edges), n_constrained_edges, replace=False)
        constrained_true_edges = set(predicted_edges[i] for i in sampled_indices)
    
    constrained_false_edges = set()
    if n_constrained_non_edges > 0 and len(predicted_non_edges) > 0:
        sampled_indices = np.random.choice(len(predicted_non_edges), n_constrained_non_edges, replace=False)
        constrained_false_edges = set(predicted_non_edges[i] for i in sampled_indices)
    
    return constrained_true_edges, constrained_false_edges


def run_pc_hc_guess_expert(
    data: np.ndarray,
    true_dag: np.ndarray,
    p_acc_edge_true: float,
    p_acc_edge_false: float,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float,
    constraint_fraction: float = 0.3,
    use_random_guidance: bool = False,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run PC algorithm with hard constraints and expert guidance.
    
    This is PC-Guess-Expert with added hard constraints: a fraction of expert predictions
    are enforced as fixed background knowledge, while remaining predictions guide test ordering.
    
    Args:
        data: Data matrix (n_samples, n_features)
        true_dag: True DAG (hidden from algorithm, only used for oracle)
        p_acc_edge_true: Probability of correctly identifying true edges
        p_acc_edge_false: Probability of correctly identifying false edges
        p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
        p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
        constraint_fraction: Fraction of predictions to use as hard constraints (0-1)
        use_random_guidance: If True, use random guidance (0.5/0.5 accuracy) for non-constrained edges
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Convert true DAG to NetworkX graph for d-separation queries
    dag_graph = nx.from_numpy_array(true_dag, create_using=nx.DiGraph)
    
    # Create probabilistic skeleton based on true DAG
    true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
    probabilistic_skeleton = _create_probabilistic_skeleton(
        true_skeleton, p_acc_edge_true, p_acc_edge_false
    )
    
    # Sample hard constraints from probabilistic skeleton
    constrained_true_edges, constrained_false_edges = _sample_hard_constraints(
        probabilistic_skeleton, constraint_fraction
    )
    
    # Create guidance skeleton
    if use_random_guidance:
        # Use random skeleton for guidance (0.5/0.5 accuracy)
        guidance_skeleton = _create_probabilistic_skeleton(true_skeleton, 0.5, 0.5)
    else:
        # Use expert skeleton for guidance
        guidance_skeleton = probabilistic_skeleton.copy()
    
    # Remove constrained edges from guidance skeleton
    for i, j in constrained_true_edges:
        guidance_skeleton[i, j] = guidance_skeleton[j, i] = 0
    for i, j in constrained_false_edges:
        guidance_skeleton[i, j] = guidance_skeleton[j, i] = 0
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    # Apply hard constraints: remove constrained false edges immediately
    for i, j in constrained_false_edges:
        current_graph[i, j] = current_graph[j, i] = 0
    
    level = -1

    while True:
        level += 1
        testable_edges = set(_get_testable_edges(current_graph, level))
        
        # Filter out hard-constrained true edges (they should never be tested)
        testable_edges = {
            (i, j) for (i, j) in testable_edges 
            if (min(i, j), max(i, j)) not in constrained_true_edges
        }
        
        if not testable_edges:
            break
        
        # Use guidance skeleton for ordering (excludes constrained edges)
        ordered_edges = _expert_edge_ordering(testable_edges, guidance_skeleton)
        processed_pairs = set()
        
        for i, j in ordered_edges:
            # Skip if edge already removed
            if current_graph[i, j] == 0:
                continue
            
            # Check if we've already processed this unordered pair
            pair_key = (min(i, j), max(i, j))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Skip if this edge is a hard constraint (protected)
            if pair_key in constrained_true_edges:
                continue
            
            # Get current neighborhood sizes
            neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
            neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
            
            # Determine which direction to test from (prefer smaller neighborhood)
            if len(neighbors_i) <= len(neighbors_j):
                # Try i first (smaller or equal)
                if len(neighbors_i) >= level:
                    test_neighbors = neighbors_i
                    test_i, test_j = i, j
                elif len(neighbors_j) >= level:
                    # Fallback to j
                    test_neighbors = neighbors_j
                    test_i, test_j = j, i
                else:
                    # Neither has enough neighbors
                    continue
            else:
                # Try j first (smaller)
                if len(neighbors_j) >= level:
                    test_neighbors = neighbors_j
                    test_i, test_j = j, i
                elif len(neighbors_i) >= level:
                    # Fallback to i
                    test_neighbors = neighbors_i
                    test_i, test_j = i, j
                else:
                    # Neither has enough neighbors
                    continue
            
            # Test using selected neighborhood
            conditioning_sets = list(combinations(test_neighbors, level))
            
            # Apply probabilistic subset ordering
            ordered_sets = _probabilistic_subset_ordering(
                conditioning_sets, test_i, test_j, dag_graph, p_acc_subset_MD, p_acc_subset_NMD
            )
            
            for cond_set in ordered_sets:
                p_value = cit(test_i, test_j, list(cond_set))
                if p_value > alpha:
                    # Remove edge (not protected by hard constraint)
                    current_graph[pair_key[0], pair_key[1]] = 0
                    current_graph[pair_key[1], pair_key[0]] = 0
                    sepsets[pair_key] = sepsets[(pair_key[1], pair_key[0])] = cond_set
                    break
    
    return current_graph.astype(int)


def _create_probabilistic_skeleton(true_skeleton: np.ndarray, p_acc_edge_true: float, p_acc_edge_false: float) -> np.ndarray:
    """
    Create a skeleton where edges are classified with specified accuracy.
    """
    n = true_skeleton.shape[0]
    prob_skeleton = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):  # Upper triangle only
            if true_skeleton[i, j] == 1:
                # True edge - classify correctly with p_acc_edge_true
                if np.random.random() < p_acc_edge_true:
                    prob_skeleton[i, j] = prob_skeleton[j, i] = 1
                # else: incorrectly classify as non-edge (leave as 0)
            else:
                # False edge - classify correctly with p_acc_edge_false
                if np.random.random() < p_acc_edge_false:
                    # Correctly classify as non-edge (leave as 0)
                    pass
                else:
                    # Incorrectly classify as edge
                    prob_skeleton[i, j] = prob_skeleton[j, i] = 1
    
    return prob_skeleton.astype(int)


def _probabilistic_subset_ordering(
    conditioning_sets: list,
    i: int,
    j: int,
    dag_graph: nx.DiGraph,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float
) -> list:
    """
    Order conditioning sets based on probabilistic classification of d-separation.
    """
    b1_sets = []  # Predicted: no mutual descendant (d-separating)
    b2_sets = []  # Predicted: has mutual descendant (not d-separating)
    
    for cond_set in conditioning_sets:
        # Check true status: is this set d-separating?
        # has_md = _contains_mutual_descendant(i, j, cond_set, true_dag)
        has_md = not _is_dsep_set(i, j, cond_set, dag_graph)
        
        # Apply probabilistic classification
        if has_md:
            # True MD - classify correctly with p_acc_subset_MD
            if np.random.random() < p_acc_subset_MD:
                b2_sets.append(cond_set)  # Correct: placed in MD bucket
            else:
                b1_sets.append(cond_set)  # Incorrect: placed in no-MD bucket
        else:
            # No MD - classify correctly with p_acc_subset_NMD
            if np.random.random() < p_acc_subset_NMD:
                b1_sets.append(cond_set)  # Correct: placed in no-MD bucket
            else:
                b2_sets.append(cond_set)  # Incorrect: placed in MD bucket
    
    # Shuffle within each group
    np.random.shuffle(b1_sets)
    np.random.shuffle(b2_sets)
    
    return b1_sets + b2_sets


def _contains_mutual_descendant(i: int, j: int, cond_set: tuple, true_dag: np.ndarray) -> bool:
    """
    Check if conditioning set contains a mutual descendant of both i and j.
    """
    descendants_i = _get_descendants(i, true_dag)
    descendants_j = _get_descendants(j, true_dag)
    mutual_descendants = descendants_i.intersection(descendants_j)
    
    for var in cond_set:
        if var in mutual_descendants:
            return True
    return False


def _is_dsep_set(i: int, j: int, cond_set: tuple, dag_graph: nx.DiGraph) -> bool:
    """Check if conditioning set d-separates i and j in the DAG."""
    # return nx.is_d_separator(dag_graph, {i}, {j}, set(cond_set))
    # return is_d_separator(dag_graph, {i}, {j}, set(cond_set))
    return nx.d_separated(dag_graph, {i}, {j}, set(cond_set))


def _get_descendants(node: int, dag: np.ndarray) -> set:
    """Get all descendants of a node in the DAG using BFS."""
    descendants = set()
    queue = [node]
    visited = set([node])
    
    while queue:
        current = queue.pop(0)
        for child in range(dag.shape[0]):
            if dag[current, child] != 0 and child not in visited:
                descendants.add(child)
                queue.append(child)
                visited.add(child)
    
    return descendants


def run_sgs_guess_expert(
    data: np.ndarray,
    true_dag: np.ndarray,
    p_acc_edge_true: float,
    p_acc_edge_false: float,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run SGS algorithm with expert guidance
    
    Args:
        data: Data matrix (n_samples, n_features)
        true_dag: True DAG (hidden from algorithm, only used for oracle)
        p_acc_edge_true: Probability of correctly identifying true edges
        p_acc_edge_false: Probability of correctly identifying false edges
        p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
        p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Convert true DAG to NetworkX graph for d-separation queries
    dag_graph = nx.from_numpy_array(true_dag, create_using=nx.DiGraph)
    
    # Create probabilistic skeleton based on true DAG
    true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
    probabilistic_skeleton = _create_probabilistic_skeleton(
        true_skeleton, p_acc_edge_true, p_acc_edge_false
    )
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    # Get ALL edges from complete graph as unordered pairs
    all_edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
    
    # Apply expert-based edge ordering to ALL edges upfront
    ordered_edges = _expert_edge_ordering_sgs(all_edges, probabilistic_skeleton)
    
    # Main SGS loop: process each edge in predetermined order
    for i, j in ordered_edges:
        if current_graph[i, j] == 0:  # Edge already removed
            continue
        
        # Get current neighbors (SGS uses smaller adjacency set)
        neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
        neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
        
        # Pool of possible conditioning variables (smaller set)
        S_pool = neighbors_i if len(neighbors_i) <= len(neighbors_j) else neighbors_j
        
        # Test all subset sizes from 0 to len(S_pool), starting from lowest to highest.
        edge_removed = False
        for subset_size in range(len(S_pool) + 1):
            ## HMMMMM is there a bug here? not sure if its actually testing marginal dependence, we should see (but i think it is...)
            conditioning_sets = list(combinations(S_pool, subset_size))
            
            # Order subsets using expert guidance
            ordered_sets = _probabilistic_subset_ordering(
                conditioning_sets, i, j, dag_graph, 
                p_acc_subset_MD, p_acc_subset_NMD
            )
            
            for cond_set in ordered_sets:
                p_value = cit(i, j, list(cond_set))
                if p_value > alpha:
                    # Remove edge immediately (edge-wise deletion)
                    current_graph[i, j] = current_graph[j, i] = 0
                    sepsets[(i, j)] = sepsets[(j, i)] = cond_set
                    edge_removed = True
                    break  # Move to next edge
            
            if edge_removed:  # Edge removed, stop testing larger subsets
                break
    
    return current_graph.astype(int)


def run_sgs_hc_guess_expert(
    data: np.ndarray,
    true_dag: np.ndarray,
    p_acc_edge_true: float,
    p_acc_edge_false: float,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float,
    constraint_fraction: float = 0.3,
    use_random_guidance: bool = False,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run SGS algorithm with hard constraints and expert guidance.
    
    This is SGS-Guess-Expert with added hard constraints: a fraction of expert predictions
    are enforced as fixed background knowledge, while remaining predictions guide test ordering.
    
    Args:
        data: Data matrix (n_samples, n_features)
        true_dag: True DAG (hidden from algorithm, only used for oracle)
        p_acc_edge_true: Probability of correctly identifying true edges
        p_acc_edge_false: Probability of correctly identifying false edges
        p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
        p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
        constraint_fraction: Fraction of predictions to use as hard constraints (0-1)
        use_random_guidance: If True, use random guidance (0.5/0.5 accuracy) for non-constrained edges
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Convert true DAG to NetworkX graph for d-separation queries
    dag_graph = nx.from_numpy_array(true_dag, create_using=nx.DiGraph)
    
    # Create probabilistic skeleton based on true DAG
    true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
    probabilistic_skeleton = _create_probabilistic_skeleton(
        true_skeleton, p_acc_edge_true, p_acc_edge_false
    )
    
    # Sample hard constraints from probabilistic skeleton
    constrained_true_edges, constrained_false_edges = _sample_hard_constraints(
        probabilistic_skeleton, constraint_fraction
    )
    
    # Create guidance skeleton
    if use_random_guidance:
        # Use random skeleton for guidance (0.5/0.5 accuracy)
        guidance_skeleton = _create_probabilistic_skeleton(true_skeleton, 0.5, 0.5)
    else:
        # Use expert skeleton for guidance
        guidance_skeleton = probabilistic_skeleton.copy()
    
    # Remove constrained edges from guidance skeleton
    for i, j in constrained_true_edges:
        guidance_skeleton[i, j] = guidance_skeleton[j, i] = 0
    for i, j in constrained_false_edges:
        guidance_skeleton[i, j] = guidance_skeleton[j, i] = 0
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    # Apply hard constraints: remove constrained false edges immediately
    for i, j in constrained_false_edges:
        current_graph[i, j] = current_graph[j, i] = 0
    
    # Get ALL edges from complete graph as unordered pairs
    all_edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
    
    # Filter out hard-constrained true edges (they should never be tested)
    all_edges = [(i, j) for (i, j) in all_edges if (i, j) not in constrained_true_edges]
    
    # Apply expert-based edge ordering using guidance skeleton
    ordered_edges = _expert_edge_ordering_sgs(all_edges, guidance_skeleton)
    
    # Main SGS loop: process each edge in predetermined order
    for i, j in ordered_edges:
        if current_graph[i, j] == 0:  # Edge already removed
            continue
        
        # Get current neighbors (SGS uses smaller adjacency set)
        neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
        neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
        
        # Pool of possible conditioning variables (smaller set)
        S_pool = neighbors_i if len(neighbors_i) <= len(neighbors_j) else neighbors_j
        
        # Test all subset sizes from 0 to len(S_pool)
        edge_removed = False
        for subset_size in range(len(S_pool) + 1):
            conditioning_sets = list(combinations(S_pool, subset_size))
            
            # Order subsets using expert guidance
            ordered_sets = _probabilistic_subset_ordering(
                conditioning_sets, i, j, dag_graph, 
                p_acc_subset_MD, p_acc_subset_NMD
            )
            
            for cond_set in ordered_sets:
                p_value = cit(i, j, list(cond_set))
                if p_value > alpha:
                    # Remove edge immediately (edge-wise deletion)
                    current_graph[i, j] = current_graph[j, i] = 0
                    sepsets[(i, j)] = sepsets[(j, i)] = cond_set
                    edge_removed = True
                    break  # Move to next edge
            
            if edge_removed:  # Edge removed, stop testing larger subsets
                break
    
    return current_graph.astype(int)


def run_sgs_guess_expert_twice(
    data: np.ndarray,
    true_dag: np.ndarray,
    p_acc_edge_true: float,
    p_acc_edge_false: float,
    p_acc_subset_MD: float,
    p_acc_subset_NMD: float,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run SGS algorithm with expert guidance, testing both neighbor sets
    
    Args:
        data: Data matrix (n_samples, n_features)
        true_dag: True DAG (hidden from algorithm, only used for oracle)
        p_acc_edge_true: Probability of correctly identifying true edges
        p_acc_edge_false: Probability of correctly identifying false edges
        p_acc_subset_MD: Probability of correctly identifying subsets with mutual descendants
        p_acc_subset_NMD: Probability of correctly identifying subsets without mutual descendants
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Create probabilistic skeleton based on true DAG
    true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
    probabilistic_skeleton = _create_probabilistic_skeleton(
        true_skeleton, p_acc_edge_true, p_acc_edge_false
    )
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    # Get ALL edges from complete graph as unordered pairs
    all_edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
    
    # Apply expert-based edge ordering to ALL edges upfront
    ordered_edges = _expert_edge_ordering_sgs(all_edges, probabilistic_skeleton)
    
    # Main SGS loop: process each edge in predetermined order
    for i, j in ordered_edges:
        if current_graph[i, j] == 0:  # Edge already removed
            continue
        
        # Get current neighbors
        neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
        neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
        
        # Randomly assign neighbor sets to s_pool_1 and s_pool_2
        if np.random.random() < 0.5:
            s_pool_1, s_pool_2 = neighbors_i, neighbors_j
        else:
            s_pool_1, s_pool_2 = neighbors_j, neighbors_i
        
        # Test both neighbor sets sequentially
        edge_removed = False
        for S_pool in [s_pool_1, s_pool_2]:
            if edge_removed:  # Break out if edge already removed
                break
                
            # Test all subset sizes from 0 to len(S_pool)
            for subset_size in range(len(S_pool) + 1):
                conditioning_sets = list(combinations(S_pool, subset_size))
                
                # Order subsets using expert guidance
                ordered_sets = _probabilistic_subset_ordering(
                    conditioning_sets, i, j, true_dag, 
                    p_acc_subset_MD, p_acc_subset_NMD
                )
                
                for cond_set in ordered_sets:
                    p_value = cit(i, j, list(cond_set))
                    if p_value > alpha:
                        # Remove edge immediately
                        current_graph[i, j] = current_graph[j, i] = 0
                        sepsets[(i, j)] = sepsets[(j, i)] = cond_set
                        edge_removed = True
                        break  # Break from cond_set loop
                
                if edge_removed:  # Break from subset_size loop
                    break
    
    return current_graph.astype(int)


def run_sgs_guess_dag(
    data: np.ndarray,
    guessed_dag: np.ndarray,
    alpha: float = 0.01,
    ci_test: str = "fisherz"
) -> np.ndarray:
    """
    Run SGS algorithm with expert-guided ordering based on guessed DAG
    
    Args:
        data: Data matrix (n_samples, n_features)
        guessed_dag: Guessed DAG adjacency matrix (n_features, n_features)
        alpha: Significance level for independence tests
        ci_test: Independence test method
    
    Returns:
        Skeleton adjacency matrix (undirected)
    """
    n_vars = data.shape[1]
    
    # Initialize CIT object for independence testing
    cit = CIT(data, ci_test)
    
    # Extract skeleton from guessed DAG
    guessed_skeleton = ((guessed_dag != 0) | (guessed_dag.T != 0)).astype(int)
    
    # Extract topological ordering from guessed DAG
    topo_order = _topological_sort(guessed_dag)
    
    # Initialize complete undirected graph
    current_graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sepsets = {}
    
    # Get ALL edges from complete graph as unordered pairs
    all_edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
    
    # Apply expert-based edge ordering using guessed skeleton
    ordered_edges = _expert_edge_ordering_sgs(all_edges, guessed_skeleton)
    
    # Main SGS loop: process each edge in predetermined order
    for i, j in ordered_edges:
        if current_graph[i, j] == 0:  # Edge already removed
            continue
        
        # Get current neighbors (SGS uses smaller adjacency set)
        neighbors_i = [k for k in range(n_vars) if current_graph[i, k] == 1 and k != j]
        neighbors_j = [k for k in range(n_vars) if current_graph[j, k] == 1 and k != i]
        
        # Pool of possible conditioning variables (smaller set)
        S_pool = neighbors_i if len(neighbors_i) <= len(neighbors_j) else neighbors_j
        
        # Test all subset sizes from 0 to len(S_pool)
        edge_removed = False
        for subset_size in range(len(S_pool) + 1):
            conditioning_sets = list(combinations(S_pool, subset_size))
            
            # Order subsets using expert guidance with topological ordering
            ordered_sets = _expert_subset_ordering(conditioning_sets, i, j, topo_order)
            
            for cond_set in ordered_sets:
                p_value = cit(i, j, list(cond_set))
                if p_value > alpha:
                    # Remove edge immediately (edge-wise deletion)
                    current_graph[i, j] = current_graph[j, i] = 0
                    sepsets[(i, j)] = sepsets[(j, i)] = cond_set
                    edge_removed = True
                    break  # Move to next edge
            
            if edge_removed:  # Edge removed, stop testing larger subsets
                break
    
    return current_graph.astype(int)


def _expert_edge_ordering_sgs(edges: list, guessed_skeleton: np.ndarray) -> list:
    """Order edges for SGS: predicted non-edges first, then predicted edges.
    Input edges are unordered pairs (i,j) where i < j."""
    
    false_edges = []  # Predicted non-edges
    true_edges = []   # Predicted edges
    
    for i, j in edges:
        if guessed_skeleton[i, j] == 0:  # Predicted non-edge
            false_edges.append((i, j))
        else:  # Predicted edge
            true_edges.append((i, j))
    
    # Shuffle within each group for randomization
    np.random.shuffle(false_edges)
    np.random.shuffle(true_edges)
    
    # Return non-edges first, then edges
    return false_edges + true_edges


def run_method(method_name: str, data: np.ndarray, **kwargs) -> tuple:
    """
    Wrapper to run specified method with runtime tracking
    
    Args:
        method_name: 'pc', 'stable_pc', 'pc_guess_dag', 'pc_guess_expert', 'pc_guess_expert_twice', 'pc_hc_guess_expert', 'sgs_guess_expert', 'sgs_hc_guess_expert', 'sgs_guess_expert_twice', or 'sgs_guess_dag'
        data: Input data
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (skeleton_graph, runtime_seconds)
    """
    alpha = kwargs.get('alpha', 0.01)
    ci_test = kwargs.get('ci_test', 'fisherz')
    
    start_time = time.time()
    
    if method_name == 'pc':
        skeleton = run_pc(data, alpha, ci_test)
    elif method_name == 'stable_pc':
        skeleton = run_stable_pc(data, alpha, ci_test)
    elif method_name == 'pc_guess_dag':
        guessed_dag = kwargs.get('guessed_dag')
        if guessed_dag is None:
            raise ValueError("pc_guess_dag requires 'guessed_dag' parameter")
        skeleton = run_pc_guess_dag(data, guessed_dag, alpha, ci_test)
    elif method_name == 'pc_guess_expert':
        true_dag = kwargs.get('true_dag')
        if true_dag is None:
            raise ValueError("pc_guess_expert requires 'true_dag' parameter")
        p_acc_edge_true = kwargs.get('p_acc_edge_true', 0.8)
        p_acc_edge_false = kwargs.get('p_acc_edge_false', 0.8)
        p_acc_subset_MD = kwargs.get('p_acc_subset_MD', 0.8)
        p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD', 0.8)
        # p_acc_edge_true = kwargs.get('p_acc_edge_true')
        # p_acc_edge_false = kwargs.get('p_acc_edge_false')
        # p_acc_subset_MD = kwargs.get('p_acc_subset_MD')
        # p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD')
       
        
        skeleton = run_pc_guess_expert(
            data, true_dag,
            p_acc_edge_true, p_acc_edge_false,
            p_acc_subset_MD, p_acc_subset_NMD,
            alpha, ci_test
        )
    elif method_name == 'pc_guess_expert_twice':
        true_dag = kwargs.get('true_dag')
        if true_dag is None:
            raise ValueError("pc_guess_expert_twice requires 'true_dag' parameter")
        p_acc_edge_true = kwargs.get('p_acc_edge_true', 0.8)
        p_acc_edge_false = kwargs.get('p_acc_edge_false', 0.8)
        p_acc_subset_MD = kwargs.get('p_acc_subset_MD', 0.8)
        p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD', 0.8)
        
        skeleton = run_pc_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true, p_acc_edge_false,
            p_acc_subset_MD, p_acc_subset_NMD,
            alpha, ci_test
        )
    elif method_name == 'sgs_guess_expert':
        true_dag = kwargs.get('true_dag')
        if true_dag is None:
            raise ValueError("sgs_guess_expert requires 'true_dag' parameter")
        p_acc_edge_true = kwargs.get('p_acc_edge_true', 0.8)
        p_acc_edge_false = kwargs.get('p_acc_edge_false', 0.8)
        p_acc_subset_MD = kwargs.get('p_acc_subset_MD', 0.8)
        p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD', 0.8)
        
        skeleton = run_sgs_guess_expert(
            data, true_dag,
            p_acc_edge_true, p_acc_edge_false,
            p_acc_subset_MD, p_acc_subset_NMD,
            alpha, ci_test
        )
    elif method_name == 'sgs_guess_expert_twice':
        true_dag = kwargs.get('true_dag')
        if true_dag is None:
            raise ValueError("sgs_guess_expert_twice requires 'true_dag' parameter")
        p_acc_edge_true = kwargs.get('p_acc_edge_true', 0.8)
        p_acc_edge_false = kwargs.get('p_acc_edge_false', 0.8)
        p_acc_subset_MD = kwargs.get('p_acc_subset_MD', 0.8)
        p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD', 0.8)
        
        skeleton = run_sgs_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true, p_acc_edge_false,
            p_acc_subset_MD, p_acc_subset_NMD,
            alpha, ci_test
        )
    elif method_name == 'sgs_guess_dag':
        guessed_dag = kwargs.get('guessed_dag')
        if guessed_dag is None:
            raise ValueError("sgs_guess_dag requires 'guessed_dag' parameter")
        skeleton = run_sgs_guess_dag(data, guessed_dag, alpha, ci_test)
    elif method_name == 'pc_hc_guess_expert':
        true_dag = kwargs.get('true_dag')
        if true_dag is None:
            raise ValueError("pc_hc_guess_expert requires 'true_dag' parameter")
        p_acc_edge_true = kwargs.get('p_acc_edge_true', 0.8)
        p_acc_edge_false = kwargs.get('p_acc_edge_false', 0.8)
        p_acc_subset_MD = kwargs.get('p_acc_subset_MD', 0.8)
        p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD', 0.8)
        constraint_fraction = kwargs.get('constraint_fraction', 0.3)
        use_random_guidance = kwargs.get('use_random_guidance', False)
        
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true, p_acc_edge_false,
            p_acc_subset_MD, p_acc_subset_NMD,
            constraint_fraction,
            use_random_guidance,
            alpha, ci_test
        )
    elif method_name == 'sgs_hc_guess_expert':
        true_dag = kwargs.get('true_dag')
        if true_dag is None:
            raise ValueError("sgs_hc_guess_expert requires 'true_dag' parameter")
        p_acc_edge_true = kwargs.get('p_acc_edge_true', 0.8)
        p_acc_edge_false = kwargs.get('p_acc_edge_false', 0.8)
        p_acc_subset_MD = kwargs.get('p_acc_subset_MD', 0.8)
        p_acc_subset_NMD = kwargs.get('p_acc_subset_NMD', 0.8)
        constraint_fraction = kwargs.get('constraint_fraction', 0.3)
        use_random_guidance = kwargs.get('use_random_guidance', False)
        
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true, p_acc_edge_false,
            p_acc_subset_MD, p_acc_subset_NMD,
            constraint_fraction,
            use_random_guidance,
            alpha, ci_test
        )
    else:
        raise ValueError(f"Unknown method: {method_name}. Use 'pc', 'stable_pc', 'pc_guess_dag', 'pc_guess_expert', 'pc_guess_expert_twice', 'pc_hc_guess_expert', 'sgs_guess_expert', 'sgs_hc_guess_expert', 'sgs_guess_expert_twice', or 'sgs_guess_dag'")
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return skeleton, runtime
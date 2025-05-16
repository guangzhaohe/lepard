import torch
import numpy as np
import fpsample
from knn_cuda import KNN


def get_knn(find_in: np.ndarray, find_for: np.ndarray, knn: int = 4):
    # both are N_X x 3 pcds
    knn = KNN(k=knn, transpose_mode=True)
    
    ref = torch.from_numpy(find_in).cuda()[None]
    query = torch.from_numpy(find_for).cuda()[None]
    dist, indx = knn(ref, query)
    return indx[0].cpu().numpy(), dist[0].cpu().numpy()  # N_find_for x knn


def build_deformation_graph(
    source_pcd,  # (N, 3) torch.Tensor
    num_nodes=256,
    k_edge=4,
    k_anchor=6,
    rbf_sigma=0.1
):
    """
    Constructs a deformation graph from a source point cloud.

    Returns:
        graph_data: dict with graph_nodes, graph_edges, graph_edges_weights, point_anchors, point_weights
    """
    # device = source_pcd.device
    N = source_pcd.shape[0]

    # 1. Farthest Point Sampling (FPS)
    def farthest_point_sampling(xyz, n_samples):
        return fpsample.bucket_fps_kdline_sampling(xyz, n_samples, h=5)

    fps_idx = farthest_point_sampling(source_pcd, num_nodes)
    graph_nodes = source_pcd[fps_idx]  # (num_nodes, 3)

    # 2. KNN graph edges on graph nodes
    graph_edges, graph_dist = get_knn(graph_nodes, graph_nodes, knn=k_edge + 1)
    graph_edges = graph_edges[:, 1:]  # Because the first one is itself
    graph_dist = graph_dist[:, 1:]

    # 3. Edge weights using RBF
    graph_edges_weights = np.exp(-graph_dist ** 2 / (2 * rbf_sigma ** 2))  # (E,)
    graph_edges_weights = graph_edges_weights / (graph_edges_weights.sum(axis=-1, keepdims=True) + 1e-8)

    # 4. Anchor assignment for all points
    anchors, anchor_dist = get_knn(graph_nodes, source_pcd, knn=k_anchor)  # here the anchors index is in graph_nodes space
    self_indices = anchor_dist[:, 0] < 1e-5
    anchors[self_indices, 0] = -1  # avoid self assignment
    for i in range(len(anchors)):  # move the placeholder to the back
        if anchors[i][0] == -1:
            anchors[i][:k_anchor - 1] = anchors[i][1:]
            anchor_dist[i][:k_anchor - 1] = anchor_dist[i][1:]
            anchors[i][-1] = -1
            anchor_dist[i][-1] = 1e10  # a large number

    # Anchor weights using RBF
    weights = np.exp(-anchor_dist / (2 * rbf_sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # normalize

    return {
        'graph_nodes': torch.from_numpy(graph_nodes),
        'graph_edges': torch.from_numpy(graph_edges),
        'graph_edges_weights': torch.from_numpy(graph_edges_weights),
        'point_anchors': torch.from_numpy(anchors).long(),
        'point_weights': torch.from_numpy(weights),
    }

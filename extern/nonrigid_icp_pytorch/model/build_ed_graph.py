from cvtb import vis
import numpy as np
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import fpsample
from knn_cuda import KNN


def get_knn(find_in, find_for, knn=4):
    # both are N_X x 3 pcds
    knn = KNN(k=knn, transpose_mode=True)
    ref = torch.from_numpy(find_in).cuda()[None]
    query = torch.from_numpy(find_for).cuda()[None]
    dist, indx = knn(ref, query)
    return indx[0].cpu().numpy(), \
        dist[0].cpu().numpy()  # N_find_for x knn


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
    device = source_pcd.device
    N = source_pcd.shape[0]

    # 1. Farthest Point Sampling (FPS)
    def farthest_point_sampling(xyz, n_samples):
        sampled_idx = fpsample.bucket_fps_kdline_sampling(xyz, n_samples, h=5)
        return sampled_idx

    fps_idx = farthest_point_sampling(source_pcd, num_nodes)
    graph_nodes = source_pcd[fps_idx]  # (num_nodes, 3)

    # 2. KNN graph edges on graph nodes
    graph_edges, node_dist = get_knn(graph_nodes, graph_nodes, knn=k_edge)

    # 3. Edge weights using RBF
    graph_edges_weights = np.exp(-node_dist ** 2 / (2 * rbf_sigma ** 2))  # (E,)
    graph_edges_weights = graph_edges_weights / (np.sum(graph_edges_weights, axis=-1, keepdims=True) + 1e-8)

    # 4. Anchor assignment for all points
    anchor_edges, anchor_dist = get_knn(graph_nodes, source_pcd, knn=k_anchor)
    anchors = anchor_edges

    # Anchor weights using RBF
    weights = np.exp(-anchor_dist / (2 * rbf_sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # normalize

    return {
        'graph_nodes': torch.from_numpy(graph_nodes),
        'graph_edges': torch.from_numpy(graph_edges),
        'graph_edges_weights': torch.from_numpy(graph_edges_weights),
        'point_anchors': torch.from_numpy(anchors),
        'point_weights': torch.from_numpy(weights),
    }


# data = np.load('dftmp.npy', allow_pickle=True).item()
# pcd = data['src_pcd']

# vis.pcd_static(pcd)

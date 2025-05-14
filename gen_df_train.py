import random
from typing import List, Dict

import torch
import numpy as np
from cvtb import vis

from knn_cuda import KNN


def procrustes_alignment(X, Y):
    assert X.shape == Y.shape, "Point clouds must have the same shape"
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    X0 = X - mu_X
    Y0 = Y - mu_Y
    H = Y0.T @ X0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_X - R @ mu_Y
    Y_aligned = (R @ Y.T).T + t
    return Y_aligned, R, t


def get_knn_in_t(find_in: np.ndarray, find_for: np.ndarray, knn: int = 4):
    # both are N_X x 3 pcds
    knn = KNN(k=knn, transpose_mode=True)
    
    ref = torch.from_numpy(find_in).cuda()[None]
    query = torch.from_numpy(find_for).cuda()[None]
    dist, indx = knn(ref, query)
    return indx[0].cpu().numpy()  # N_find_for x knn


def generate_repeated_array(total_numbers, repeat_time):
    return np.repeat(np.arange(total_numbers), repeat_time)


def generate_evaluation_labels(data: Dict):
    raise NotImplementedError()


def generate_training_labels(data: Dict, knn: int = 4):
    src_pcd = data['points_mesh']  # n_src, 3
    tgt_pcds = data['points']  # n_f, n_tgt, 3
    gt_tracks = data['tracks']  # n_f, n_src, 3
    
    # Randomly sample 1 frame
    n_f = tgt_pcds.shape[0]
    frame_id = random.randint(0, n_f - 1)
    
    tgt_pcd = tgt_pcds[frame_id]
    gt_track = gt_tracks[frame_id]
    
    # Do procrustes anal.
    _, rot, trans = procrustes_alignment(src_pcd, gt_track)

    # Calculate residual flow
    src_pcd_warp = src_pcd @ rot.T + trans
    s2t_flow = gt_track - src_pcd_warp
    
    # Get correspondence, for every point in target get the nearest ones (use knn) in the src
    src_indices = get_knn_in_t(src_pcd, tgt_pcd, knn=knn).flatten()
    tgt_indices = generate_repeated_array(len(tgt_pcd), repeat_time=knn)
    corr = np.stack([src_indices, tgt_indices]).transpose((1, 0))  # n_corr, 2

    # from cvtb import vis
    # vis.pcd(np.stack([src_pcd[corr[:, 0]], tgt_pcd[corr[:, 1]]]), fps=1)
    
    return {
        'rot': rot,
        'trans': trans,
        's2t_flow': s2t_flow,
        's_pc': src_pcd,
        't_pc': tgt_pcd,
        'correspondences': corr
    }


if __name__ == '__main__':
    data_path = 'dftmp.npy'
    data = np.load(data_path, allow_pickle=True).item()

    src_pcd = data['src_pcd']
    tgt_pcd = data['tgt_pcd']
    gt_tracks = data['gt_tracks']    

    # Match dataset format
    fake_batch = {
        'points_mesh': src_pcd,
        'points': tgt_pcd,
        'tracks': gt_tracks,
    }
    
    training_batch = generate_training_labels(fake_batch)
    breakpoint()

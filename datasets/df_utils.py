import os
import h5py
from os.path import join as osp
from typing import List, Literal
from multiprocessing import Pool

import cv2 
import torch
import noise
import trimesh
import pyrender
import fpsample
import numpy as np
import open3d as o3d
# import pytorch3d.ops as ops
from torch.utils import data
from scipy.spatial import KDTree
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


def vis_mesh(points: np.ndarray, faces: np.ndarray = None):
    # N, 3
    if faces is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
    else:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.visualization.draw_geometries([mesh])


def get_normals(points: np.ndarray, 
                faces: np.ndarray = None,
                estimate_knn: int = 15,
                outward: bool = True):
    # points: N, 3
    # faces: None or M, 3
    if faces is None:
        print('1')
        pcd = o3d.geometry.PointCloud()
        print('2')

        print('1')
        pcd.points = o3d.utility.Vector3dVector(points)
        print('3')
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=estimate_knn))
        
        print('4')
        normals = np.asarray(pcd.normals)
        print('5')

    else:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
    
    if outward:
        center = np.mean(points, axis=0)
        c2p = points - center[None]
        negdot = np.einsum('ij,ij->i', c2p, normals) < 0
        normals[negdot] *= -1

    return normals


def fps(pcd: np.ndarray, num_samples: int, method: Literal['fpsample', 'o3d'] = 'fpsample') -> np.ndarray:
    if method == 'fpsample':
        return pcd[fpsample.bucket_fps_kdline_sampling(pcd, num_samples, h=5)]
    elif method == 'o3d':  # p3d also here
        # pcd_o3d = o3d.geometry.PointCloud()
        # pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        # return np.asarray(pcd_o3d.farthest_point_down_sample(num_samples).points)
        pcd_t = torch.tensor(pcd)[None]
        return ops.sample_farthest_points(pcd_t, K=num_samples)[0].numpy()[0]


def noisify_pcd(pcd: np.ndarray, num: int):
    ret = np.concatenate([pcd, np.random.uniform(-1., 1., (num, 3))])
    np.random.shuffle(ret)
    return ret


# randomly choose between uniform and knn
def smart_sample(
        pcd: np.array,  # N, 3
        num_sample: int,
        uni_probability: float,
    ) -> np.array:  # list of indices

    lucky_number = np.random.uniform()

    if lucky_number < uni_probability:
        indices = np.random.choice(pcd.shape[0], 
                                   size=num_sample if pcd.shape[0] > num_sample else pcd.shape[0],
                                   replace=False)
    else:
        starting_index = np.random.randint(pcd.shape[0])
        tree = KDTree(pcd)
        _, indices = tree.query([pcd[starting_index]], k=num_sample - 1)
        indices = np.concatenate([indices.flatten(), np.array([starting_index])])

    np.random.shuffle(indices)
    return indices


def what_pcd_format_are_we_lookin_at(dir: str):
    # obj, ply, npy
    def is_it_in_there(files: List, fmat: str) -> bool:
        for f in files:
            if fmat in f: return True
        return False

    fs = os.listdir(dir)
    filter_format = lambda x: sorted([f for f in fs if x in f])

    if is_it_in_there(fs, '.obj'): return filter_format('.obj')
    elif is_it_in_there(fs, '.ply'): return filter_format('.ply')
    elif is_it_in_there(fs, '.npy'): return filter_format('.npy')
    else: return None


def load_obj(file_path):
    vertices = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))

    return np.array(vertices)


def open_pcd(file_path):  # return N, 3 pcd
    if '.obj' in file_path:
        return load_obj(file_path)
    if '.ply' in file_path:
        return np.asarray(o3d.io.read_point_cloud(file_path).points)
    if '.npy' in file_path:
        return np.load(file_path)
    raise TypeError(f'Extension format not supported for opening point cloud')


def generate_smooth_sequence(num: int):
    np.random.seed(0)
    coordinates = np.arange(num) / 1000 + 1000 * np.random.rand()
    perlin_noises = [noise.pnoise1(coord, octaves=4, persistence=0.5, lacunarity=2.0) for coord in coordinates]
    return np.array(perlin_noises)


def generate_rotation(num: int) -> np.ndarray:  # num, 4, 4
    angle_x = generate_smooth_sequence(num)
    angle_y = generate_smooth_sequence(num)
    angle_z = generate_smooth_sequence(num)

    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    stretch = lambda x: 2. * norm(x) * np.pi

    rot_x = stretch(angle_x)  # num,
    rot_y = stretch(angle_y)
    rot_z = stretch(angle_z)

    assert np.all(rot_x >= 0)
    assert np.all(rot_y >= 0)
    assert np.all(rot_z >= 0)

    mat_x = np.zeros((num, 3, 3))
    mat_y = np.zeros((num, 3, 3))
    mat_z = np.zeros((num, 3, 3))

    mat_x[:, 0, 0] = 1.
    mat_x[:, 1, 1] = np.cos(rot_x)
    mat_x[:, 1, 2] = -np.sin(rot_x)
    mat_x[:, 2, 1] = np.sin(rot_x)
    mat_x[:, 2, 2] = np.cos(rot_x)

    mat_y[:, 0, 0] = np.cos(rot_y)
    mat_y[:, 0, 2] = np.sin(rot_y)
    mat_y[:, 1, 1] = 1.
    mat_y[:, 2, 0] = -np.sin(rot_y)
    mat_y[:, 2, 2] = np.cos(rot_y)

    mat_z[:, 0, 0] = np.cos(rot_z)
    mat_z[:, 0, 1] = -np.sin(rot_z)
    mat_z[:, 1, 0] = np.sin(rot_z)
    mat_z[:, 1, 1] = np.cos(rot_z)
    mat_z[:, 2, 2] = 1.

    rot_mat = np.empty((num, 3, 3))
    for f in range(num):
        rot_mat[f] = mat_x[f] @ mat_y[f] @ mat_z[f]
    return rot_mat


def apply_random_rotation(pts: np.ndarray, num: int):  # from static points to a sequence of points
    pts = pts - np.mean(pts)
    rot_mat = generate_rotation(num)  # N, 4, 4

    N = pts.shape[0]
    ret = np.empty((num, N, 3))

    for f in range(num):
        ret[f] = pts @ rot_mat[f].T

    return ret  # F, N, 3


# from PointOdyssey: https://github.com/y-zheng18/point_odyssey
def farthest_point_sampling(
        p: np.ndarray,  # N, 3
        K: int,
    ) -> np.ndarray:  # K, 3
    """
    greedy farthest point sampling
    p: point cloud
    K: number of points to sample
    """

    farthest_point = np.zeros((K, 3))
    max_idx = np.random.randint(0, p.shape[0] -1)
    farthest_point[0] = p[max_idx]
    for i in range(1, K):
        pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
        distance = np.min(pairwise_distance, axis=1, keepdims=True)
        max_idx = np.argmax(distance)
        farthest_point[i] = p[max_idx]
    return farthest_point


def center_per_frame(vert: np.array, transform: List = None):
    # f, p, 3
    f, p = vert.shape[:2]
    mean = np.mean(vert, axis=1, keepdims=True)  # f, 1, 3
    if transform is not None:
        tran_mat = np.eye(4)[None].repeat(f, axis=0)  # nf, 4, 4
        tran_mat[:, :3, 3] = mean.squeeze()
        transform.append(tran_mat.transpose((0, 2, 1)))
    return vert - mean


def rand_scaling(vert: np.ndarray,
                 rand_scaling_range: List[float],
                 transform: List = None):
    rand_scaling = np.random.rand(3) * (rand_scaling_range[1] - rand_scaling_range[0]) + rand_scaling_range[0]
    vert *= rand_scaling
    scale_mat = np.eye(4)[None]
    scale_mat[:, :3, :3] /= rand_scaling
    if transform is not None:
        transform.append(scale_mat.transpose((0, 2, 1)))
    return vert


def rand_rotate_point_cloud_z(point_cloud, transform: List = None):
    random_rotation_angle = np.random.uniform(0, 360)
    angle_radians = np.radians(random_rotation_angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0.],
        [np.sin(angle_radians),  np.cos(angle_radians), 0.],
        [                   0.,                     0., 1.]
    ])
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    if transform is not None:
        rot_mat = np.eye(4)[None]
        rot_mat[:, :3, :3] = rotation_matrix
        transform.append(rot_mat)
    return rotated_point_cloud


def rand_rotate_point_cloud(gt_verts, transform: List = None):
    rot_matrix = Rotation.random().as_matrix().T
    gt_verts = gt_verts @ rot_matrix
    rot_mat = np.eye(4)[None]
    rot_mat[:, :3, :3] = rot_matrix.T
    if transform is not None:
        transform.append(rot_mat)
    return gt_verts


def normalize_pcd_seq(vert: np.ndarray, padding: float, transform: List = None):
    # verts: nf, nv, 3

    abs_vert = np.abs(vert)  # nf, nv, 3
    abs_max = np.max(abs_vert)
    ratio = (1. - padding) / abs_max
    vert *= ratio

    if transform is not None:
        tran_mat = np.eye(4)[None]
        tran_mat[:, :3, :3] /= ratio
        transform.append(tran_mat.transpose((0, 2, 1)))

    return vert


# translation -> rotation -> scaling/normalization
def proc_frame_points(vert: np.ndarray,
                      ret_reverse_transform: bool,

                      use_rand_rotation: bool,
                      rand_rotate_z_only: bool,

                      use_rand_padding: bool,
                      rand_padding_range: List[float],
                      ):
    if ret_reverse_transform:
        reverse_transform = []
    else:
        reverse_transform = None

    # translation
    vert = center_per_frame(vert, transform=reverse_transform)

    # rotation
    if use_rand_rotation:
        if rand_rotate_z_only:
            vert = rand_rotate_point_cloud_z(vert, transform=reverse_transform)
        else:
            vert = rand_rotate_point_cloud(vert, transform=reverse_transform)

    # scaling & normalization
    if use_rand_padding:
        rand_padding = np.random.rand() * (rand_padding_range[1] - rand_padding_range[0]) + rand_padding_range[0]  # scaling
    else:
        rand_padding = 0.075
    vert = normalize_pcd_seq(vert, padding=rand_padding, transform=reverse_transform)  # translation

    if ret_reverse_transform:
        return vert, reverse_transform
    else:
        return vert


def vis_pcd(point_cloud):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    layout = go.Layout(scene=dict(aspectmode='data'))
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def rand_uniform_sampler(vertices, triangles, num_samples):
    triangle_vertices = vertices[triangles]
    cross_product = np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0],
                            triangle_vertices[:, 2] - triangle_vertices[:, 0])
    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    probabilities = triangle_areas / np.sum(triangle_areas)
    sampled_triangle_indices = np.random.choice(len(triangles), size=num_samples, p=probabilities)
    u = np.random.rand(num_samples, 1)
    v = np.random.rand(num_samples, 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    rand = np.random.uniform(0., 1., (num_samples, 3))
    rand /= np.sum(rand, axis=-1, keepdims=True)
    u, v, w = rand[:, :1], rand[:, 1:2], rand[:, 2:3]
    sampled_points = (
        u * vertices[triangles[sampled_triangle_indices, 0]] +
        v * vertices[triangles[sampled_triangle_indices, 1]] +
        w * vertices[triangles[sampled_triangle_indices, 2]]
    )
    return sampled_points


def generate_random_ordering(args):
    verts, m = args
    return verts[np.random.permutation(m)]


def rand_reorder_by_frame(aug_verts, num_processes=None):
    nf, nv = aug_verts.shape[:2]
    def feed():
        for i in range(nf): yield (aug_verts[i], nv)
    if num_processes is None:
        num_processes = os.cpu_count() // 2
    with Pool(processes=num_processes) as pool:
        orderings = pool.map(generate_random_ordering, feed())
    orderings = np.stack(orderings)
    return orderings


# https://github.com/rabbityl/DeformingThings4D/blob/7cb946173968d88419b7432139f4be682cf61622/code/anime_renderer.py#L83C27-L83C27
def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


class Renderer():
    def __init__(self, height=480, width=640):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        R = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        
        
def render_depth(k, r, t, size, v, f):
    ext = np.eye(4)
    ext[:3, :3] = r
    ext[:3, 3] = t
    ext = np.linalg.inv(ext)
    renderer = Renderer(size, size)
    mesh = renderer.mesh_opengl(trimesh.Trimesh(vertices=v, faces=f))
    _, depth = renderer(size, size, k, ext, mesh = mesh)
    renderer.delete()
    return depth.astype(np.float32)


def back_project(k, r, t, dpt_map):
    H, W = dpt_map.shape
    grid = np.stack(np.meshgrid(np.arange(W), np.arange(H))).transpose((1, 2, 0))
    valid = dpt_map > 1e-6

    pixels = grid[valid]
    pixels_ = np.ones((pixels.shape[0], 3))
    pixels_[:, :2] = pixels

    cam_pts = pixels_ @ np.linalg.inv(k).T
    # cam_pts = cam_pts / np.linalg.norm(cam_pts, axis=-1, keepdims=True)

    dpt = dpt_map[valid]

    pts_3d = dpt[:, None] * cam_pts

    pts_3d_world = (pts_3d - t) @ r

    return pts_3d_world


def normalize_numpy(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def generate_random_semisphere_camera(radius=2, fov=60, size=300):
    theta = np.random.uniform(0, np.pi * 2)
    phi = np.random.uniform(0, np.pi / 2. - 1e-6)

    cam_center = np.array([np.cos(phi) * np.cos(theta),
                           np.cos(phi) * np.sin(theta),
                           np.sin(phi)]) * radius
    
    z_axis = normalize_numpy(-cam_center)
    x_axis = normalize_numpy(np.cross(np.array([0., 0., 1.]), cam_center))
    y_axis = normalize_numpy(np.cross(z_axis, x_axis))

    r = np.stack([x_axis, y_axis, z_axis]).reshape(3, 3)
    t = (-r @ cam_center.reshape(3, 1)).flatten()

    focal = size / (2. * np.tan(np.deg2rad(fov) / 2.))
    k = np.array([[focal, 0., size / 2],
                [0., focal, size / 2],
                [0., 0., 1.]])
    
    return k, r, t


def back_project_depth(
        verts: np.ndarray,
        faces: np.ndarray,

        size,

        k, r, t,

        return_dpt,
):  
    dpt = render_depth(k, r, t, size, verts, faces)
    new_pcd = back_project(k, r, t, dpt)
    if return_dpt:
        return new_pcd, dpt
    else:
        return new_pcd
    

def sample_pixel(img, pixels):
    pixels = pixels.astype(np.float32)
    x = pixels[:, :1]
    y = pixels[:, 1:]
    sampled = cv2.remap(img, x, y, cv2.INTER_LINEAR)
    return sampled.squeeze()


def get_feature_weight(sampled_verts, dpts, k, r, t, size, thresh=0.05):
    # sampled verts: nf, nt, 3
    # dpts: nf, h, w
    
    F, T = sampled_verts.shape[:2]

    sampled_verts = sampled_verts.reshape(-1, 3)
    pixel_coords = (sampled_verts @ r.T + t) @ k.T
    pixel_coords = pixel_coords[:, :2] /  pixel_coords[:, 2:]  # F*T, 2

    pixel_coords = pixel_coords.reshape(F, T, 2)
    pixel_dpts = []

    for f in range(F):
        pixel_dpts.append(sample_pixel(dpts[f], pixel_coords[f]))
    pixel_dpts = np.stack(pixel_dpts).reshape(F, T)  # F, T
    
    cam_center = -r.T @ t
    dirs = sampled_verts - cam_center
    z_axis = r[2, :]
    dist = dirs @ z_axis.reshape(3, 1)
    dist = dist.reshape(F, T)

    mask = np.abs(dist - pixel_dpts) < thresh

    return mask.astype(np.float32)


def get_first_appear(feature_weight_gt):
    # feature_weight_gt: F, T
    feature_weight_gt = np.concatenate([feature_weight_gt, np.ones((1, feature_weight_gt.shape[1]))], axis=0)
    return np.argmax(feature_weight_gt.T, axis=1)


# specific for training
class DfaustTrain(data.Dataset):
    def __init__(self, 
                 root_dir: str,
                 
                 split='test',  # passed in by main.py
                 
                 num_tracks: int=256,
                 num_points: int=2048,
                 num_mesh_points: int=100000,
                 max_frame: int=24,
                 
                 rand_start: bool=True,
                 rand_start_min_length: int=50,
                 rand_scaling: bool=True,
                 rand_scaling_range: List[float]=[0.5, 1.5],
                 rand_rotation: bool=True,
                 rand_rotate_z_only: bool=True,
                 rand_padding: bool=True,
                 rand_padding_range: List[float]=[0.05, 0.1],
                 rand_reordering: bool=False,
                 rand_perturb: bool=True,
                 rand_perturb_avg_sd: List[float]=[0., 1.e-5],

                 uni_prob: float=1.0,
                 noise_ratio: float=0.,
                 over_sampling: float=3.0,
                 fps_method: str='fpsample',

                 depth_proj_prob: float=1.,
                 track_mesh: bool=True,
                 rand_query_mesh_prob: float=1.,
                 
                 knn: int=4,
                 
                 **kwargs,
                 ) -> None:
        super().__init__()
        # assert split in ['train', 'val'], f'Split type {split} is not supported'
        self.split = split
        np.random.seed(42)

        self.root_dir = root_dir
        
        self.max_frame = max_frame
        self.num_tracks = num_tracks
        self.num_points = num_points
        self.num_mesh_points = num_mesh_points
        self.knn = knn
        
        self.rand_start = rand_start
        self.rand_start_min_length = int(rand_start_min_length)
        
        self.rand_scaling = rand_scaling
        self.rand_scaling_range = rand_scaling_range
        
        self.rand_rotation = rand_rotation
        self.rand_rotate_z_only = rand_rotate_z_only
        
        self.rand_padding = rand_padding
        self.rand_padding_range = rand_padding_range
        
        self.rand_reordering = rand_reordering
        
        self.rand_perturb = rand_perturb
        self.rand_perturb_avg_sd = rand_perturb_avg_sd

        self.uni_prob = uni_prob
        self.noise_ratio = noise_ratio
        self.over_sampling = over_sampling
        self.fps_method = fps_method

        self.depth_proj_prob = depth_proj_prob
        self.track_mesh = track_mesh
        self.rand_mesh = rand_query_mesh_prob

        if self.split == 'train':
            split_file = 'splits/train.lst'
        elif self.split == 'val':
            split_file = 'splits/val.lst'
        else:
            # raise NotImplementedError()
            split_file = 'splits/test.lst'

        with open(os.path.join(self.root_dir, split_file)) as f:
            split_data = [line.split('\n')[0] for line in f.readlines()]

        with h5py.File(os.path.join(self.root_dir, 'registrations_f.hdf5'), 'r') as f:
            f_seqs = [np.array(f[seq_name]) for seq_name in split_data if seq_name in f.keys()]
            f_faces = [np.array(f['faces'])] * len(f_seqs)

        with h5py.File(os.path.join(self.root_dir, 'registrations_m.hdf5'), 'r') as f:
            m_seqs = [np.array(f[seq_name]) for seq_name in split_data if seq_name in f.keys()]
            m_faces = [np.array(f['faces'])] * len(m_seqs)

        self.seqs = f_seqs + m_seqs
        self.faces = f_faces + m_faces
        
    def __len__(self):
        return len(self.seqs) * 3  # go three times

    def __getitem__(self, idx):
        idx = idx % len(self.seqs)
        reverse_transform = []

        vert_data = self.seqs[idx].transpose(2, 0, 1)[0]
        offset_data = self.seqs[idx].transpose(2, 0, 1)[1:] - vert_data[None]  # nf - 1
        face_data = self.faces[idx]

        nv = vert_data.shape[0]
        nf = offset_data.shape[0] + 1

        # nf, nv, nt, vert_data, face_data, offset_data = anime_read(self.animes[idx])  # int, int, int, Nv*3, Nt*3, (Nf-1)*Nv*3
        
        assert not np.isnan(vert_data).any()
        assert not np.isnan(face_data).any()
        assert not np.isnan(offset_data).any()
        assert nf == offset_data.shape[0] + 1

        gt_verts_all = vert_data + offset_data  # nf, nv, 3
        gt_verts_all = np.concatenate([vert_data[None], gt_verts_all], axis=0)  # F, N, 3

        # randomly choose frames
        if nf > self.max_frame:
            start_frame = np.random.randint(0, nf - self.max_frame + 1)
            gt_verts = gt_verts_all[start_frame : start_frame + self.max_frame].copy()
        else:
            start_frame = 0
            gt_verts = gt_verts_all.copy()

        if self.track_mesh:
            rand_mesh = np.random.uniform() < self.rand_mesh
            if rand_mesh:
                chosen_id = np.random.randint(nf)
            else:
                chosen_id = start_frame
            query_mesh = gt_verts_all[chosen_id]
            gt_verts = np.concatenate([query_mesh[None], gt_verts], axis=0)
            
        nf = gt_verts.shape[0]
        gt_verts_copy = gt_verts.copy()

        gt_verts, reverse_transform = proc_frame_points(gt_verts, 
                                                        ret_reverse_transform=True,
                                                        use_rand_rotation=self.rand_rotation,
                                                        rand_rotate_z_only=self.rand_rotate_z_only,
                                                        use_rand_padding=self.rand_padding,
                                                        rand_padding_range=self.rand_padding_range)
        
        # pcd seq aug, nothing to do with gt
        aug_verts = gt_verts.copy()

        depth_proj = np.random.uniform() < self.depth_proj_prob

        if depth_proj:
            # project first
            subsampled_verts = []
            dpts = []
            SIZE = 300
            k, r, t = generate_random_semisphere_camera(radius=2, fov=60, size=SIZE)
            sample_num_points = self.num_points
            for f in range(nf):
                subsampled_vert, dpt = back_project_depth(aug_verts[f], face_data, SIZE, k, r, t, return_dpt=True)
                dpts.append(dpt)
                if sample_num_points > subsampled_vert.shape[0]:
                    sample_num_points = subsampled_vert.shape[0]
                subsampled_verts.append(subsampled_vert)
            dpts = np.stack(dpts)  # nf, SIZE, SIZE

            rand_sample_buffer = np.empty((nf, sample_num_points, 3)) 

            num_point_samples = int(rand_sample_buffer.shape[1] * (1. - self.noise_ratio))
            for i in range(nf): 
                rand_sample_buffer_i = fps(subsampled_verts[i], num_point_samples, method=self.fps_method)
                rand_sample_buffer[i] = noisify_pcd(rand_sample_buffer_i, rand_sample_buffer.shape[1] - num_point_samples)

            if self.rand_reordering:  # reorder
                rand_sample_buffer = rand_reorder_by_frame(rand_sample_buffer)
                
            if self.rand_perturb:
                rand_sample_buffer += np.random.randn(*rand_sample_buffer.shape) * self.rand_perturb_avg_sd[1] + self.rand_perturb_avg_sd[0]    
        else:
            if self.split == 'train':
                sample_num_points = self.num_points
            else:
                sample_num_points = min(self.num_points, nv)

            if self.rand_reordering:  # reorder
                aug_verts = rand_reorder_by_frame(aug_verts)
                
            if self.rand_perturb:
                aug_verts += np.random.randn(*aug_verts.shape) * self.rand_perturb_avg_sd[1] + self.rand_perturb_avg_sd[0]    

            # vis_pcd(aug_verts[0])
            rand_sample_buffer = np.empty((nf, sample_num_points, 3))  # resample mesh, weighted by face area

            num_point_samples = int(rand_sample_buffer.shape[1] * (1. - self.noise_ratio))
            num_over_samples = int(self.over_sampling * num_point_samples)
            for i in range(nf): 
                rand_sample_buffer_i_over = rand_uniform_sampler(aug_verts[i], face_data, num_over_samples)
                rand_sample_buffer_i = fps(rand_sample_buffer_i_over, num_point_samples, method=self.fps_method)
                rand_sample_buffer[i] = noisify_pcd(rand_sample_buffer_i, rand_sample_buffer.shape[1] - num_point_samples)

        rand_sample_buffer[rand_sample_buffer > 1.] = 1. - 1e-5
        rand_sample_buffer[rand_sample_buffer < -1.] = -1. + 1e-5
        
        # sampled_verts_indices = np.random.choice(nv, size=self.num_tracks, replace=False)
        sampled_verts_indices = smart_sample(gt_verts[0], self.num_mesh_points, self.uni_prob)
        sampled_verts = gt_verts[:, sampled_verts_indices]
        
        transform = np.eye(4)[None]
        for tran_mat in reversed(reverse_transform):
            transform = np.matmul(transform, tran_mat)

        if self.track_mesh:
            points_mesh = sampled_verts[0]
            tracks_mesh = sampled_verts[0]
            transform = transform[1:]
            sampled_verts = sampled_verts[1:]
            gt_verts_copy = gt_verts_copy[1:]
            rand_sample_buffer = rand_sample_buffer[1:]

        feature_weight_gt = np.ones((sampled_verts.shape[0], self.num_tracks))
        first_appear = np.zeros((self.num_tracks))

        assert rand_sample_buffer.shape[0] > 0

        ret = {
            'tracks': sampled_verts,  # F, N, 3 -> gt
            'points': rand_sample_buffer,  # nf, npoint, 3 -> sparse/partial input for each frame
            'points_mesh': points_mesh  # N, 3 -> track mesh input
        }

        return ret
    

def eval_metric(pred, gt):
    # both F, N, 3
    ate = torch.abs(pred-gt).mean()
    l2 = torch.norm(pred-gt,dim=-1)

    a01 = l2 < 0.01  # F, N
    d01 = torch.sum(a01).float() / (a01.shape[0] * a01.shape[1])

    a02 = l2 < 0.05  # F, N
    d02 = torch.sum(a02).float() / (a02.shape[0] * a02.shape[1])

    return ate, d01, d02


if __name__ == '__main__':
    df_trainset = DfaustTrain(
        root_dir='/home/idarc/hgz/SuperGraph/DeformationPyramid/data/DFAUST',
        split='train'
    )
    df_testset = DfaustTrain(
        root_dir='/home/idarc/hgz/SuperGraph/DeformationPyramid/data/DFAUST',
        split='test',
    )
    data = df_trainset[0]
    src_pcd = data['points_mesh']
    tgt_pcd = data['points']
    gt_tracks = data['tracks']
    # from cvtb import vis
    # vis.pcds([tgt_pcd[0], gt_tracks[0]])

    np.save('dftmp.npy', {
        'src_pcd': src_pcd,
        'tgt_pcd': tgt_pcd,
        'gt_tracks': gt_tracks
    })
    breakpoint()
                 

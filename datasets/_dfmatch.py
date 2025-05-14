import os, sys, glob, torch
# sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from typing import List, Dict
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from torch.utils.data import Dataset
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, KDTree_corr
from lib.utils import load_obj
HMN_intrin = np.array( [443, 256, 443, 250 ])
cam_intrin = np.array( [443, 256, 443, 250 ])

from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences
from datasets.df_utils import DfaustTrain
from datasets.gen_df_train import generate_training_labels, generate_evaluation_labels


CACHE_DF = True  # Use true for actual training
NUM_EVAL_FRAME = 24
NUM_REPEAT = 10
NUM_KNN = 4
ROOT_DIR_DF = '/home/idarc/hgz/SuperGraph/DeformationPyramid/data/DFAUST'


class _DFMatch(Dataset):

    def __init__(self, config, split, data_augmentation=True):
        super(_DFMatch, self).__init__()

        assert split in ['train','val','test']

        if 'overfit' in config.exp_dir:
            d_slice = config.batch_size
        else :
            d_slice = None

        # self.entries = self.read_entries(  config.split[split] , config.data_root, d_slice=d_slice )

        if CACHE_DF:
            self.data: List = self.cache_data(split, num_repeat=NUM_REPEAT)  # 400 * 10 = 4000
        else:
            self.data = DfaustTrain(split)

        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000

        self.overlap_radius = 0.0375

        self.cache = {}
        self.cache_size = 30000
    
    def get_entry(self, index):
        index_valid = int(index % len(self.data)) 
        if CACHE_DF:
            return self.data[index_valid]
        else:
            data = self.data[index_valid]
            return generate_training_labels(data, knn=NUM_KNN)
    
    def cache_data(self, split, num_repeat: int = 1):
        df_dataset = self.get_df_dataset(split)
        df_len = len(df_dataset)
        df_total = int(num_repeat * df_len)
        data = []
        for i_total in tqdm(range(df_total)):
            i_repeat = i_total % df_len  # Useless
            i_df = i_total // df_len
            data = df_dataset[i_df]
            entry: Dict = generate_training_labels(data, knn=NUM_KNN)
            data.append(entry)
        return data

    def get_df_dataset(self, split):
        # The notion is that for training split we do not need 24 frames for every sample
        # However for evaluation we want the while 24 of them
        return DfaustTrain(
            root = ROOT_DIR_DF,
            split = 'train' if split == 'train' else 'test',
            max_frame = 1 if split == 'train' else NUM_EVAL_FRAME,
        )

    # def read_entries (self, split, data_root, d_slice=None, shuffle= False):
    #     entries = glob.glob(os.path.join(data_root, split, "*/*.npz"))
    #     if shuffle:
    #         random.shuffle(entries)
    #     if d_slice:
    #         return entries[:d_slice]
    #     return entries

    def __len__(self):
        if CACHE_DF:
            return len(self.data)
        else:
            return int(len(self.data) * NUM_REPEAT)

    def __getitem__(self, index, debug=False):

        # if index in self.cache:
        #     entry = self.cache[index]

        # else :
        #     entry = np.load(self.entries[index])
        #     if len(self.cache) < self.cache_size:
        #         self.cache[index] = entry
        
        entry = self.get_entry(index=index)

        # get transformation
        rot = entry['rot']
        trans = entry['trans']
        s2t_flow = entry['s2t_flow']
        src_pcd = entry['s_pc']
        tgt_pcd = entry['t_pc']
        correspondences = entry['correspondences'] # obtained with search radius 0.015 m
        src_pcd_deformed = src_pcd + s2t_flow
        if "metric_index" in entry:
            metric_index = entry['metric_index'].squeeze()
        else:
            metric_index = None

        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.show()

        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                src_pcd_deformed = np.matmul(rot_ab, src_pcd_deformed.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
            s2t_flow = src_pcd_deformed - src_pcd

        if debug:
            # wrapp_src = (np.matmul(rot, src_pcd.T)+ trans).T
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_red)
            mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], scale_factor=scale_factor, color=c_blue)
            mlab.show()

        if (trans.ndim == 1):
            trans = trans[:, None]

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        #R * ( Ps + flow ) + t  = Pt
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index



if __name__ == '__main__':
    from lib.utils import load_config
    from easydict import EasyDict as edict
    from lib.tictok import Timers
    import yaml
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return '_'.join([str(i) for i in seq])
    yaml.add_constructor('!join', join)

    config = "/home/liyang/workspace/Regformer/configs/train/4dmatch.yaml"
    with open(config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = edict(config)
    config.timers=Timers()
    D = _4DMatch(config, "test")

    for i in range (len(D)):

        try:
            if i%1000 == 0 :
                print (i,"/",len(D))
            D.__getitem__(i, debug=True)
        except:
            # print(i, "/", len(D))
            pass
        
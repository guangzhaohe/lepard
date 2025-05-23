"""
Evaluate on DFAUST testset by chaining pair-wise non-rigid prediction
"""
import os, torch, json, argparse, shutil
from easydict import EasyDict as edict
import yaml
from datasets.dataloader import get_dataloader, get_datasets
from models.pipeline import Pipeline
from lib.utils import setup_seed
from lib.tester import get_trainer
from models.loss import MatchMotionLoss
from lib.tictok import Timers
from time import perf_counter
from configs.models import architectures
from torch import optim
import numpy as np
from datasets.df_utils import DfaustTrain
from datasets.dataloader import prepare_data
from datasets.gen_df_train import generate_training_labels
from typing import Dict, List
from datasets._dfmatch import NUM_KNN, ROOT_DIR_DF
from tqdm import tqdm
from models.matching import Matching as CM
from models.loss import MatchMotionLoss as MML
from cvtb import vis
from extern.nonrigid_icp_pytorch.model.registration import Registration
from yacs.config import CfgNode as CN
from knn_cuda import KNN


setup_seed(0)


def eval_metric(pred, gt):
    # F, N, 3
    ate = torch.abs(pred-gt).mean()
    l2 = torch.norm(pred-gt,dim=-1)

    a01 = l2 < 0.01  # F, N
    d01 = torch.sum(a01).float() / (a01.shape[0] * a01.shape[1])

    a02 = l2 < 0.05  # F, N
    d02 = torch.sum(a02).float() / (a02.shape[0] * a02.shape[1])

    return ate, d01, d02


def get_knn(find_in: torch.Tensor, find_for: torch.Tensor, knn: int = 1):
    # both are N_X x 3 pcds
    knn = KNN(k=knn, transpose_mode=True)
    ref = find_in[None]
    query = find_for[None]
    dist, indx = knn(ref, query)
    return indx[0, :, 0]  # N_find_for


def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])


yaml.add_constructor('!join', join)


# Convert a pair of src tar data output to lepard's labels
def from_src_tar_to_training_labels(
    src_pcd: np.ndarray, 
    tar_pcd: np.ndarray,
) -> List:
    tmp_data = {
        'points_mesh': src_pcd,  # n_src, 3
        'points': tar_pcd[None],  # 1, n_tar, 3
        'tracks': src_pcd[None],  # 1, n_src, 3 -> placeholder
    }
    entry = generate_training_labels(tmp_data, knn=NUM_KNN)
    
    # from entry to training labels
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

    if (trans.ndim == 1):
        trans = trans[:, None]

    src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
    tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
    rot = rot.astype(np.float32)
    trans = trans.astype(np.float32)
    
    return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index


def to_cuda(inputs: Dict, device: str = 'cuda'):
    for k, v in inputs.items():
        if type(v) == list:
            inputs[k] = [item.to(device) for item in v]
        elif type(v) in [ dict, float, type(None), np.ndarray]:
            pass
        else:
            inputs[k] = v.to(device)
    return inputs


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['dataset']+config['folder'], config['exp_dir'])
    config['tboard_dir'] = 'snapshot/%s/%s/tensorboard' % (config['dataset']+config['folder'], config['exp_dir'])
    config['save_dir'] = 'snapshot/%s/%s/checkpoints' % (config['dataset']+config['folder'], config['exp_dir'])
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')

    # model initialization
    config.kpfcn_config.architecture = architectures[config.dataset]
    config.model = Pipeline(config)

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )

    #create learning rate scheduler
    if  'overfit' in config.exp_dir :
        config.scheduler = optim.lr_scheduler.MultiStepLR(
            config.optimizer,
            milestones=[config.max_epoch-1], # fix lr during overfitting
            gamma=0.1,
            last_epoch=-1)

    else:
        config.scheduler = optim.lr_scheduler.ExponentialLR(
            config.optimizer,
            gamma=config.scheduler_gamma,
        )

    config.timers = Timers()

    # create dataset and dataloader
    # train_set, val_set, test_set = get_datasets(config)
    neighborhood_limits = np.array([53, 23, 31, 37])  # NOTE: for DFAUST
    # config.train_loader, neighborhood_limits = get_dataloader(train_set,config,shuffle=True)
    # config.test_loader, _ = get_dataloader(test_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    
    # config.desc_loss = MetricLoss(config)
    config.desc_loss = MatchMotionLoss(config['train_loss'])

    config.train_loader = None
    config.test_loader = None
    config.val_loader = None

    # Pretrained weights will be loaded here
    trainer = get_trainer(config)

    # example_input = next(iter(config.test_loader))
    # inputs = np.load('tmp.npy', allow_pickle=True).item()

    # for k, v in inputs.items():
    #     if type(v) == list:
    #         inputs[k] = [item.to('cuda') for item in v]
    #     elif type(v) in [ dict, float, type(None), np.ndarray]:
    #         pass
    #     else:
    #         inputs[k] = v.to('cuda')

    df_dataset = DfaustTrain(
        root_dir=ROOT_DIR_DF,
        split='test',
    )

    # NICP config
    incp_cfg = CN()
    incp_cfg.iters = 500
    incp_cfg.gpu_mode = True
    incp_cfg.lr = 0.5
    incp_cfg.deformation_model = 'ED'
    incp_cfg.w_silh = 0.1
    incp_cfg.w_ldmk = 1
    incp_cfg.w_arap = 1
    incp_cfg.w_chamfer = 0.1
    incp_cfg.w_depth = 0

    ate, d01, d02, timing = [], [], [], []
    
    for idx, data_batch in enumerate(tqdm(df_dataset)):
        
        # Get what we need
        src_pcd = data_batch['points_mesh']  # n_src, 3
        tar_pcds = data_batch['points']  # n_f, n_tar, 3
        
        # This will only be used for evauation
        gt_tracks = data_batch['tracks']  # n_f, n_src, 3
        
        n_f = len(tar_pcds)
        
        registered_pcds = []  # length = n_f

        timing_i = 0.
        
        for f in range(n_f):
            
            if f == 0:  # src is now provided
                src_pcd_f = src_pcd
                tar_pcd_f = tar_pcds[f]
            else:  # src is now from previous estimations
                # src_pcd_f = registered_pcds[-1]
                src_pcd_f = src_pcd  # NOTE: No-Chaining
                tar_pcd_f = tar_pcds[f]
                
            labels: Dict = from_src_tar_to_training_labels(src_pcd_f, tar_pcd_f)  # Dict to be collated
            data_f = prepare_data(labels, config=config, neighborhood_limits=neighborhood_limits)
            data_f_cuda = to_cuda(data_f)
            # breakpoint()

            timing_start_i = perf_counter()

            output = trainer.model(data_f_cuda)
            # match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=True)
            
            # From what I guess, `src_ind_coarse`, `tgt_ind_coarse`, `coarse_match_pred`, `s_pcd`, `t_pcd`
            s_pcd = output['s_pcd'][0]
            t_pcd = output['t_pcd'][0]

            # breakpoint()
            # We need s_pcd and t_pcd's indices in input points
            s_pcd_ind_ori = get_knn(torch.from_numpy(src_pcd_f).cuda(), s_pcd, knn=1)
            t_pcd_ind_ori = get_knn(torch.from_numpy(tar_pcd_f).cuda(), t_pcd, knn=1)

            coarse_match_pred = output['coarse_match_pred']
            match_ind_src = s_pcd_ind_ori[coarse_match_pred[:, 1]]
            match_ind_tar = t_pcd_ind_ori[coarse_match_pred[:, 2]]

            # np.save('tmp.npy', [src_pcd_f[match_ind_src.cpu().numpy()], tar_pcd_f[match_ind_tar.cpu().numpy()]])

            # register w/ nicp
            model = Registration(src_pcd_f, config=incp_cfg)
            match_lm = torch.stack([match_ind_src, match_ind_tar]).permute(1, 0).cuda()
            registered_pcd = model.register_a_depth_frame(torch.from_numpy(tar_pcd_f).cuda(), landmarks=match_lm)

            timing_end_i = perf_counter()
            timing_i += timing_end_i - timing_start_i

            registered_pcds.append(registered_pcd.detach().cpu().numpy())
            del model

            # breakpoint()
            # registered_pcds.append(output['registered_pcd'])
            
        
        # TODO: Evaluate here
        ate_i, d01_i, d02_i = eval_metric(
            torch.from_numpy(np.stack(registered_pcds)).cuda(),
            torch.from_numpy(gt_tracks).cuda()
        )

        ate.append(ate_i)
        d01.append(d01_i)
        d02.append(d02_i)
        timing.append(timing_i / n_f)

        print(f'Case {idx}:02d - ate: {ate_i} - d01: {d01_i} - d02: {d02_i} - timing: {timing_i / n_f}')

    num_cases = len(ate)
    print(f'Average - ate: {sum(ate) / num_cases} - d01: {sum(d01) / num_cases} - d02: {sum(d02) / num_cases} - timing: {sum(timing) / num_cases}')


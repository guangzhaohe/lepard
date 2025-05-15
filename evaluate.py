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


setup_seed(0)


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
    
    for idx, data_batch in enumerate(tqdm(df_dataset)):
        
        # Get what we need
        src_pcd = data_batch['points_mesh']  # n_src, 3
        tar_pcds = data_batch['points']  # n_f, n_tar, 3
        
        # This will only be used for evauation
        gt_tracks = data_batch['tracks']  # n_f, n_src, 3
        
        n_f = len(tar_pcds)
        
        registered_pcds = []  # length = n_f
        
        for f in range(n_f):
            
            if f == 0:  # src is now provided
                src_pcd_f = src_pcd
                tar_pcd_f = tar_pcds[f]
            else:  # src is now from previous estimations
                src_pcd_f = registered_pcds[-1]
                tar_pcd_f = tar_pcds[f]
                
            labels: Dict = from_src_tar_to_training_labels(src_pcd_f, tar_pcd_f)  # Dict to be collated
            data_f = prepare_data(labels, config=config, neighborhood_limits=neighborhood_limits)
            data_f_cuda = to_cuda(data_f)
            output = trainer.model(data_f_cuda)
            # match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=True)
            
            # From what I guess, `src_ind_coarse`, `tgt_ind_coarse`, `coarse_match_pred`, `s_pcd`, `t_pcd`
            s_pcd = output['s_pcd'][0].cpu().numpy()
            t_pcd = output['t_pcd'][0].cpu().numpy()

            # src_ind_coarse = output['src_ind_coarse'].cpu().numpy()
            # tgt_ind_coarse = output['tgt_ind_coarse'].cpu().numpy()

            # src_coarse = s_pcd[src_ind_coarse]
            # tgt_coarse = t_pcd[tgt_ind_coarse]

            coarse_match_pred = output['coarse_match_pred'].cpu().numpy()

            # Landmarks used for N-ICP matching
            # For N-ICP we actually need the original indices
            src_lm = s_pcd[coarse_match_pred[:, 1]]
            tar_lm = t_pcd[coarse_match_pred[:, 2]]

            # Let's make sure landmarks align with original src input and output
            # np.save('tmp.npy', {
            #     'og_src': src_pcd_f, 
            #     'og_tar': tar_pcd_f, 
            #     'src_lm': src_lm, 
            #     'tar_lm': tar_lm,
            # })

            breakpoint()
            registered_pcds.append(output['registered_pcd'])
            
        # TODO: Evaluate here
        
    
    trainer.test()

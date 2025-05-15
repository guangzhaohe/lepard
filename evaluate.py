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
from datasets._dfmatch import NUM_KNN
from tqdm import tqdm
from models.matching import Matching as CM
from models.loss import MatchMotionLoss as MML


setup_seed(0)


def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])


yaml.add_constructor('!join', join)


# Convert a pair of src tar data output to lepard's labels
def from_src_tar_to_training_labels(
    src_pcd: np.ndarray, 
    tar_pcd: np.ndarray,
) -> Dict:
    tmp_data = {
        'points_mesh': src_pcd,  # n_src, 3
        'points': tar_pcd[None],  # 1, n_tar, 3
        'tracks': src_pcd[None],  # 1, n_src, 3 -> placeholder
    }
    training_labels = generate_training_labels(tmp_data, knn=NUM_KNN)
    return training_labels


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
    inputs = np.load('tmp.npy', allow_pickle=True).item()

    for k, v in inputs.items():
        if type(v) == list:
            inputs[k] = [item.to('cuda') for item in v]
        elif type(v) in [ dict, float, type(None), np.ndarray]:
            pass
        else:
            inputs[k] = v.to('cuda')

    df_dataset = DfaustTrain(split='test')
    
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
            breakpoint()
            output = trainer.model(data_f_cuda)
            # match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=True)
            
            breakpoint()
            registered_pcds.append(output['registered_pcd'])
            
        # TODO: Evaluate here
        
    
    trainer.test()

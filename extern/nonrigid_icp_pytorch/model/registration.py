import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage import io
from PIL import Image
from timeit import default_timer as timer
import datetime
import argparse
from .geometry import *

import yaml
import matplotlib.pyplot as plt
import torch
from lietorch import SO3, SE3, LieGroupParameter
import torch.optim as optim
from .loss import *
from .point_render import PCDRender

from .build_ed_graph import build_deformation_graph



class Registration():


    def __init__(self, src_pcd, config):
        # n_src, 3
        config.device = 'cuda'
        self.device = config.device

        self.deformation_model = config.deformation_model

        self.config = config

        """initialize deformation graph"""
        data = build_deformation_graph(src_pcd)
        self.graph_nodes = data['graph_nodes'].to(self.device)
        self.graph_edges = data['graph_edges'].to(self.device)
        self.graph_edges_weights = data['graph_edges_weights'].to(self.device)

        """initialize point clouds"""
        # valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
        self.source_pcd = torch.from_numpy(src_pcd).to(self.device)
        self.point_anchors = data['point_anchors'].to(self.device)
        self.anchor_weight = data['point_weights'].to(self.device)
        self.anchor_loc = data["graph_nodes"][self.point_anchors.cpu()].to(self.device)
        self.frame_point_len = [ len(self.source_pcd)]

    def register_a_depth_frame(self, tgt_pcd, landmarks=None):
        """
        :param tgt_depth_path:
        :return:
        """

        """load target frame"""
        # tgt_depth = io.imread( tgt_depth_path )/1000.
        # depth_mask = torch.from_numpy(tgt_depth > 0)
        # tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1,2,0)
        # self.tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float().to(self.device)
        self.tgt_pcd = tgt_pcd
        # pix_2_pcd = self.map_pixel_to_pcd( depth_mask ).to(self.device)

        # self.visualize_results(self.tgt_pcd)
        warped_pcd = self.solve(landmarks=landmarks)
        # self.visualize_results( self.tgt_pcd, warped_pcd)
        return warped_pcd
        # breakpoint()

    def solve(self, **kwargs ):


        if self.deformation_model == "ED":
            # Embeded_deformation, c.f. https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
            return self.optimize_ED(**kwargs)


    def optimize_ED(self,  landmarks=None):
        '''
        :param landmarks:
        :return:
        '''


        """translations"""
        node_translations = torch.zeros_like(self.graph_nodes)
        self.t = torch.nn.Parameter(node_translations)
        self.t.requires_grad = True

        """rotations"""
        phi = torch.zeros_like(self.graph_nodes)
        node_rotations = SO3.exp(phi)
        self.R = LieGroupParameter(node_rotations)


        """optimizer setup"""
        optimizer = optim.Adam([self.R, self.t], lr= self.config.lr )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)


        """render reference pcd"""
        # sil_tgt, d_tgt, _ = self.render_pcd( self.tgt_pcd )


        # Transform points
        for i in range(self.config.iters):

            anchor_trn = self.t [self.point_anchors]
            anchor_rot = self.R [ self.point_anchors]
            warped_pcd = ED_warp(self.source_pcd, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)

            err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)
            err_ldmk = landmark_cost(warped_pcd, self.tgt_pcd, landmarks) if landmarks is not None else 0

            cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.config.w_chamfer > 0 else 0

            loss = \
                err_arap * self.config.w_arap + \
                err_ldmk * self.config.w_ldmk + \
                cd * self.config.w_chamfer

            # print( i, loss)
            if loss.item() < 1e-7:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return warped_pcd







    def render_pcd (self, x):
        INF = 1e+6
        px, dx = self.renderer(x)
        px, dx  = map(lambda feat: feat.squeeze(), [px, dx ])
        dx[dx < 0] = INF
        mask = px[..., 0] > 0
        return px, dx, mask


    def map_pixel_to_pcd(self, valid_pix_mask):
        ''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
        :param valid_pix_mask:
        :return:
        '''
        image_size = valid_pix_mask.shape
        pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
        pix_2_pcd_map [~valid_pix_mask] = -1
        return pix_2_pcd_map


    def visualize_results(self, tgt_pcd, warped_pcd=None):
        return

        # import mayavi.mlab as mlab
        c_red = (224. / 255., 0 / 255., 125 / 255.)
        c_pink = (224. / 255., 75. / 255., 232. / 255.)
        c_blue = (0. / 255., 0. / 255., 255. / 255.)
        scale_factor = 0.007
        source_pcd = self.source_pcd.cpu().numpy()
        tgt_pcd = tgt_pcd.cpu().numpy()

        # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
        # if warped_pcd is None:
        #     mlab.points3d(source_pcd[ :, 0], source_pcd[ :, 1], source_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_red)
        # else:
        #     warped_pcd = warped_pcd.detach().cpu().numpy()
        #     mlab.points3d(warped_pcd[ :, 0], warped_pcd[ :, 1], warped_pcd[:,  2], resolution=4, scale_factor=scale_factor , color=c_pink)
        # mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_blue)
        # mlab.show()
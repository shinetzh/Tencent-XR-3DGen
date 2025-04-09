#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce

from models.archs.sdf_decoder import * 
from models.archs.encoders.conv_pointnet import ConvPointnet
from utils import mesh, evaluate


class SdfModelNopn(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        model_specs = self.specs["SdfModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.plane_resolution = specs["plane_resolution"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)
        self.model = SdfDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)
        
        self.model.train()
        #print(self.model)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), self.specs["sdf_lr"])
        return optimizer
    
    def normalize_coordinate2(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane =='xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6) # (-1, 1)
        xy_new = xy_new + 1.0 # range (0, 2)

        # f there are outliers out of the range
        if xy_new.max() >= 2:
            xy_new[xy_new >= 2] = 2 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane, padding=0.1):
        xy = self.normalize_coordinate2(query.clone(), plane=plane, padding=padding)
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        vgrid = xy - 1.0
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model( torch.cat((xyz, point_features),dim=-1) )
        return pred_sdf # [B, num_points] 


    def get_points_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        idx = int(plane_features.shape[1] / 3)
        fea = {}
        fea['xz'], fea['xy'], fea['yz'] = plane_features[:,0:idx,...], plane_features[:,idx:idx*2,...], plane_features[:,idx*2:,...]
        #print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])
        plane_feat_sum = 0
        plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
        plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)


    def forward(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model( torch.cat((xyz, point_features),dim=-1) )
        return pred_sdf # [B, num_points] 
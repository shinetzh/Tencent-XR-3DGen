#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import json
from easydict import EasyDict as edict
import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce

from decoder.sdfcolor_decoder.sdf_decoder_yh import * 
from utils import mesh, evaluate

class SdfColorModel(pl.LightningModule):
    def __init__(self, config_json):
        super().__init__()
        with open(config_json, encoding='utf-8') as f:
            self.dataset_dict = edict(json.load(f))
        
        self.config = self.dataset_dict.config.Config

        self.mlp_path = self.dataset_dict.config.MLP

        self.model_geo = SdfDecoderYh(d_in=self.config.channels, 
                                  d_out=1, 
                                  d_hidden=self.config.width, 
                                  n_layers=self.config.n_layers, 
                                  skip_in=self.config.skip_in,
                                  ).cuda()
        self.model_color = SdfDecoderYh(d_in=self.config.channels, 
                                  d_out=3, 
                                  d_hidden=self.config.width, 
                                  n_layers=self.config.n_layers, 
                                  skip_in=self.config.skip_in,
                                  ).cuda()

        ckpt_state_dict = torch.load(self.mlp_path, map_location="cuda")


        self.model_geo.load_state_dict(ckpt_state_dict["Geo"], strict=True)
        self.model_color.load_state_dict(ckpt_state_dict["Tex"], strict=True)
        print("loaded Geo and Tex model from {}".format(self.mlp_path))

        self.model_geo.eval()
        self.model_color.eval()
        print(self.model_color)

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
        if plane == 'zx':
            xy = p[:, :, [2, 0]]
        elif plane =='yx':
            xy = p[:, :, [1, 0]]
        else:
            xy = p[:, :, [1, 2]]

        return xy
        # # xy_new = xy / (1 + padding + 10e-6) # (-1, 1)
        # xy_new = xy
        # xy_new = xy_new + 1.0 # range (0, 2)

        # # f there are outliers out of the range
        # if xy_new.max() >= 2:
        #     xy_new[xy_new >= 2] = 2 - 10e-6
        # if xy_new.min() < 0:
        #     xy_new[xy_new < 0] = 0.0
        # return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane, padding=0.1):
        xy = self.normalize_coordinate2(query.clone(), plane=plane, padding=padding)
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # vgrid = xy - 1.0
        vgrid = xy
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        plane_features_geo = plane_features[:, 0:3, ...]
        plane_features_color = plane_features[:, 3:, ...]
        sdf_features = self.get_points_plane_features(plane_features_geo, xyz) # point_features: B, N, D
        pred_sdf = self.model_geo(sdf_features)
        rgb_features = self.get_points_plane_features(plane_features_color, xyz)
        pred_rgb = self.model_color(rgb_features)
        return pred_sdf, pred_rgb


    def get_points_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        fea = {}
        fea['yx'], fea['zx'], fea['yz'] = plane_features[:, 0, ...], plane_features[:, 1, ...], plane_features[:, 2, ...]
        #print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])
        plane_feat_sum = 0
        plane_feat_sum += self.sample_plane_feature(query, fea['yx'], 'yx')
        plane_feat_sum += self.sample_plane_feature(query, fea['zx'], 'zx')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)


    def forward(self, plane_features_list, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        if isinstance(plane_features_list, list):
            plane_features_geo, plane_features_color = plane_features_list
        else:
            plane_features_geo = plane_features_list[:, 0:3, ...]
            plane_features_color = plane_features_list[:, 3:, ...]
        pred_sdf = self.forward_sdf(plane_features_geo, xyz)
        pred_rgb = self.forward_rgb(plane_features_color, xyz)
        return pred_sdf, pred_rgb


    def forward_sdf(self, plane_features, xyz):
        sdf_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model_geo(sdf_features)
        return pred_sdf


    def forward_rgb(self, plane_features, xyz):
        features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_rgb = self.model_color(features)
        return pred_rgb


class SdfModelYhColor(pl.LightningModule):
    def __init__(self, config_json):
        super().__init__()
        with open(config_json, encoding='utf-8') as f:
            self.dataset_dict = edict(json.load(f))
        
        self.geo_config = self.dataset_dict.config.GeoCfg
        self.tex_config = self.dataset_dict.config.TexCfg
        self.geo_mlp = self.dataset_dict.config.GeoMLP
        self.tex_mlp = self.dataset_dict.config.TexMLP

        self.model_geo = SdfDecoderYh(d_in=self.geo_config.channels, 
                                  d_out=1, 
                                  d_hidden=self.geo_config.width, 
                                  n_layers=self.geo_config.n_layers, 
                                  skip_in=self.geo_config.skip_in,
                                  ).cuda()
        self.model_color = SdfDecoderYh(d_in=self.tex_config.channels, 
                                  d_out=3, 
                                  d_hidden=self.tex_config.width, 
                                  n_layers=self.tex_config.n_layers, 
                                  skip_in=self.tex_config.skip_in,
                                  ).cuda()


        self.model_geo.load_state_dict(torch.load(self.geo_mlp, map_location='cuda'))
        self.model_color.load_state_dict(torch.load(self.tex_mlp, map_location='cuda'))
        self.model_geo.eval()
        self.model_color.eval()
        print(self.model_color)

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
        if plane == 'zx':
            xy = p[:, :, [2, 0]]
        elif plane =='yx':
            xy = p[:, :, [1, 0]]
        else:
            xy = p[:, :, [1, 2]]

        return xy
        # # xy_new = xy / (1 + padding + 10e-6) # (-1, 1)
        # xy_new = xy
        # xy_new = xy_new + 1.0 # range (0, 2)

        # # f there are outliers out of the range
        # if xy_new.max() >= 2:
        #     xy_new[xy_new >= 2] = 2 - 10e-6
        # if xy_new.min() < 0:
        #     xy_new[xy_new < 0] = 0.0
        # return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane, padding=0.1):
        xy = self.normalize_coordinate2(query.clone(), plane=plane, padding=padding)
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # vgrid = xy - 1.0
        vgrid = xy
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        plane_features_geo = plane_features[:, 0:3, ...]
        plane_features_color = plane_features[:, 3:, ...]
        sdf_features = self.get_points_plane_features(plane_features_geo, xyz) # point_features: B, N, D
        pred_sdf = self.model(sdf_features)
        rgb_features = self.get_points_plane_features(plane_features_color, xyz)
        pred_rgb = self.model_color(rgb_features)
        return pred_sdf, pred_rgb


    def get_points_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        fea = {}
        fea['yx'], fea['zx'], fea['yz'] = plane_features[:, 0, ...], plane_features[:, 1, ...], plane_features[:, 2, ...]
        #print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])
        plane_feat_sum = 0
        plane_feat_sum += self.sample_plane_feature(query, fea['yx'], 'yx')
        plane_feat_sum += self.sample_plane_feature(query, fea['zx'], 'zx')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)


    def forward(self, plane_features_list, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        if isinstance(plane_features_list, list):
            plane_features_geo, plane_features_color = plane_features_list
        else:
            plane_features_geo = plane_features_list[:, 0:3, ...]
            plane_features_color = plane_features_list[:, 3:, ...]
        pred_sdf = self.forward_sdf(plane_features_geo, xyz)
        pred_rgb = self.forward_rgb(plane_features_color, xyz)
        return pred_sdf, pred_rgb


    def forward_sdf(self, plane_features, xyz):
        sdf_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model_geo(sdf_features)
        return pred_sdf


    def forward_rgb(self, plane_features, xyz):
        features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_rgb = self.model_color(features)
        return pred_rgb


class SdfModelYh(pl.LightningModule):
    def __init__(self, config_json, type="geo"):
        super().__init__()
        with open(config_json, encoding='utf-8') as f:
            self.dataset_dict = edict(json.load(f))
        if type == "geo":
            self.yhconfig = self.dataset_dict.config.GeoCfg
            ckpt_path = self.dataset_dict.config.GeoMLP
            dout = 1
        elif type == "tex":
            self.yhconfig = self.dataset_dict.config.TexCfg
            ckpt_path = self.dataset_dict.config.TexMLP
            dout = 3
        self.model = SdfDecoderYh(d_in=self.yhconfig.channels, 
                                  d_out=dout, 
                                  d_hidden=self.yhconfig.width, 
                                  n_layers=self.yhconfig.n_layers, 
                                  skip_in=self.yhconfig.skip_in,
                                  ).cuda()

        self.model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
        self.model.eval()
        print(self.model)

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
        if plane == 'zx':
            xy = p[:, :, [2, 0]]
        elif plane =='yx':
            xy = p[:, :, [1, 0]]
        else:
            xy = p[:, :, [1, 2]]

        return xy
        # # xy_new = xy / (1 + padding + 10e-6) # (-1, 1)
        # xy_new = xy
        # xy_new = xy_new + 1.0 # range (0, 2)

        # # f there are outliers out of the range
        # if xy_new.max() >= 2:
        #     xy_new[xy_new >= 2] = 2 - 10e-6
        # if xy_new.min() < 0:
        #     xy_new[xy_new < 0] = 0.0
        # return xy_new

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane, padding=0.1):
        xy = self.normalize_coordinate2(query.clone(), plane=plane, padding=padding)
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # vgrid = xy - 1.0
        vgrid = xy

        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model(point_features)
        return pred_sdf # [B, num_points] 


    def get_points_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        fea = {}
        fea['yx'], fea['zx'], fea['yz'] = plane_features[:, 0, ...], plane_features[:, 1, ...], plane_features[:, 2, ...]
        #print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])
        plane_feat_sum = 0
        plane_feat_sum += self.sample_plane_feature(query, fea['yx'], 'yx')
        plane_feat_sum += self.sample_plane_feature(query, fea['zx'], 'zx')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)


    def forward(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model(point_features)
        return pred_sdf # [B, num_points] 
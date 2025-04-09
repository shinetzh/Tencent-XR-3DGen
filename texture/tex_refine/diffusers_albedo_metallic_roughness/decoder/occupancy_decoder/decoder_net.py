import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import grad

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad

class TriplaneGrid(nn.Module):
    def __init__(self,
                 channels,
                 resolution=256,
                 n=1,):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.n = n
        self.plane_axes = torch.tensor(
            [[[0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]],
            [[0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]],
            [[0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]]]).float().cuda()
        
        ######## relax triplane grid ###########
        # init_mlp = nn.Sequential(
        #     nn.Linear(3, 32), nn.Softplus(beta=100),
        #     nn.Linear(32, 32), nn.Softplus(beta=100),
        #     nn.Linear(32, channels), #nn.Softplus(beta=100)
        # )
        # self.activation = nn.Softplus(beta=100)
        # nn.init.normal_(init_mlp[4].weight, 0.0, np.sqrt(2) / np.sqrt(channels))
        # nn.init.constant_(init_mlp[4].bias, 0.0)
        # nn.init.normal_(init_mlp[0].weight, 0.0, np.sqrt(2) / np.sqrt(32))
        # nn.init.constant_(init_mlp[0].bias, 0.0)
        # nn.init.normal_(init_mlp[2].weight, 0.0, np.sqrt(2) / np.sqrt(32))
        # nn.init.constant_(init_mlp[2].bias, 0.0)
        # init_mlp[0] = nn.utils.weight_norm(init_mlp[0])
        # init_mlp[2] = nn.utils.weight_norm(init_mlp[2])
        # init_mlp[4] = nn.utils.weight_norm(init_mlp[4])
        # init_mlp = init_mlp.cuda()
        # X = torch.linspace(-1., 1., resolution)
        # inputs = torch.stack(torch.meshgrid(X, X, X), -1).cuda().reshape(-1,3) # [256,256,3]
        # outputs = init_mlp(inputs).reshape(resolution, resolution, resolution, channels).permute(3,0,1,2) # [3xRxR,C]
        # self.triplane = nn.Parameter(outputs[None, ...].expand(n, -1, -1, -1, -1).detach().cpu())
        ########################################

        ######## relax triplane mlp ############
        # self.triplane = []
        # for i in range(n):
        #     self.triplane.append(
        #         nn.Sequential(
        #             nn.Linear(3, 32), nn.Softplus(beta=100),
        #             nn.Linear(32,32), nn.Softplus(beta=100),
        #             nn.Linear(32, channels), nn.Softplus(beta=100),
        #         ).cuda()
        #     )
        #     nn.init.normal_(self.triplane[-1][4].weight, 0.0, np.sqrt(2) / np.sqrt(channels))
        #     nn.init.constant_(self.triplane[-1][4].bias, 0.0)
        #     nn.init.normal_(self.triplane[-1][0].weight, 0.0, np.sqrt(2) / np.sqrt(32))
        #     nn.init.constant_(self.triplane[-1][0].bias, 0.0)
        #     nn.init.normal_(self.triplane[-1][2].weight, 0.0, np.sqrt(2) / np.sqrt(32))
        #     nn.init.constant_(self.triplane[-1][2].bias, 0.0)
        #     self.triplane[-1][4] = nn.utils.weight_norm(self.triplane[-1][4])
        #     self.triplane[-1][0] = nn.utils.weight_norm(self.triplane[-1][0])
        #     self.triplane[-1][2] = nn.utils.weight_norm(self.triplane[-1][2])
        #########################################

        # #########################################
        # sphere init method 2:
        init_mlp = nn.Sequential(
            nn.Linear(3, 128), nn.Softplus(beta=100),
            nn.Linear(128, channels),
        )
        nn.init.normal_(init_mlp[2].weight, 0.0, np.sqrt(2) / np.sqrt(channels))
        nn.init.constant_(init_mlp[2].bias, 0.0)
        nn.init.normal_(init_mlp[0].weight, 0.0, np.sqrt(2) / np.sqrt(128))
        nn.init.constant_(init_mlp[0].bias, 0.0)
        init_mlp[0] = nn.utils.weight_norm(init_mlp[0])
        init_mlp[2] = nn.utils.weight_norm(init_mlp[2])
        init_mlp = init_mlp.cuda()
        X = torch.linspace(-1., 1., resolution)
        MG = torch.meshgrid(X, X) # [(256,256), (256,256)]
        Ze = torch.zeros_like(MG[0]) # (256,256)
        plane1 = torch.stack([MG[0], MG[1], Ze], -1).reshape(-1,3) # (x,y,z=0)
        plane2 = torch.stack([MG[0], Ze, MG[1]], -1).reshape(-1,3) # (x,y=0,z)
        plane3 = torch.stack([Ze, MG[1], MG[0]], -1).reshape(-1,3) # (x=0,y,z)
        inputs = torch.cat([plane1, plane2, plane3], 0).cuda() # [3xRxR,3]
        outputs = init_mlp(inputs).reshape(3, resolution, resolution, channels).permute(0,3,1,2) # [3xRxR,C]->[3,C,R,R]
        self.triplane = nn.Parameter(outputs[None, ...].expand(n, -1, -1, -1, -1).detach().cpu() / 3)
        # #########################################
        # self.triplane = nn.Parameter(torch.rand([n, 3, channels, resolution, resolution]))



    # y,x; z,x; y,z
    def project_onto_planes(self, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = self.plane_axes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
        inv_planes = torch.linalg.inv(self.plane_axes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    def upscale(self, new_reso):
        triplane = self.triplane.view(-1, self.channels, self.resolution, self.resolution).detach()
        new_triplane = F.interpolate(triplane, (new_reso, new_reso), mode='bilinear', align_corners=True)
        self.resolution = new_reso
        self.triplane = nn.Parameter(new_triplane.reshape(-1,3,self.channels, self.resolution, self.resolution))

    def forward(self, inputs):
        # inputs: [N,3]
        # ====================================
        coord = self.project_onto_planes(inputs[None,...]).unsqueeze(1)
        triplane_feat = self.triplane.view(1*3, self.channels, self.resolution, self.resolution).clamp(-1.0, 1.0)
        feat = F.grid_sample(
            triplane_feat,
            coord.float(),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ).permute(0,3,2,1).reshape(1,3,-1,self.channels)
        feat = feat.sum(1).reshape(-1, self.channels)
        feat_final = torch.cat([feat], dim=-1)

        # for i in range(self.channels):
        #     feat_final[:, i:i+1] *= (1.-math.cos(
        #         math.pi*(max(min(alpha_ratio*self.channels-i, 1.), 0.))
        #     )) * .5
            # feat_final[:, self.channels+i:self.channels+i+1] *= (1.-math.cos(
            #     math.pi*(max(min(alpha_ratio*self.channels-i, 1.), 0.))
            # )) * .5
        # ====================================
        
        # ====================================
        # relax triplane mlp
        # feat_final = self.triplane[oid](inputs)
        # ====================================

        # ====================================
        # relax triplane grid
        # feat = F.grid_sample(
        #     self.triplane[oid:oid+1], 
        #     inputs.reshape(1,1,1,-1,3), 
        #     mode='bilinear', 
        #     align_corners=True
        # ).reshape(self.channels, -1).permute(1,0)
        # feat_final = self.activation(feat)
        # ====================================

        return feat_final

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SdfDecoderYh(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=[],
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SdfDecoderYh, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, inputs):     
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        if self.d_out == 3:
            x = torch.sigmoid(x)
        return x


def gradient_tri(net, x, oid, alpha_ratio, triplane=None):
    eps = 1e-6
    x_nei = torch.stack([
        x + torch.as_tensor([[eps, 0.0, 0.0]]).to(x),
        x + torch.as_tensor([[-eps, 0.0, 0.0]]).to(x),
        x + torch.as_tensor([[0.0, eps, 0.0]]).to(x),
        x + torch.as_tensor([[0.0, -eps, 0.0]]).to(x),
        x + torch.as_tensor([[0.0, 0.0, eps]]).to(x),
        x + torch.as_tensor([[0.0, 0.0, -eps]]).to(x)
    ], dim=0).view(-1,3)
    sdf_nei = net(x_nei, oid, alpha_ratio, triplane)
    sdf_nei = sdf_nei.view(6,-1,1)
    gradients = torch.cat([
        0.5 * (sdf_nei[0] - sdf_nei[1]) / eps,
        0.5 * (sdf_nei[2] - sdf_nei[3]) / eps,
        0.5 * (sdf_nei[4] - sdf_nei[5]) / eps,
    ], dim=-1)
    return gradients
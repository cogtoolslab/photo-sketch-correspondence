""" Extracted from NC-Net. """
from __future__ import print_function, division
import numpy as np
import random
import torch
import torch.nn.functional as F

from torch.nn.modules.module import Module


def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))


def mask_outlier(flow):
    mask = (flow < -1.0) & (flow > 1.0)
    flow[mask] = 1e4
    return flow


def flow_grid_sample(baseflow, flow, mode, padding_mode, align_corners):
    out = F.grid_sample(baseflow.permute(0, 3, 1, 2), flow, mode=mode, padding_mode=padding_mode, align_corners=align_corners).permute(0, 2, 3, 1)
    return out


class AffineGridGen(Module):
    """Dense correspondence map generator, corresponding to an affine transform."""
    def __init__(self, out_h=240, out_w=240, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        b = theta.size()[0]
        if not theta.size() == (b, 2, 3):
            theta = theta.view(-1, 2, 3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)


class AffineGridGenV2(Module):
    """Dense correspondence map generator, corresponding to an affine  transform."""
    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(AffineGridGenV2, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: load_size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self, theta):
        b = theta.size(0)
        if not theta.size() == (b, 6):
            theta = theta.view(b, 6)
            theta = theta.contiguous()

        t0 = theta[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1 = theta[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2 = theta[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3 = theta[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4 = theta[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5 = theta[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X * t0 + grid_Y * t1 + t2
        grid_Yp = grid_X * t3 + grid_Y * t4 + t5

        return torch.cat((grid_Xp, grid_Yp), 3)


class HomographyGridGen(Module):
    """Dense correspondence map generator, corresponding to a homography transform."""
    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: load_size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self, theta):
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0 = H[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, 7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, 8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)

        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8

        grid_Xp /= k
        grid_Yp /= k

        return torch.cat((grid_Xp, grid_Yp), 3)


def homography_mat_from_4_pts(theta):
    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()

    xp = theta[:, :4].unsqueeze(2);
    yp = theta[:, 4:].unsqueeze(2)

    x = torch.FloatTensor([-1, -1, 1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    y = torch.FloatTensor([-1, 1, -1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    z = torch.zeros(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    o = torch.ones(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    single_o = torch.ones(1).unsqueeze(1).unsqueeze(0).expand(b, 1, 1)

    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()

    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
                   torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
    # add h33
    h = torch.cat([h, single_o], 1)

    H = h.squeeze(2)

    return H


class TpsGridGen(Module):
    """Dense correspondence map generator, corresponding to a TPS transform.
    https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/pytorch.py"""
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: load_size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # load_size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # load_size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_X.requires_grad = False
            self.P_Y.requires_grad = False
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        if self.reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * self.reg_factor
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: load_size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: load_size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: load_size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


class ComposedGeometricTnf(object):
    """
    Composed geometric transformation (affine+tps)
    """

    def __init__(self, tps_grid_size=3, tps_reg_factor=0, out_h=240, out_w=240,
                 offset_factor=1.0,
                 padding_crop_factor=None,
                 use_cuda=True):
        self.padding_crop_factor = padding_crop_factor

        self.affTnf = GeometricTnf(out_h=out_h, out_w=out_w,
                                   geometric_model='affine',
                                   offset_factor=offset_factor if padding_crop_factor is None else padding_crop_factor,
                                   use_cuda=use_cuda)

        self.tpsTnf = GeometricTnf(out_h=out_h, out_w=out_w,
                                   geometric_model='tps',
                                   tps_grid_size=tps_grid_size,
                                   tps_reg_factor=tps_reg_factor,
                                   offset_factor=offset_factor if padding_crop_factor is None else 1.0,
                                   use_cuda=use_cuda)

    def __call__(self, theta_aff, theta_aff_tps, use_cuda=True):
        sampling_grid_aff = self.affTnf(image_batch=None,
                                        theta_batch=theta_aff.view(-1, 2, 3),
                                        return_sampling_grid=True,
                                        return_warped_image=False)

        sampling_grid_aff_tps = self.tpsTnf(image_batch=None,
                                            theta_batch=theta_aff_tps,
                                            return_sampling_grid=True,
                                            return_warped_image=False)

        if self.padding_crop_factor is not None:
            sampling_grid_aff_tps = sampling_grid_aff_tps * self.padding_crop_factor

        # put 1e10 value in region out of bounds of sampling_grid_aff

        # compose transformations
        sampling_grid_aff_tps_comp = flow_grid_sample(sampling_grid_aff, sampling_grid_aff_tps,
                                                      padding_mode="border", align_corners=True, mode="bilinear")

        return sampling_grid_aff_tps_comp


class GeometricTnf(object):
    """
    Geometric transformation to an image batch (wrapped in a PyTorch tensor)
    ( can be used with no transformation to perform bilinear resizing )
    """

    def __init__(self, geometric_model='affine', tps_grid_size=3, tps_reg_factor=0, out_h=240, out_w=240,
                 offset_factor=None, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda
        self.offset_factor = offset_factor

        if geometric_model == 'affine' and offset_factor is None:
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'affine' and offset_factor is not None:
            self.gridGen = AffineGridGenV2(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'hom':
            self.gridGen = HomographyGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'tps':
            self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, grid_size=tps_grid_size,
                                      reg_factor=tps_reg_factor, use_cuda=use_cuda)
        if offset_factor is not None:
            self.gridGen.grid_X = self.gridGen.grid_X / offset_factor
            self.gridGen.grid_Y = self.gridGen.grid_Y / offset_factor

        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, out_h=None, out_w=None, return_warped_image=True,
                 return_sampling_grid=False, padding_factor=1.0, crop_factor=1.0):
        if image_batch is None:
            b = 1
        else:
            b = image_batch.size(0)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b, 2, 3).contiguous()
            theta_batch.requires_grad=False

            # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h != self.out_h or out_w != self.out_w):
            if self.geometric_model == 'affine':
                gridGen = AffineGridGen(out_h, out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'hom':
                gridGen = HomographyGridGen(out_h, out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'tps':
                gridGen = TpsGridGen(out_h, out_w, use_cuda=self.use_cuda)
        else:
            gridGen = self.gridGen

        sampling_grid = gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        if padding_factor != 1 or crop_factor != 1:
            sampling_grid = sampling_grid * (padding_factor * crop_factor)
        # rescale grid according to offset_factor
        if self.offset_factor is not None:
            sampling_grid = sampling_grid * self.offset_factor
        sampling_grid = mask_outlier(sampling_grid)

        if return_sampling_grid and not return_warped_image:
            return sampling_grid

        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid, align_corners=True)

        if return_sampling_grid and return_warped_image:
            return (warped_image_batch, sampling_grid)

        return warped_image_batch


class SynthecticAffHomoTPSTransfo:
    """Generates a flow field of given load_size, corresponding to a randomly sampled Affine, Homography, TPS or Affine-TPS
    transformation. """
    def __init__(self, size_output_flow=(480, 640), random_t=0.25, random_s=(0.5, 1.5), random_alpha=np.pi / 12,
                 random_t_tps_for_afftps=None, random_t_hom=0.4, random_t_tps=0.4, tps_grid_size=3, tps_reg_factor=0,
                 flip=False, transformation_types=None, parametrize_with_gaussian=False, use_cuda=True):
        """
        For all transformation parameters, image is taken as in interval [-1, 1]. Therefore all parameters must be
        within [0, 1]. The range of sampling is then [-parameter, parameter] or [1-parameter, 1+parameter] for the
        scale.
        Args:
            size_output_flow: desired output load_size for generated flow field
            random_t: max translation for affine transform.
            random_s: max scale for affine transform
            random_alpha: max rotation and shearing angle for the affine transform
            random_t_tps_for_afftps: max translation parameter for the tps transform generation, used for the
                         affine-tps transforms
            random_t_hom: max translation parameter for the homography transform generation
            random_t_tps: max translation parameter for the tps transform generation
            tps_grid_size: tps grid load_size
            tps_reg_factor:
            transformation_types: list of transformations to samples.
                                  Must be selected from ['affine', 'hom', 'tps', 'afftps']
            parametrize_with_gaussian: sampling distribution for the transformation parameters. Gaussian ? otherwise,
                                       uses a uniform distribution
            use_cuda: use_cuda?
        """

        if not isinstance(size_output_flow, tuple):
            size_output_flow = (size_output_flow, size_output_flow)
        self.out_h, self.out_w = size_output_flow
        self.parametrize_with_gaussian = parametrize_with_gaussian
        # for homo
        self.random_t_hom = random_t_hom
        # for tps
        self.random_t_tps = random_t_tps
        # for affine
        self.random_t = random_t
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.tps_grid_size = tps_grid_size
        if random_t_tps_for_afftps is None:
            random_t_tps_for_afftps = random_t_tps
        self.random_t_tps_for_afftps = random_t_tps_for_afftps
        self.use_cuda = use_cuda
        self.flip = flip
        if transformation_types is None:
            transformation_types = ['affine', 'hom', 'tps', 'afftps']
        self.transformation_types = transformation_types
        if 'hom' in self.transformation_types:
            self.homo_grid_sample = HomographyGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)
        if 'affine' in self.transformation_types:
            self.aff_grid_sample = AffineGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)
        # self.aff_grid_sample_2 = AffineGridGenV2(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda) for offset
        if 'tps' in self.transformation_types:
            self.tps_grid_sample = TpsGridGen(out_h=self.out_h, out_w=self.out_w, grid_size=tps_grid_size,
                                              reg_factor=tps_reg_factor, use_cuda=use_cuda)
        if 'afftps' in self.transformation_types:
            self.tps_aff_grid_sample = ComposedGeometricTnf(tps_grid_size=tps_grid_size, tps_reg_factor=tps_reg_factor,
                                                            out_h=self.out_h, out_w=self.out_w, offset_factor=1.0,
                                                            padding_crop_factor=None, use_cuda=use_cuda)

    def __call__(self, *args, **kwargs):
        """Generates a flow_field (flow_gt) from sampling a geometric transformation. """

        geometric_model = self.transformation_types[random.randrange(0, len(self.transformation_types))]
        # sample the theta
        theta_tps, theta_hom, theta_aff = 0.0, 0.0, 0.0
        if self.parametrize_with_gaussian:
            if geometric_model == 'affine' or geometric_model == 'afftps':
                rot_angle = np.random.normal(0, self.random_alpha, 1)
                sh_angle = np.random.normal(0, self.random_alpha, 1)

                # use uniform, because gaussian distribution is unbounded
                lambda_1 = np.random.uniform(self.random_s[0], self.random_s[1],
                                             1)  # between 0.75 and 1.25 for random_s = 0.25
                lambda_2 = np.random.uniform(self.random_s[0], self.random_s[1], 1)  # between 0.75 and 1.25
                tx = np.random.normal(0, self.random_t, 1)
                ty = np.random.normal(0, self.random_t, 1)

                R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
                R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

                D = np.diag([lambda_1[0], lambda_2[0]])

                A = R_alpha @ R_sh.transpose() @ D @ R_sh

                theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])
                theta_aff = torch.Tensor(theta_aff.astype(np.float32)).unsqueeze(0)
                theta_aff = theta_aff.cuda() if self.use_cuda else theta_aff
            if geometric_model == 'hom':
                theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
                theta_hom = theta_hom + np.random.normal(0, self.random_t_hom, 8)
                theta_hom = torch.Tensor(theta_hom.astype(np.float32)).unsqueeze(0)
                theta_hom = theta_hom.cuda() if self.use_cuda else theta_hom
            if geometric_model == 'tps':
                x = np.linspace(-1.0, 1.0, self.tps_grid_size)
                y = np.linspace(-1.0, 1.0, self.tps_grid_size)
                X, Y = np.meshgrid(x, y)
                theta_tps = np.concatenate([Y, X]).flatten()
                theta_tps = theta_tps + np.random.normal(0, self.random_t_tps, self.tps_grid_size * self.tps_grid_size * 2)
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps
            if geometric_model == 'afftps':
                x = np.linspace(-1.0, 1.0, self.tps_grid_size)
                y = np.linspace(-1.0, 1.0, self.tps_grid_size)
                X, Y = np.meshgrid(x, y)
                theta_tps = np.concatenate([Y, X]).flatten()
                theta_tps = theta_tps + np.random.normal(0, self.random_t_tps_for_afftps, self.tps_grid_size * self.tps_grid_size * 2)
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps
        else:
            if geometric_model == 'affine' or geometric_model == 'afftps':
                rot_angle = (np.random.rand(1) - 0.5) * 2 * self.random_alpha
                # between -np.pi/12 and np.pi/12 for random_alpha = np.pi/12
                sh_angle = (np.random.rand(1) - 0.5) * 2 * self.random_alpha
                lambda_1 = np.random.uniform(self.random_s[0], self.random_s[1],
                                             1)  # between 0.75 and 1.25 for random_s = 0.25
                lambda_2 = np.random.uniform(self.random_s[0], self.random_s[1], 1)  # between 0.75 and 1.25
                tx = (2 * np.random.rand(1) - 1) * self.random_t  # between -0.25 and 0.25 for random_t=0.25
                ty = (2 * np.random.rand(1) - 1) * self.random_t


                R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
                R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

                D = np.diag([lambda_1[0], lambda_2[0]])

                A = R_alpha @ R_sh.transpose() @ D @ R_sh

                theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])
                theta_aff = torch.Tensor(theta_aff.astype(np.float32)).unsqueeze(0)
                theta_aff = theta_aff.cuda() if self.use_cuda else theta_aff
            if geometric_model == 'hom':
                theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
                theta_hom = theta_hom + (np.random.rand(8) - 0.5) * 2 * self.random_t_hom
                theta_hom = torch.Tensor(theta_hom.astype(np.float32)).unsqueeze(0)
                theta_hom = theta_hom.cuda() if self.use_cuda else theta_hom

            if geometric_model == 'tps':
                x = np.linspace(-1.0, 1.0, self.tps_grid_size)
                y = np.linspace(-1.0, 1.0, self.tps_grid_size)
                X, Y = np.meshgrid(x, y)
                theta_tps = np.concatenate([Y, X]).flatten()
                theta_tps = theta_tps + (np.random.rand(self.tps_grid_size * self.tps_grid_size * 2) - 0.5) * 2 * self.random_t_tps
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps
            if geometric_model == 'afftps':
                x = np.linspace(-1.0, 1.0, self.tps_grid_size)
                y = np.linspace(-1.0, 1.0, self.tps_grid_size)
                X, Y = np.meshgrid(x, y)
                theta_tps = np.concatenate([Y, X]).flatten()
                theta_tps = theta_tps + (np.random.rand(self.tps_grid_size * self.tps_grid_size * 2) - 0.5) * 2 * self.random_t_tps_for_afftps
                theta_tps = torch.Tensor(theta_tps.astype(np.float32)).unsqueeze(0)
                theta_tps = theta_tps.cuda() if self.use_cuda else theta_tps

        if geometric_model == 'hom':
            flow_gt = self.homo_grid_sample.forward(theta_hom)
            # flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2xhw
        elif geometric_model == 'affine':
            flow_gt = self.aff_grid_sample.forward(theta_aff)
            # flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2xhw

        elif geometric_model == 'tps':
            flow_gt = self.tps_grid_sample.forward(theta_tps)
            # flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2xhw

        elif geometric_model == 'afftps':
            flow_gt = self.tps_aff_grid_sample(theta_aff, theta_tps)
            # flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2xhw

        else:
            raise NotImplementedError

        if self.flip:
            if torch.rand(1) < 0.5:
                flow_gt = torch.flip(flow_gt, dims=[2])

        return flow_gt


class FlowComposition:
    def __init__(self, list_of_gens):
        self.gens = list_of_gens

    def __call__(self, *args, **kwargs):
        flow = self.gens[0]()
        for i in range(1, len(self.gens)):
            flow2 = self.gens[i]()
            flow = flow_grid_sample(flow, flow2, mode="bilinear", padding_mode="border", align_corners=True)
        return flow




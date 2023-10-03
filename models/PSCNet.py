import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_outlier(flow):
    """
    Mask the out-of-bound flow with 1e4
    :param flow: the displacement field
    :return: the masked displacement field
    """
    mask = (flow < -1.0) & (flow > 1.0)
    flow[mask] = 1e4
    return flow


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, groups: int = 1, bias=False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=bias)


class STN(nn.Module):
    """
    The Spatial Transformer Network (STN) that computes correlation between feature maps
    and estimates displacement field.
    """
    def __init__(self, corr_layer, feat_size, stn_size, stn_layer):
        """
        :param corr_layer: (list of ints) The feature layer(s) that we compute the correlations over.
                           Normally set to [3] or [2, 3].
        :param feat_size: (int) Size of the feature map. Default: 16.
        :param stn_size: (int) Size of the predicted displacement field. Default: 16.
        :param stn_layer: (int) Number of STN layers. Default: 5.
        """
        super(STN, self).__init__()
        self.corr_layer = corr_layer
        self.feat_size = feat_size
        self.stn_size = stn_size

        # build STN blocks at the scale of 4x4, 8x8, and 16x16
        self.net_4x = self._build_block(4, stn_layer)
        self.net_8x = self._build_block(8, stn_layer)
        self.net_16x = self._build_block(16, stn_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # generate an identity displacement field
        self.register_buffer("base_map",
                             F.affine_grid(torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0),
                                           [1, 1, self.stn_size, self.stn_size], align_corners=True))

    def _build_block(self, output_size, n_layers):
        ds = self.feat_size // output_size
        assert ds in [1, 2, 4, 8, 16]
        dims = self.feat_size * self.feat_size * len(self.corr_layer)

        layers = [
            nn.Conv2d(dims, dims, ds, stride=ds),
            nn.BatchNorm2d(dims),
            nn.LeakyReLU(0.1),
        ]

        for i in range(n_layers):
            layers += [
                conv3x3(dims, dims // 2),
                nn.BatchNorm2d(dims // 2),
                nn.LeakyReLU(0.1)
            ]
            dims = dims // 2

        layers += [
            conv3x3(dims, 2, bias=True),
            nn.Upsample(size=(self.stn_size, self.stn_size), mode="bilinear", align_corners=True)
        ]

        return nn.Sequential(*layers)

    def corr(self, map1, map2):
        '''
        Computes the correlation matrix from the N selected feature layers
        :param map1: Dict of feature maps from source image
        :param map2: Dict of feature maps from target image
        :return: The concatenated correlation matrix (B, W * H * N, W, H)
        '''
        corrs = []

        map1 = self.interpolate(map1, (self.feat_size, self.feat_size))
        map2 = self.interpolate(map2, (self.feat_size, self.feat_size))

        for i in self.corr_layer:
            map1_ = map1["layer%i" % i]
            map2_ = map2["layer%i" % i]
            N, C, W, H = map1_.shape
            map1_ = map1_.permute(0, 2, 3, 1).contiguous().view(N, W * H, C)
            map2_ = map2_.view(N, C, W * H)
            corr = torch.matmul(map1_, map2_).view(N, W * H, W, H)
            corrs.append(corr)
        corrs = torch.cat(corrs, dim=1)

        return corrs

    def interpolate(self, map1, size):
        """
        Interpolate each feature layer to the output size.
        :param map1: Dict of feature maps.
        :param size: (tuple of ints) Output size of feature map.
        :return:
        """
        map1_ = {}
        for i in self.corr_layer:
            map1_["layer%i" % i] = F.interpolate(map1["layer%i" % i], size, mode="bilinear", align_corners=False)
        return map1_

    def grid_sample(self, map1, flow):
        """
        Warp the feature maps using the flow (displacement field).
        :param map1: Dict of feature maps.
        :param flow: Displacement field.
        :return:
        """
        map1_ = {}
        for i in self.corr_layer:
            map1_["layer%i" % i] = F.grid_sample(map1["layer%i" % i], flow,
                                                 mode="bilinear", padding_mode="zeros", align_corners=True)
        return map1_

    def forward(self, map1, map2, training=False):

        # computes displacement field at scale 4x4
        corr_4x = self.corr(map2, map1)
        # init with the identity displacement field
        flow_4x = self.net_4x(corr_4x).permute(0, 2, 3, 1) + self.base_map
        flow_4x = mask_outlier(flow_4x)

        # computes displacement field at scale 8x8
        # warp source feature maps with the 4x4 displacement field, so the block only predicts the residue.
        map1_8x = self.grid_sample(map1, flow_4x)
        corr_8x = self.corr(map2, map1_8x)
        flow_8x = self.net_8x(corr_8x).permute(0, 2, 3, 1)
        flow_8x = flow_4x + flow_8x
        flow_8x = mask_outlier(flow_8x)

        # computes displacement field at scale 16x16
        # warp source feature maps with the 8x8 displacement field, so the block only predicts the residue.
        map1_16x = self.grid_sample(map1, flow_8x)
        corr_16x = self.corr(map2, map1_16x)
        flow_16x = self.net_16x(corr_16x).permute(0, 2, 3, 1)
        flow_16x = flow_8x + flow_16x
        flow_16x = mask_outlier(flow_16x)

        # return multi-scale flows during training.
        if training:
            return flow_16x, flow_8x, flow_4x
        else:
            return flow_16x


class PSCNet(nn.Module):
    """
    The overall network with feature encoder and warp estimator.
    """
    def __init__(self, framework, backbone, dim, corr_layer, feat_size=16, stn_size=16,
                 pretrained_encoder="", stn_layer=5,
                 replace_stride_with_dilation=[False, False, False], **kwargs):
        """

        :param framework: Feature encoder training framework. Default: MoCo v2.
        :param backbone: Feature encoder backbone. Default: ResNet w/ conditional BN.
        :param dim: (int) MoCo dim. Default: 128.
        :param corr_layer: (list of ints) The feature layer(s) that we compute the correlations over.
                           Normally set to [3] or [2, 3].
        :param feat_size: (int) Size of the feature map. Default: 16.
        :param stn_size: (int) Size of the predicted displacement field. Default: 16.
        :param stn_layer: (int) Number of STN layers. Default: 5.
        :param pretrained_encoder: (str) Path to a pretrained encoder weights. Default: "".
        :param replace_stride_with_dilation: (list of bools) Whether to replace the stride with dilation
                                             in ResNet block 2, 3, 4. Used for larger feature maps.
        :param kwargs: Other parameters to the MoCo framework.
        """
        super().__init__()

        self.corr_layer = corr_layer
        self.feat_size = feat_size
        self.stn_size = stn_size

        self.encoder_q = backbone(num_classes=dim, pretrained=pretrained_encoder,
                                  replace_stride_with_dilation=replace_stride_with_dilation)
        self.encoder_k = backbone(num_classes=dim, pretrained=pretrained_encoder,
                                  replace_stride_with_dilation=replace_stride_with_dilation)

        self.encoder_q.fc = nn.Identity()
        self.encoder_k.fc = nn.Identity()

        self.framework = framework(self.encoder_q, self.encoder_k, dim, **kwargs)

        self.stn = STN(corr_layer, feat_size, stn_size, stn_layer)

        self.register_buffer("pos_map", F.affine_grid(torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0),
                                                      [1, 1, stn_size, stn_size], align_corners=True).permute(0, 3, 1, 2))

    def forward_framework(self, im_q, im_k, cond_q, cond_k):
        """
        Forward the MoCo framework.
        :param im_q: Query images.
        :param im_k: Key images.
        :param cond_q: BN condition for query images (photo or sketch).
        :param cond_k: BN condition for key images (photo or sketch).
        :return: Logits, target, query feature maps, key feature maps (See MoCo framework for details)
        """
        output, target, res1, res2 = self.framework(im_q=im_q, im_k=im_k,
                                                    cond_q=cond_q, cond_k=cond_k,
                                                    return_map=True)

        return output, target, res1, res2

    def forward_backbone(self, im, cond, corr_only=False):
        """
        Forward the feature encoder backbone.
        :param im: Images.
        :param cond: Condition of BN (photo or sketch).
        :param corr_only: Return feature maps up to the wanted layers only. Used for faster computation.
        :return: FC features, feature maps.
        """
        fc, res = self.framework.encoder_q(im, cond, return_map=True,
                                           corr_layer=self.corr_layer if corr_only else None)

        return fc, res

    def forward_stn(self, map1, map2, dense_mtx=False):
        """
        Forward the STN.
        :param map1: Source feature maps.
        :param map2: Target feature maps.
        :param dense_mtx: (bool) Return dense correspondence matrix.
                          (Only for visualization. We compute error metric with a more accurate method.)
        :return: Forward flow, backward flow, and (optionally) dense correspondence matrix.
        """

        fwd_flow = self.stn(map1, map2)
        bwd_flow = self.stn(map2, map1)

        if dense_mtx:
            dist = self.stoch_dist([fwd_flow])
            return fwd_flow, bwd_flow, dist
        else:
            return fwd_flow, bwd_flow

    def stoch_dist(self, fwd_flows):
        """
        Find the dense correspondence matrix from displacement field.
        Only for visualization. We compute error metric with a more accurate method.
        :param fwd_flows: Forward displacement field.
        :return: The dense correspondence matrix.
        """
        N = fwd_flows[0].shape[0]
        pos_map = self.pos_map.repeat(N, 1, 1, 1)
        fwd_map = F.grid_sample(pos_map, fwd_flows[0], padding_mode="border", align_corners=True)

        for fwd_flow in fwd_flows[1:]:
            fwd_map = F.grid_sample(fwd_map, fwd_flow, padding_mode="border", align_corners=True)

        dist = torch.cdist(pos_map.permute(0, 2, 3, 1).view(N, -1, 2), fwd_map.permute(0, 2, 3, 1).view(N, -1, 2))
        return dist

    def compute_similarity(self, map1, map2):
        """
        Compute similarity and weight map at the selected feature layers.
        :param map1: Dict of feature maps.
        :param map2: Dict of feature maps.
        :return: The similarity and weight map.
        """
        map1_list = []
        map2_list = []
        for i in self.corr_layer:
            map1_list.append(F.interpolate(map1["layer%i" % i], (self.stn_size, self.stn_size), mode="bilinear"))
            map2_list.append(F.interpolate(map2["layer%i" % i], (self.stn_size, self.stn_size), mode="bilinear"))
        map1 = torch.cat(map1_list, dim=1)
        map2 = torch.cat(map2_list, dim=1)

        map1 = F.interpolate(map1, (self.stn_size, self.stn_size), mode="bilinear")
        map2 = F.interpolate(map2, (self.stn_size, self.stn_size), mode="bilinear")

        # compute similarity and weight map
        N, C, W, H = map1.shape
        D = W * H

        map1 = map1.permute(0, 2, 3, 1).contiguous().view(N, D, C)
        map2 = map2.view(N, C, D)

        corr = torch.matmul(map1, map2).view(N, D, D)
        corr = F.normalize(corr, dim=1, p=1)
        corr = F.normalize(corr, dim=2, p=1)
        weight = torch.max(corr, dim=2)[0]
        weight = weight - weight.min()
        weight = weight / weight.max()
        return [corr], [weight.detach()]

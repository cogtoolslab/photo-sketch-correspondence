import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import kornia.augmentation as aug

from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from models.PSCNet import mask_outlier
from utils.visualization import gen_graph, prep_img_tensor, prep_feat_tensor
from utils.spatial_transforms import SynthecticAffHomoTPSTransfo
from utils.meters import AverageMeter, ProgressMeter, accuracy
from utils.pck import proj_kps, compute_pck


def detach(tensor_dict):
    """
    Detach all tensors in a dict
    :param tensor_dict: Dict of tensors
    :return: Dict of detached tensors
    """
    for key in tensor_dict.keys():
        tensor_dict[key] = tensor_dict[key].detach()
    return tensor_dict





def step_pair(data, color_aug, syn_flow_gen, i, model, criterion, args, mem):
    """
    Run one step of encoder training under pair supervision
    :param data: Images.
    :param color_aug: Data augmentation in color.
    :param syn_flow_gen: Synthetic warp generator (Affine + TPS).
    :param i: Index of step.
    :param model: The model.
    :param criterion: Dict of loss criterion.
    :param args: Additional arguments
    :param mem: A memory dict that store data for visualization during training.
    :return: Contrastive loss, acc@1, acc@5, memory dict
    """
    raw1, raw2 = data

    if args.gpu is not None:
        raw1 = raw1.cuda(args.gpu, non_blocking=True)
        raw2 = raw2.cuda(args.gpu, non_blocking=True)

    # generate synthetic flows
    syn_flow_src = torch.cat([mask_outlier(syn_flow_gen()) for _ in range(len(raw1))], dim=0)
    syn_flow_dst = torch.cat([mask_outlier(syn_flow_gen()) for _ in range(len(raw1))], dim=0)

    # switch combination of key/query ~ photo/sketch
    if i % 2 == 0:
        cond_src = 0  # photo
        cond_dst = 1  # sketch
        images_src = raw1  # query
        images_dst = raw2  # key

    else:
        cond_src = 1
        cond_dst = 0
        images_src = raw2
        images_dst = raw1

    # color augmentation
    images_src = color_aug(images_src)
    images_dst = color_aug(images_dst)

    with torch.cuda.amp.autocast():
        # spatial augmentation (using synthetic flow)
        images_src = F.grid_sample(images_src.half(),
                                   syn_flow_src.half(),
                                   mode="bilinear", padding_mode="border", align_corners=True)
        images_dst = F.grid_sample(images_dst.half(),
                                   syn_flow_dst.half(),
                                   mode="bilinear", padding_mode="border", align_corners=True)

        output, target, res_src, res_dst = model.module.forward_framework(im_q=images_src,
                                                                          im_k=images_dst,
                                                                          cond_q=cond_src,
                                                                          cond_k=cond_dst)

        clr_loss = criterion["ce"](output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # add data to the memory dict for visualization during training
        if len(mem["image1"]) < 10:
            mem["image1"].append(prep_img_tensor(images_src))
            mem["image2"].append(prep_img_tensor(images_dst))
            mem["warp_image12"].append(prep_img_tensor(images_src))
            mem["warp_image21"].append(prep_img_tensor(images_dst))
            mem["res2_1"].append(prep_feat_tensor(res_src["layer2"]))
            mem["res2_2"].append(prep_feat_tensor(res_dst["layer2"]))
            mem["res3_1"].append(prep_feat_tensor(res_src["layer3"]))
            mem["res3_2"].append(prep_feat_tensor(res_dst["layer3"]))
            mem["weight3_1"].append(None)
            mem["weight3_2"].append(None)
            mem["dist"].append(None)

        return clr_loss, acc1, acc5, mem


def step_instance(data, color_aug, syn_flow_gen, i, model, criterion, args, mem):
    """
    Run one step of encoder training under instance supervision (for ablation only)
    :param data: Images.
    :param color_aug: Data augmentation in color.
    :param syn_flow_gen: Synthetic warp generator (Affine + TPS).
    :param i: Index of step.
    :param model: The model.
    :param criterion: Dict of loss criterion.
    :param args: Additional arguments
    :param mem: A memory dict that store data for visualization during training.
    :return: Contrastive loss, acc@1, acc@5, memory dict
    """
    raw1, raw2 = data

    if i % 2 == 0:
        cond_src = 0
        cond_dst = 0
        raw2 = raw1.clone()

    else:
        cond_src = 1
        cond_dst = 1
        raw1 = raw2.clone()

    if args.gpu is not None:
        raw1 = raw1.cuda(args.gpu, non_blocking=True)
        raw2 = raw2.cuda(args.gpu, non_blocking=True)

    syn_flow_src = torch.cat([mask_outlier(syn_flow_gen()) for _ in range(len(raw1))], dim=0)
    syn_flow_dst = torch.cat([mask_outlier(syn_flow_gen()) for _ in range(len(raw1))], dim=0)

    # compute output
    if i % 2 == 0:
        images_src = raw1
        images_dst = raw2

    else:
        images_src = raw2
        images_dst = raw1

    images_src = color_aug(images_src)
    images_dst = color_aug(images_dst)

    with torch.cuda.amp.autocast():
        images_src = F.grid_sample(images_src.half(),
                                   syn_flow_src.half(),
                                   mode="bilinear", padding_mode="border", align_corners=True)
        images_dst = F.grid_sample(images_dst.half(),
                                   syn_flow_dst.half(),
                                   mode="bilinear", padding_mode="border", align_corners=True)

        output, target, res_src, res_dst = model.module.forward_framework(im_q=images_src,
                                                                          im_k=images_dst,
                                                                          cond_q=cond_src,
                                                                          cond_k=cond_dst)

        clr_loss = criterion["ce"](output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if len(mem["image1"]) < 10:
            mem["image1"].append(prep_img_tensor(images_src))
            mem["image2"].append(prep_img_tensor(images_dst))
            mem["warp_image12"].append(prep_img_tensor(images_src))
            mem["warp_image21"].append(prep_img_tensor(images_dst))
            mem["res2_1"].append(prep_feat_tensor(res_src["layer2"]))
            mem["res2_2"].append(prep_feat_tensor(res_dst["layer2"]))
            mem["res3_1"].append(prep_feat_tensor(res_src["layer3"]))
            mem["res3_2"].append(prep_feat_tensor(res_dst["layer3"]))
            mem["weight3_1"].append(None)
            mem["weight3_2"].append(None)
            mem["dist"].append(None)

        return clr_loss, acc1, acc5, mem


def train(train_loader, model, criterion, optimizer, scaler, epoch, args, writer):
    """
    Train the feature encoder or the warp estimator for one epoch.
    :param train_loader: The dataloader of train dataset.
    :param model: The model.
    :param criterion: Dict of loss criterion.
    :param optimizer: The optimizer.
    :param scaler: The scaler for mixed precision training.
    :param epoch: Current epoch.
    :param args: Additional arguments.
    :param writer: Tensorboard writer.
    :return: A memory dict that store data for visualization during training.
    """

    mem = defaultdict(list)

    # init meters
    clr_losses = AverageMeter('CLRLoss', ':.4e')
    sim_losses = AverageMeter('CROSSLoss', ':.4e')
    syn_losses = AverageMeter('SUPLoss', ':.4e')
    con_losses = AverageMeter('CONLoss', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [clr_losses, sim_losses, syn_losses, con_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # generate an identity displacement field, which is equivalent to a position map
    pos_map = F.affine_grid(torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0),
                            [1, 1, args.stn_size, args.stn_size], align_corners=True).cuda(args.gpu).repeat(
        args.batch_size, 1, 1, 1)

    # init color augmentation
    color_aug = nn.Sequential(
        aug.ColorJitter(0.4, 0.4, 0.4, 0.1),
        aug.RandomGrayscale(p=0.2),
        aug.RandomGaussianBlur((13, 13), (0.1, 2.0), p=0.5),
        aug.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )

    # init synthetic flow generator of Affine and TPS transformations
    syn_flow_gen = SynthecticAffHomoTPSTransfo(random_t=1 / 2, random_alpha=np.pi / 8, random_t_tps=1 / 4,
                                            tps_grid_size=4,
                                            random_s=(0.5, 1.5), tps_reg_factor=0.0, random_t_tps_for_afftps=1 / 4,
                                            size_output_flow=(256, 256), flip=False,
                                            transformation_types=args.trans_type, use_cuda=True)

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):
        n_iters = epoch * len(train_loader) + i
        # train feature encoder
        if args.task == "encoder" or args.task == "both":
            # instance supervision
            if args.supervision == "instance":
                clr_loss, acc1, acc5, mem = step_instance(data, color_aug, syn_flow_gen,
                                                          i, model, criterion, args, mem)
            # pair supervision
            else:
                clr_loss, acc1, acc5, mem = step_pair(data, color_aug, syn_flow_gen,
                                                      i, model, criterion, args, mem)

            encoder_loss = clr_loss * args.clr_loss_weight

            clr_losses.update(clr_loss.item(), data[0].size(0))
            top1.update(acc1[0], data[0].size(0))
            top5.update(acc5[0], data[0].size(0))

            if writer is not None:
                n_iters = epoch * len(train_loader) + i
                clr_losses.log(writer, n_iters)

                top1.log(writer, n_iters)
                top5.log(writer, n_iters)

            # backward pass
            scaler.scale(encoder_loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

        ####################################################
        # train warp estimator
        if args.task == "estimator" or args.task == "both":
            # freeze feature encoder
            if args.freeze:
                for param in model.module.framework.parameters():
                    param.requires_grad = False

            raw1, raw2 = data

            # generate synthetic flows
            syn_gt_flow_src = torch.cat([mask_outlier(syn_flow_gen()) for _ in range(len(raw1))], dim=0)
            syn_gt_flow_dst = torch.cat([mask_outlier(syn_flow_gen()) for _ in range(len(raw1))], dim=0)

            # switch combination of source/destination ~ photo/sketch
            if i % 2 == 0:
                cond_src = 0  # photo
                cond_dst = 1  # sketch
                img_src_aug = raw1  # source
                img_dst_aug = raw2  # destination
                img_src_raw = raw1.clone()  # source w/o spatial aug

            else:
                cond_src = 1
                cond_dst = 0
                img_src_aug = raw2
                img_dst_aug = raw1
                img_src_raw = raw2.clone()

            if args.gpu is not None:
                img_src_aug = img_src_aug.cuda(args.gpu, non_blocking=True)
                img_dst_aug = img_dst_aug.cuda(args.gpu, non_blocking=True)
                img_src_raw = img_src_raw.cuda(args.gpu, non_blocking=True)

            img_src_aug = color_aug(img_src_aug)
            img_dst_aug = color_aug(img_dst_aug)
            img_src_raw = color_aug(img_src_raw)

            with torch.cuda.amp.autocast():
                img_src_aug = F.grid_sample(img_src_aug.half(),
                                            syn_gt_flow_src.half(),
                                            mode="bilinear", padding_mode="border", align_corners=True)
                img_dst_aug = F.grid_sample(img_dst_aug.half(),
                                            syn_gt_flow_dst.half(),
                                            mode="bilinear", padding_mode="border", align_corners=True)

                _, feat_dst_aug = model.module.forward_backbone(img_dst_aug, cond_dst, corr_only=True)
                _, feat_src_aug = model.module.forward_backbone(img_src_aug, cond_src, corr_only=True)
                with torch.no_grad():
                    _, feat_src_raw = model.module.forward_backbone(img_src_raw, cond_src, corr_only=True)

                # compute cross-modal warp: src_raw -> dst_aug
                cross_fwd_flow, cross_bwd_flow = model.module.forward_stn(detach(feat_src_raw), detach(feat_dst_aug))

                # for debugging: log the difference (delta) between predicted flow and identity flow
                delta_cross = (criterion["mask_mse"](cross_fwd_flow.detach(), pos_map) +
                               criterion["mask_mse"](cross_bwd_flow.detach(), pos_map)) / 2
                if writer is not None:
                    writer.add_scalar('metric/Delta_cross', delta_cross.item(), n_iters)

                # render warped cross modal image
                cross_fwd_flow_large = F.interpolate(cross_fwd_flow.permute(0, 3, 1, 2),
                                                     (data[0].shape[-1], data[0].shape[-1]),
                                                     mode="bilinear",
                                                     align_corners=True).permute(0, 2, 3, 1)
                cross_bwd_flow_large = F.interpolate(cross_bwd_flow.permute(0, 3, 1, 2),
                                                     (data[0].shape[-1], data[0].shape[-1]),
                                                     mode="bilinear",
                                                     align_corners=True).permute(0, 2, 3, 1)

                warped_img_src = F.grid_sample(img_src_raw, cross_fwd_flow_large, mode="bilinear",
                                               padding_mode="border",
                                               align_corners=True)
                warped_img_dst = F.grid_sample(img_dst_aug, cross_bwd_flow_large, mode="bilinear",
                                               padding_mode="border",
                                               align_corners=True)

                ####################################################
                # compute similarity loss
                sim_loss = 0
                if args.sim_loss_weight != 0:
                    if args.perceptual:
                        _, warped_feat_src = model.module.forward_backbone(warped_img_src, cond_src, corr_only=True)
                        _, warped_feat_dst = model.module.forward_backbone(warped_img_dst, cond_dst, corr_only=True)
                    else:
                        warped_feat_src = model.module.stn.grid_sample(feat_src_raw, cross_fwd_flow)
                        warped_feat_dst = model.module.stn.grid_sample(feat_dst_aug, cross_bwd_flow)

                    sim_1, weight_1 = model.module.compute_similarity(warped_feat_src,
                                                                      detach(feat_dst_aug))
                    sim_2, weight_2 = model.module.compute_similarity(warped_feat_dst,
                                                                      detach(feat_src_raw))

                    # compute weighted similarity loss
                    if args.weighted:
                        for j in range(len(sim_1)):
                            corr_target = torch.arange(sim_1[j].shape[1]).view(1, -1).repeat(sim_1[j].shape[0], 1).cuda(
                                args.gpu)
                            curr_sim_loss = torch.mean(
                                criterion["ce_none"](sim_1[j] / args.corr_t, corr_target) * weight_1[j] +
                                criterion["ce_none"](sim_2[j] / args.corr_t, corr_target) * weight_2[j])
                            sim_loss += curr_sim_loss
                            if writer is not None:
                                writer.add_scalar('metric/sim_%i' % j, curr_sim_loss.item(), n_iters)
                    else:
                        for j in range(len(sim_1)):
                            corr_target = torch.arange(sim_1[j].shape[1]).view(1, -1).repeat(sim_1[j].shape[0], 1).cuda(
                                args.gpu)
                            sim_loss += torch.mean(
                                criterion["ce_none"](sim_1[j] / args.corr_t, corr_target) +
                                criterion["ce_none"](sim_2[j] / args.corr_t, corr_target))
                    sim_loss = sim_loss / len(sim_1)
                    sim_losses.update(sim_loss.item(), img_src_raw.size(0))
                    if writer is not None:
                        sim_losses.log(writer, n_iters)

                ####################################################
                # compute synthetic flow loss
                # WE NO LONGER USE THIS LOSS

                syn_loss = 0
                if args.syn_loss_weight != 0:
                    syn_pred_flow16, syn_pred_flow8, syn_pred_flow4 = model.module.stn(detach(feat_src_raw), feat_src_aug, training=True)
                    syn_gt_flow_src = F.interpolate(syn_gt_flow_src.permute(0, 3, 1, 2), (args.stn_size, args.stn_size),
                                                    mode="bilinear",
                                                    align_corners=True).permute(0, 2, 3, 1)

                    syn_loss4 = criterion["mask_mse"](syn_pred_flow4, syn_gt_flow_src)
                    syn_loss8 = criterion["mask_mse"](syn_pred_flow8, syn_gt_flow_src)
                    syn_loss16 = criterion["mask_mse"](syn_pred_flow16, syn_gt_flow_src)

                    syn_loss = syn_loss4 * 1.0 + syn_loss8 * 0.5 + syn_loss16 * 0.25

                    delta_sup = criterion["mask_mse"](syn_pred_flow16.detach(), pos_map)

                    syn_losses.update(syn_loss.item(), img_src_aug.size(0))
                    if writer is not None:
                        syn_losses.log(writer, n_iters)
                        writer.add_scalar('metric/Delta_syn', delta_sup.item(), n_iters)
                        writer.add_scalar('metric/4x', syn_loss4.item(), n_iters)
                        writer.add_scalar('metric/8x', syn_loss8.item(), n_iters)
                        writer.add_scalar('metric/16x', syn_loss16.item(), n_iters)

                ####################################################
                # compute consistency loss
                con_loss = 0
                if args.con_loss_weight != 0:
                    cross_cycle_map = F.grid_sample(cross_fwd_flow.permute(0, 3, 1, 2), cross_bwd_flow, mode="nearest",
                                                    padding_mode="zeros", align_corners=True).permute(0, 2, 3, 1)
                    con_loss = criterion["mask_mse"](cross_cycle_map, pos_map)

                    if args.syn_loss_weight != 0:
                        syn_bwd_flow = model.module.stn(feat_src_aug, detach(feat_src_raw))
                        syn_cycle_map = F.grid_sample(syn_pred_flow16.permute(0, 3, 1, 2), syn_bwd_flow, mode="nearest",
                                                      padding_mode="zeros", align_corners=True).permute(0, 2, 3, 1)
                        con_loss = con_loss + criterion["mask_mse"](syn_cycle_map, pos_map)

                    con_losses.update(con_loss.item(), img_src_raw.size(0))
                    if writer is not None:
                        con_losses.log(writer, n_iters)

                estimator_loss = sim_loss * args.sim_loss_weight + \
                                 syn_loss * args.syn_loss_weight + \
                                 con_loss * args.con_loss_weight

            # backward pass
            scaler.scale(estimator_loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

            if args.freeze:
                for param in model.module.framework.parameters():
                    param.requires_grad = True

            # log data to memory dict for visualization during training
            if len(mem["image1"]) < 20:
                mem["image1"].append(prep_img_tensor(img_src_raw))
                mem["image2"].append(prep_img_tensor(img_dst_aug))
                mem["warp_image12"].append(prep_img_tensor(warped_img_src))
                mem["warp_image21"].append(prep_img_tensor(warped_img_dst))
                mem["res2_1"].append(prep_feat_tensor(feat_src_raw["layer2"]))
                mem["res2_2"].append(prep_feat_tensor(feat_dst_aug["layer2"]))
                mem["res3_1"].append(prep_feat_tensor(feat_src_raw["layer3"]))
                mem["res3_2"].append(prep_feat_tensor(feat_dst_aug["layer3"]))

                # visualize weight map
                if args.sim_loss_weight != 0:
                    mem["weight3_1"].append(weight_1[-1][0].view(args.stn_size, args.stn_size).detach().cpu().float())
                    mem["weight3_2"].append(weight_2[-1][0].view(args.stn_size, args.stn_size).detach().cpu().float())
                else:
                    mem["weight3_1"].append(None)
                    mem["weight3_2"].append(None)
                mem["dist"].append(None)

        if i % args.print_freq == 0:
            progress.display(i)

    return mem


def eval_knn(model, valid_loader, epoch, mem, args, writer, plot=False):
    """
    1) Compute the KNN retrieval accuracy (retrieve photo from sketch, and sketch from photo).
       Used for measuring the convergence of feature encoder contrastive learning.
    2) Visualize examples of warping.
    :param model: The model.
    :param valid_loader: Dataloader of validation dataset.
    :param epoch: Current epoch.
    :param mem: Memory dict for visualization.
    :param args: Additional arguments.
    :param writer: Tensorboard Writer.
    :param plot: (bool) Plot the figure or not.
    """
    knn = NearestNeighbors(n_jobs=16)

    with torch.no_grad():
        features1 = []
        features2 = []

        for i, (images1, images2) in tqdm(enumerate(valid_loader), total=len(valid_loader)):

            images1 = images1.cuda(args.gpu, non_blocking=True)
            images2 = images2.cuda(args.gpu, non_blocking=True)

            # collect image features
            fc1, res1 = model.forward_backbone(images1, cond=0)
            fc2, res2 = model.forward_backbone(images2, cond=1)

            features1.append(fc1.detach().cpu().numpy())
            features2.append(fc2.detach().cpu().numpy())

            # collect examples of warping
            fwd_flow, bwd_flow, dist = model.forward_stn(res1, res2, dense_mtx=True)
            fwd_flow = F.interpolate(fwd_flow.permute(0, 3, 1, 2), (256, 256), mode="bilinear",
                                     align_corners=True).permute(0, 2, 3, 1)
            bwd_flow = F.interpolate(bwd_flow.permute(0, 3, 1, 2), (256, 256), mode="bilinear",
                                     align_corners=True).permute(0, 2, 3, 1)

            # visualize examples of warping
            if len(mem["image1"]) < 30 and plot:
                mem["image1"].append(prep_img_tensor(images1))
                mem["image2"].append(prep_img_tensor(images2))
                warped_images12 = F.grid_sample(images1, fwd_flow, mode="bilinear", padding_mode="zeros",
                                                align_corners=True)
                warped_images21 = F.grid_sample(images2, bwd_flow, mode="bilinear", padding_mode="zeros",
                                                align_corners=True)
                mem["warp_image12"].append(prep_img_tensor(warped_images12))
                mem["warp_image21"].append(prep_img_tensor(warped_images21))
                mem["res2_1"].append(prep_feat_tensor(res1["layer2"]))
                mem["res2_2"].append(prep_feat_tensor(res2["layer2"]))
                mem["res3_1"].append(prep_feat_tensor(res1["layer3"]))
                mem["res3_2"].append(prep_feat_tensor(res2["layer3"]))
                mem["weight3_1"].append(None)
                mem["weight3_2"].append(None)
                mem["dist"].append(dist[0].detach().cpu().numpy().view())

    # compute similarity between image features
    features1 = np.concatenate(features1)
    features2 = np.concatenate(features2)

    dist_mtx = 1 - pairwise_distances(features1, features2, metric='cosine')
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(dist_mtx)

    # visualize pairwise similarity matrix (clearer diagonal suggests better retrieval)
    writer.add_figure('pairwise', fig, epoch)
    writer.add_histogram('pairwise/positives', dist_mtx.diagonal(), epoch)
    writer.add_histogram('pairwise/negatives', dist_mtx[~np.eye(dist_mtx.shape[0], dtype=bool)].flatten(), epoch)

    # compute KNN accuracy
    knn = knn.fit(features2)
    _, indices = knn.kneighbors(features1, 1)
    acc = np.sum(indices.flatten() == np.arange(len(features1))) / len(features1)
    print("========== Epoch %d sketch2photo knn validation accuracy: %.6f ==========" % (epoch, acc))
    writer.add_scalar('eval/S>P_KNN', acc, epoch)

    knn = knn.fit(features1)
    _, indices = knn.kneighbors(features2, 1)
    acc = np.sum(indices.flatten() == np.arange(len(features2))) / len(features2)
    print("========== Epoch %d photo2sketch knn validation accuracy: %.6f ==========" % (epoch, acc))
    writer.add_scalar('eval/P>S_KNN', acc, epoch)

    if plot:
        writer.add_figure('visualization', gen_graph(mem), epoch)


def eval_pck(model, test_loader, epoch, args, writer, image_size=256):
    """
    Compute the PCK error metric.
    :param model: The model.
    :param test_loader: The dataloader of test dataset.
    :param epoch: Current epoch.
    :param args: Additional arguments.
    :param writer: Tensorboard Writer.
    :param image_size: Size of image. Default: 256.
    """
    with torch.no_grad():
        pck05_list = []
        pck10_list = []

        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            photo, sketch, photo_kps, sketch_kps = data
            photo = photo.cuda(args.gpu, non_blocking=True)
            sketch = sketch.cuda(args.gpu, non_blocking=True)

            # get feature maps
            _, photo_res = model.encoder_q(photo, cond=0, return_map=True)
            _, sketch_res = model.encoder_q(sketch, cond=1, return_map=True)

            # estimate displacement field
            fwd_flow, bwd_flow = model.forward_stn(photo_res, sketch_res)
            fwd_flow = F.interpolate(fwd_flow.permute(0, 3, 1, 2), (image_size, image_size), mode="bilinear",
                                align_corners=True).permute(0, 2, 3, 1).cpu()
            bwd_flow = F.interpolate(bwd_flow.permute(0, 3, 1, 2), (image_size, image_size), mode="bilinear",
                                align_corners=True).permute(0, 2, 3, 1).cpu()

            # project keypoints & compute error
            pred_sketch_kps = proj_kps(bwd_flow, photo_kps, image_size)
            pck10, pck05 = compute_pck(sketch_kps, pred_sketch_kps, image_size)
            pck10_list.append(pck10)
            pck05_list.append(pck05)

            pred_photo_kps = proj_kps(fwd_flow, sketch_kps, image_size)
            pck10, pck05 = compute_pck(photo_kps, pred_photo_kps, image_size)
            pck10_list.append(pck10)
            pck05_list.append(pck05)

        pck10_list = np.concatenate(pck10_list, axis=0)
        pck05_list = np.concatenate(pck05_list, axis=0)

        print("========== Epoch %d pck@0.1: %.6f ==========" % (epoch, np.mean(pck10_list)))
        writer.add_scalar('eval/pck@0.1', np.mean(pck10_list), epoch)
        writer.add_histogram('pck@0.1', np.array(pck10_list), epoch)

        print("========== Epoch %d pck@0.05: %.6f ==========" % (epoch, np.mean(pck05_list)))
        writer.add_scalar('eval/pck@0.05', np.mean(pck05_list), epoch)
        writer.add_histogram('pck@0.05', np.array(pck05_list), epoch)

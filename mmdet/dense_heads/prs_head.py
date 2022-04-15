import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS, build_loss
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead
from mmcv.ops import DeformConv2d
from mmcv.runner import BaseModule

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

class AdaptiveConv(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=3,
                 groups=1,
                 deform_groups=1,
                 bias=False,
                 type='offset',
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv'))):
        super(AdaptiveConv, self).__init__(init_cfg)
        assert type in ['offset', 'dilation']
        self.adapt_type = type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                deform_groups=deform_groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def forward(self, x, offset, num_anchors):
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.reshape(N, H, W, -1)
            offset = offset.permute(0, 3, 1, 2)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x

# TODO: add loss evaluator for SSD
@HEADS.register_module()
class PRSHead(AnchorHead):

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 background_label=None,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_convs_refine',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(AnchorHead, self).__init__(init_cfg, **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        num_anchors = self.anchor_generator.num_base_anchors
        self.num_anchors = num_anchors
        self.anchor_strides = anchor_generator['strides']
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i],
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        # dcn
        dcn = []
        reg_convs_refine = []
        cls_convs_refine = []
        for i in range(len(in_channels)):
            dcn.append(AdaptiveConv(num_anchors[i]*in_channels[i], in_channels[i],deform_groups=self.num_anchors[i]))
            reg_convs_refine.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs_refine.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * self.cls_out_channels,
                    kernel_size=3,
                    padding=1))
        self.dcn = nn.ModuleList(dcn)
        self.reg_convs_refine = nn.ModuleList(reg_convs_refine)
        self.cls_convs_refine = nn.ModuleList(cls_convs_refine)
        self.relu = nn.ReLU(inplace=True)
        # self.BCE = build_loss(loss_cls_pre)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        fg_scores, bbox_preds = self(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in fg_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = fg_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        if gt_labels is None:
            loss_inputs = (anchor_list, valid_flag_list, fg_scores, bbox_preds, gt_bboxes, img_metas)
        else:
            loss_inputs = (anchor_list, valid_flag_list, fg_scores, bbox_preds, gt_bboxes, gt_labels, img_metas)
        losses = self.loss_pre(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        new_feats = []

        # pre-processing
        for i in range(len(fg_scores)):
            score = fg_scores[i]
            s,_ = torch.max(score, dim = 1)
            s = s.unsqueeze(1)
            s = F.sigmoid(s)
            new_feats.append(s*x[i]+x[i])
        # new_feats = x

        anchor_list_refine = self.refine_bboxes(anchor_list, bbox_preds, img_metas)
        offset_list = self.anchor_offset(anchor_list_refine, self.anchor_strides, featmap_sizes)

        cls_scores, bbox_preds_refine = self.forward_post(new_feats, offset_list)
        if gt_labels is None:
            loss_inputs = (anchor_list_refine, valid_flag_list, cls_scores, bbox_preds_refine, gt_bboxes, img_metas)
        else:
            loss_inputs = (anchor_list_refine, valid_flag_list, cls_scores, bbox_preds_refine, gt_bboxes, gt_labels, img_metas)
        losses_post = self.loss_post(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        losses.update(losses_post)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(anchor_list_refine, cls_scores, bbox_preds_refine, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []

        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

        return cls_scores, bbox_preds

    def forward_post(self, feats, offset_list):
        cls_scores = []
        bbox_preds = []
        for i in range(len(feats)):
            x = feats[i]
            shape = list(x.shape)
            x = x.unsqueeze(1).expand((shape[0],self.num_anchors[i],shape[1],shape[2],shape[3]))
            shape[1] = shape[1] * self.num_anchors[i]
            x = x.reshape(shape)
            offset = offset_list[i]
            feat = self.relu(self.dcn[i](x, offset, self.num_anchors[i]))
            cls_score = self.cls_convs_refine[i](feat)
            bbox_pred = self.reg_convs_refine[i](feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def loss_single_post(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights*0.5,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        return loss_cls[None], loss_bbox

    def loss_single_pre(self, fg_scores, bbox_pred, anchor, fg_labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        fg_loss = (F.binary_cross_entropy(
            F.sigmoid(fg_scores), fg_labels, reduction='none') * label_weights).mean()


        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights*0.5,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)


        return fg_loss, loss_bbox

    def loss_pre(self,
             anchor_list,
             valid_flag_list,
             fg_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in fg_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        # device = fg_scores[0].device

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        fg_labels = all_labels.clone().float()
        fg_labels[fg_labels != self.num_classes] = 1
        fg_labels[fg_labels == self.num_classes] = 0

        all_fg_scores = torch.cat([
            f.permute(0, 2, 3, 1).reshape(
                num_images, -1) for f in fg_scores
        ], 1)


        fg_losses, losses_bbox = multi_apply(
            self.loss_single_pre,
            all_fg_scores,
            all_bbox_preds,
            all_anchors,
            fg_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(losses_fg=fg_losses, loss_bbox=losses_bbox)

    def loss_post(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        # device = cls_scores[0].device

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        fg_labels = all_labels.clone().float()
        fg_labels[fg_labels != self.num_classes] = 1
        fg_labels[fg_labels == self.num_classes] = 0


        losses_cls, losses_bbox_ref = multi_apply(
            self.loss_single_post,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, losses_bbox_ref=losses_bbox_ref)

    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        def _shape_offset(anchors, stride, ks=3, dilation=1):
            # currently support kernel_size=3 and dilation=1
            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size, num_anchors):
            feat_h, feat_w = featmap_size

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            xx = xx.unsqueeze(1).expand(xx.shape+(num_anchors,)).reshape(-1)
            yy = yy.unsqueeze(1).expand(yy.shape+(num_anchors,)).reshape(-1)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl],
                                                     self.num_anchors[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        """Refine bboxes through stages."""
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        fg_scores, bbox_preds = self(feats)

        featmap_sizes = [featmap.size()[-2:] for featmap in fg_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = fg_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        new_feats = []

        # pre-processing
        # for i in range(len(cls_scores)):
        #     score = fg_scores[i]
        #     s,_ = torch.max(score, dim = 1)
        #     s = s.unsqueeze(1)
        #     s = F.sigmoid(s)
        #     new_feats.append(s*feats[i]+feats[i])
        new_feats = feats

        anchor_list_refine = self.refine_bboxes(anchor_list, bbox_preds, img_metas)
        offset_list = self.anchor_offset(anchor_list_refine, self.anchor_strides, featmap_sizes)
        cls_scores, bbox_preds_refine = self.forward_post(new_feats, offset_list)
        results_list = self.get_bboxes(anchor_list_refine[0], cls_scores, bbox_preds_refine, img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   anchor_list,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        # device = cls_scores[0].device
        # featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # mlvl_anchors2 = self.anchor_generator.grid_anchors(
        #     featmap_sizes, device=device)
        # anchor_list = images_to_levels(anchor_list, num_levels)
        mlvl_anchors = [anchor_list[i].detach() for i in range(num_levels)]
        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list
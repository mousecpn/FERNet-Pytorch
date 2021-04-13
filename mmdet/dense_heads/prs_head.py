import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import cv2
from torchvision.utils import save_image
import numpy as np
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead
from mmcv.ops import DeformConv2d
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)

# TODO: add loss evaluator for SSD
@HEADS.register_module()
class PRSHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

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
                 test_cfg=None):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        num_anchors = self.anchor_generator.num_base_anchors
        self.num_anchors = num_anchors

        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4 ,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i]* (num_classes + 1),
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
        dcn_reg_convs = []
        dcn_cls_convs = []
        for i in range(len(in_channels)):
            dcn_reg_convs.append(
                DeformConv2d(
                    in_channels[i]*num_anchors[i],
                    4*num_anchors[i],
                    kernel_size=3,
                    bias=True,
                    padding=3,
                    groups= num_anchors[i],
                    deform_groups= num_anchors[i],
                    dilation = 3
                ))
            dcn_cls_convs.append(
                DeformConv2d(
                    in_channels[i]*num_anchors[i],
                    (num_classes + 1)*num_anchors[i],
                    kernel_size=3,
                    bias=True,
                    padding=3,
                    groups= num_anchors[i],
                    deform_groups= num_anchors[i],
                    dilation=3
                ))
        self.dcn_reg_convs = nn.ModuleList(dcn_reg_convs)
        self.dcn_cls_convs = nn.ModuleList(dcn_cls_convs)
        self.BCE = nn.BCEWithLogitsLoss()
        self.count = 0.0
        self.total_count = 3230.0*24.0
        self.offset_transfrom = nn.Linear(2,2)

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
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def cam(self, feature, kernel_size = 1, path='cam.jpg'):
        weights = torch.mean(feature, dim=(1, 2))
        cam = torch.matmul(feature.permute(1, 2, 0), weights)
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        cam = 1 - cam

        if kernel_size > 1:
            cam = cam.reshape((1, 1) + cam.shape)
            cam = F.max_pool2d(cam, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            cam = cam.squeeze()
        heatmap = cv2.applyColorMap(np.uint8(255 * cam.detach().cpu().numpy()), cv2.COLORMAP_JET)
        cv2.imwrite(path, heatmap)
        return cam

    def forward(self, feats, img_metas):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        new_feats = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

        # refinement
        for i in range(len(cls_scores)):
            score = cls_scores[i]
            batch_size = score.shape[0]
            if self.background_label == self.num_classes:
                score = score.reshape(batch_size,-1, self.num_classes+1,score.shape[-2],score.shape[-1])
                score = torch.softmax(score,dim=2)
                score = torch.sum(score[:,:, :-1, :, :],dim=2,keepdim=True)
                score = score.reshape(batch_size, -1, score.shape[-2],score.shape[-1])
            else:
                score = score.reshape(batch_size, -1, self.num_classes + 1, score.shape[-2], score.shape[-1])
                score = torch.softmax(score, dim=2)
                score = torch.sum(score[:,:, 1:, :, :],dim=2,keepdim=True)
                score = score.reshape(batch_size, -1, score.shape[-2], score.shape[-1])
            # score  = score[:-1]
            s,_ = torch.max(score, dim = 1)

            ########################## test #######################################
            # ori = cv2.imread(img_metas[0]['filename'])
            # ori = torch.tensor(ori).permute(2,0,1)
            # im = F.interpolate(ori.unsqueeze(0)/255.0, size=(300, 300), mode="nearest").squeeze(0)
            # save_image(im, 'sample.jpg')
            # term = s[0].unsqueeze(0)
            # term = F.interpolate(term.unsqueeze(0), size=(300,300), mode="nearest").squeeze(0)
            # self.cam(feats[i][0])
            # save_image(term,'out.jpg')
            ########################## test #######################################


            s = s.unsqueeze(1)
            # s = F.sigmoid(s)
            new_feats.append(s*feats[i] + feats[i])

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        cls_scores_new = []
        bbox_preds_new = []
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        img_w = img_metas[0]['img_shape'][1]
        img_h = img_metas[0]['img_shape'][0]
        self.count += len(feats)
        for i in range(len(feats)):
            featmap_size = featmap_sizes[i]
            anchors = [a[i] for a in anchor_list]
            anchors = torch.stack(anchors)
            bbox_pred = bbox_preds[i]
            w = anchors[:, :, 2] - anchors[:, :, 0]
            h = anchors[:, :, 3] - anchors[:, :, 1]

            w = w / img_w * featmap_size[0]
            h = h / img_h * featmap_size[1]
            w = w.view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 1)
            h = h.view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 1)
            dxy = bbox_pred.permute(0, 2, 3, 1)
            dxy = bbox_pred.view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 4)[:, :, :, :, :2]
            off_x = dxy[:, :, :, :, 0].view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 1) * w * 0.1
            off_y = dxy[:, :, :, :, 1].view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 1) * h * 0.1
            offset = torch.cat((off_y, off_x), dim = -1)

            ################### test ##################################################
            # cx = (anchors[:, :, 2] + anchors[:, :, 0])/2
            # cy = (anchors[:, :, 3] + anchors[:, :, 1])/2
            # cx = cx.view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 1)
            # cy = cy.view(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 1)
            # ori_cv = ori.permute(1,2,0).numpy().copy()
            # ori_cv = cv2.resize(ori_cv, (300, 300))
            # s = cls_scores[i].reshape(bbox_pred.shape[0], self.num_anchors[i], self.num_classes+1, featmap_size[0], featmap_size[1])
            # s_max,_ = torch.max(s,dim=2)
            # for f in range(featmap_size[0]):
            #     for k in range(featmap_size[1]):
            #         # for a in range(self.num_anchors[i]):
            #             # if s[0,a,-1,f,k] == s_max[0,a,f,k]:
            #             #     www = s[0,a,-1,f,k]
            #             #     continue
            #         a = 0
            #         if k % 4 != 0 or f % 4 != 0:
            #             continue
            #         y = int( (f+1) / featmap_size[0] * ori_cv.shape[0])
            #         x = int( (k+1) / featmap_size[1] * ori_cv.shape[1])
            #         ox = int( off_x[0,f,k,a,0])
            #         oy = int( off_y[0,f,k,a,0])
            #         color = np.random.random(3)*255
            #         color = (int(color[0]),int(color[1]),int(color[2]))
            #         cv2.circle(ori_cv, (int(x), int(y)), 1,color , 4)
            #         ax = cx[0,f,k,a,0]
            #         ay = cy[0,f,k,a,0]
            #         wt = w[0,f,k,a,0]
            #         ht = h[0,f,k,a,0]
            #         box = [x+ox,y+oy,wt,ht]
            #         box = xywh2xyxy(box)
            #         cv2.rectangle(ori_cv, (box[0], box[1]), (box[2], box[3]), color, 2)
            #         # cv2.circle(ori_cv, (int(x)+ox, int(y)+oy), 1, color, 4)
            # cv2.imwrite('attention.jpg',ori_cv)

            ################### test ##################################################


            offset = offset.unsqueeze(-1).expand(bbox_pred.shape[0], featmap_size[0], featmap_size[1], self.num_anchors[i], 2, 9).permute(0,1,2,3,5,4)
            offset = self.offset_transfrom(offset.detach())

            offset = offset.reshape(bbox_pred.shape[0], self.num_anchors[i], featmap_size[0], featmap_size[1], 18)
            offset = offset.permute(0,1,4,2,3).reshape(bbox_pred.shape[0], self.num_anchors[i]*18, featmap_size[0], featmap_size[1]).contiguous()
            feat = new_feats[i]
            c = feat.shape[1]
            feat = feat.permute(1,0,2,3).expand(self.num_anchors[i], c, bbox_pred.shape[0], featmap_size[0], featmap_size[1]).reshape(self.num_anchors[i]*c, bbox_pred.shape[0], featmap_size[0], featmap_size[1])
            feat = feat.permute(1,0,2,3).contiguous()
            cls_score_new = self.dcn_cls_convs[i](feat, offset)
            bbox_pred_new = self.dcn_reg_convs[i](feat, offset)
            cls_scores_new.append(cls_score_new)
            bbox_preds_new.append(bbox_pred_new)
        return [cls_scores_new, cls_scores], [bbox_preds_new, bbox_preds]


    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
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
        # loss_cls = loss_cls[None] * self.count / self.total_count
        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        return loss_cls[None], loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        cls_scores_new, cls_scores = cls_scores
        bbox_preds_new, bbox_preds = bbox_preds

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
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
        all_cls_scores_new = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores_new
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        # [stage, bs, anchors*4, h, w] -> [bs, stage*h*w*anchors, 4]
        all_bbox_preds_new = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds_new
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


        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)

        losses_cls2, losses_bbox2 = multi_apply(
            self.loss_single,
            all_cls_scores_new,
            all_bbox_preds_new,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, losses_cls2=losses_cls2, loss_bbox=losses_bbox, losses_bbox2=losses_bbox2)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        # cls_scores_new, cls_scores = cls_scores
        # bbox_preds_new, bbox_preds = bbox_preds
        cls_scores, cls_scores_old = cls_scores
        bbox_preds, bbox_preds_old = bbox_preds
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def xywh2xyxy(box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x1 = int(x - w//2)
    x2 = int(x + w//2)
    y1 = int(y - h//2)
    y2 = int(y + h//2)
    return [x1,y1,x2,y2]

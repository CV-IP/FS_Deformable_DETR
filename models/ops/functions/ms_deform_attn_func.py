# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        # 将注意力权重与value进行weighted sum ，调用cuda实现版本
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # 多尺度可变形注意力，根据采样点位置在多尺度value中插值采样出对应的特征图，最后和注意力权重进行weighted sum得到输出
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape  # batchsize， key个数， head个数， 维度
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape    #Lq_: query个数， L_:level数， P_: 采样点个数 
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)   # 区分每个level的key
    sampling_grids = 2 * sampling_locations - 1      # 因为需要使用grid_sample因此需要将采样点映射到-1，1之间
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # 最后就是将注意力权重和采样特征进行weighted sum：
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

    '''
    可以发现其本质是利用F.grid_sample函数进行采样，该函数使用时需要将采样点归一化到之间。
    输入value对应着keys， value_spatial_shapes用于对value进行拆分，拆分成不同的level，在不同的level中进行采样，每个level采样n_head*n_point个向量。
    这里想到之前《纯pytorch版本的deformable cnn的实现》进行采样的过程，其实循环部分也可以借鉴这里，直接将采样点并在一起进行采样。
    attention_weights分别加权每一个D维的向量，总共相当于每个query的L_*P_个特征进行加权求和。
    最终的输出是N * Lq_ * d_model, 其中d_model是总的特征的维度，Lq_是query的个数。
    '''

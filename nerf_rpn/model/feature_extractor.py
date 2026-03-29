import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import MLP, Permute
import math
from functools import partial

from .fpn import FPN


# Simplified 3D Residual Block
class ResidualBlockSimplified(nn.Module):
    """The simplified Basic Residual block of ResNet."""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += X
        return F.relu(Y)


# ResNet Bottleneck for 3D convolution
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


# ResNet_FPN
class ResNet_FPN_64(nn.Module):
    """ A smaller backbone for 64^3 inputs. """
    # block: the type of ResNet layer
    # layers: the depth of each size of layers, i.e. the num of layers before the next
    def __init__(self, block, layers, input_dim=4, use_fpn=True):
        super(ResNet_FPN_64, self).__init__()
        self.in_planes = 16
        self.out_channels = 64
        self.conv1 = nn.Conv3d(input_dim, 16, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  16, layers[0], stride=1)
        self.layer2 = self._make_layer(block,  32, layers[1], stride=2)
        self.layer3 = self._make_layer(block,  64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        # Top layer
        self.toplayer = nn.Conv3d(
            512, 64, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv3d(
            256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(
            128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(
            64, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, X, Y, Z = y.size()
        return F.interpolate(x, size=(X, Y, Z), mode='trilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool3d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


class ResNet_FPN_256(nn.Module):
    # block: the type of ResNet layer
    # layers: the depth of each size of layers, i.e. the num of layers before the next

    '''
    Args:
        layers: list of int. Its size could be variable. The length will be the ouput
                length. The value is the depth of layers at that level
        is_max_pool: If it is False, the network will not use downsample

    Returns (of self.forward function):
        A feature list. Its size is equal to the size of self.layers.
    '''

    def __init__(self, block, layers, input_dim=4, is_max_pool=False):
        super(ResNet_FPN_256, self).__init__()
        self.in_planes = 64
        self.out_channels = 256
        self.conv1 = nn.Conv3d(input_dim, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)

        # Bottom-up layers
        self.layers = nn.ModuleList()
        self.start_deep = self.in_planes
        self.is_max_pool = is_max_pool
        for i in range(len(layers)):
            self.layers.append(self._make_layer(block, self.start_deep * (2**i), layers[i],
                                                stride=1 if i == 0 else 2))

        # Smooth layers
        self.smooths = nn.ModuleList()
        for i in range(len(layers)-1):
            self.smooths.append(nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1))

        # Lateral layers
        self.latlayers = nn.ModuleList()
        for i in range(len(layers)-1, -1, -1):
            self.latlayers.append(
                nn.Conv3d(block.expansion * self.start_deep * (2**i), self.out_channels, 
                          kernel_size=1, stride=1, padding=0)
            )

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, X, Y, Z = y.size()
        return F.interpolate(x, size=(X, Y, Z), mode='nearest') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        if self.is_max_pool:
            c1 = F.max_pool3d(c1, kernel_size=3, stride=2, padding=1)
        c_out = [c1]
        for i in range(len(self.layers)):
            c_out.append(self.layers[i](c_out[i]))

        # Top-down
        p5 = self.latlayers[0](c_out[-1])
        p_out = [p5]
        for i in range(len(self.latlayers)-1):
            p_out.append(self._upsample_add(p_out[i], self.latlayers[i+1](c_out[-2-i])))

        # Smooth
        for i in range(len(self.smooths)):
            p_out[i+1] = self.smooths[i](p_out[i+1])

        p_out.reverse()
        return p_out


# Simplified ResNet (for debug)
class ResNetSimplified_64(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.residuals = nn.ModuleList()
        for i in range(num_residuals):
            self.residuals.append(ResidualBlockSimplified(out_channels))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        for i in range(len(self.residuals)):
            Y = self.residuals[i](Y)
        return (Y,)


class ResNetSimplified_256(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.residuals = nn.ModuleList()
        for i in range(num_residuals):
            self.residuals.append(ResidualBlockSimplified(out_channels))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.pool1(Y)
        for i in range(len(self.residuals)):
            Y = self.residuals[i](Y)
        return (Y,)



# ------------------------------------------------------------------
# 3D Second-Order Pooling
# ------------------------------------------------------------------
class SecondOrderPooling3D(nn.Module):
    def __init__(self, pool_dim='d'):
        super(SecondOrderPooling3D, self).__init__()
        assert pool_dim in ['d', 'h', 'w']
        self.pool_dim = pool_dim

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.pool_dim == 'd':
            mean_val = x.mean(dim=(3,4))
            max_val  = x.amax(dim=(3,4))
        elif self.pool_dim == 'h':
            mean_val = x.mean(dim=(2,4))
            max_val  = x.amax(dim=(2,4))
        else:
            mean_val = x.mean(dim=(2,3))
            max_val  = x.amax(dim=(2,3))
        pooled = mean_val + max_val
        return pooled


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat((avg_pool, max_pool), dim=1)
        attn_map = torch.sigmoid(self.conv(concat))
        return x * attn_map


class MultiDimAttentionPooling3d(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super(MultiDimAttentionPooling3d, self).__init__()
        padding = kernel_size // 2
        self.initial_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.sop_d = SecondOrderPooling3D(pool_dim='d')
        self.sop_h = SecondOrderPooling3D(pool_dim='h')
        self.sop_w = SecondOrderPooling3D(pool_dim='w')
        self.conv_d = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.conv_h = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.channel_attn = ChannelAttention3D(in_channels)
        self.down_conv = nn.Conv3d(in_channels * 5, in_channels, kernel_size=1, stride=2, padding=0, bias=True)

    def forward(self, x):
        x_in = self.initial_conv(x)
        d_pooled = self.sop_d(x_in)
        h_pooled = self.sop_h(x_in)
        w_pooled = self.sop_w(x_in)
        d_attn = self.conv_d(d_pooled)
        h_attn = self.conv_h(h_pooled)
        w_attn = self.conv_w(w_pooled)
        d_attn = d_attn.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_in.size(2), x_in.size(3), x_in.size(4))
        h_attn = h_attn.unsqueeze(2).unsqueeze(-1).expand(-1, -1, x_in.size(2), x_in.size(3), x_in.size(4))
        w_attn = w_attn.unsqueeze(2).unsqueeze(2).expand(-1, -1, x_in.size(2), x_in.size(3), x_in.size(4))
        c_attn = self.channel_attn(x_in)
        x_d = x_in * d_attn
        x_h = x_in * h_attn
        x_w = x_in * w_attn
        x_c = c_attn
        combined = torch.cat((x_in, x_d, x_h, x_w, x_c), dim=1)
        out = self.down_conv(combined)
        return out


# ------------------------------------------------------------------
# 3D Squeeze-and-Excitation Block
# ------------------------------------------------------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        assert channels > reduction
        self.channels = channels
        self.reduction = reduction
        self.avg_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.max_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        avg_pool = x.mean(dim=[2, 3, 4])
        max_pool, _ = x.reshape(B, C, -1).max(dim=2)
        avg_out = self.avg_fc(avg_pool)
        max_out = self.max_fc(max_pool)
        scale = self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        return x * scale


# ------------------------------------------------------------------
# Windowed Multi-Head Cross-Attention in 3D
# ------------------------------------------------------------------
class WMHCA3D(nn.Module):
    def __init__(self, in_channels, num_heads=4, window_size=(2, 2, 2),
                 use_pre_attn_se=True, use_post_attn_se=True, se_reduction=16):
        super().__init__()
        assert in_channels % num_heads == 0
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.conv_q = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv_k = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.se_block_pre = SEBlock3D(in_channels, se_reduction) if use_pre_attn_se else None
        self.proj = nn.Linear(in_channels, in_channels)
        self.se_block_post = SEBlock3D(in_channels, se_reduction) if use_post_attn_se else None
        self.fusion_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.norm = nn.LayerNorm(in_channels)
        self.attn_drop = nn.Dropout(p=0.0)
        self.proj_drop = nn.Dropout(p=0.0)

    def forward(self, x_q, x_k):
        B, C, D, H, W = x_q.shape
        d_w, h_w, w_w = self.window_size

        q_enc = self.conv_q(x_q)
        k_enc = self.conv_k(x_k)
        v_enc = k_enc
        skip = q_enc

        pad_d = (math.ceil(D / d_w) * d_w) - D
        pad_h = (math.ceil(H / h_w) * h_w) - H
        pad_w = (math.ceil(W / w_w) * w_w) - W
        if (pad_d or pad_h or pad_w) > 0:
            q_enc = F.pad(q_enc, (0, pad_w, 0, pad_h, 0, pad_d))
            k_enc = F.pad(k_enc, (0, pad_w, 0, pad_h, 0, pad_d))
            v_enc = F.pad(v_enc, (0, pad_w, 0, pad_h, 0, pad_d))
            _, C, D_p, H_p, W_p = q_enc.shape
        else:
            D_p, H_p, W_p = D, H, W

        num_d = D_p // d_w
        num_h = H_p // h_w
        num_w = W_p // w_w
        num_windows = num_d * num_h * num_w
        window_volume = d_w * h_w * w_w

        q_windows = q_enc.view(B, C, num_d, d_w, num_h, h_w, num_w, w_w)
        q_windows = q_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(B * num_windows, window_volume, C)

        k_windows = k_enc.view(B, C, num_d, d_w, num_h, h_w, num_w, w_w)
        k_windows = k_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(B * num_windows, window_volume, C)

        v_windows = v_enc.view(B, C, num_d, d_w, num_h, h_w, num_w, w_w)
        v_windows = v_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(B * num_windows, window_volume, C)

        q_windows = self.norm(q_windows)
        k_windows = self.norm(k_windows)
        v_windows = self.norm(v_windows)

        Q = self.q_proj(q_windows)
        K = self.k_proj(k_windows)
        V = self.v_proj(v_windows)

        def split_heads(x):
            x = x.view(-1, window_volume, self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.se_block_pre is not None:
            Bn = attn_logits.shape[0]
            attn_5d = attn_logits.view(Bn, self.num_heads, 1, window_volume, window_volume)
            attn_5d = self.se_block_pre(attn_5d)
            attn_logits = attn_5d.view(Bn, self.num_heads, window_volume, window_volume)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        x_attn = torch.matmul(attn_weights, V)
        x_attn = x_attn.permute(0, 2, 1, 3).contiguous().view(-1, window_volume, self.in_channels)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        x_attn = x_attn.view(B, num_d, num_h, num_w, d_w, h_w, w_w, C)
        x_attn = x_attn.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x_attn = x_attn.view(B, C, num_d * d_w, num_h * h_w, num_w * w_w)

        if (pad_d or pad_h or pad_w) > 0:
            x_attn = x_attn[:, :, :D_p - pad_d, :H_p - pad_h, :W_p - pad_w]

        if self.se_block_post is not None:
            x_attn = self.se_block_post(x_attn)

        x_out = x_attn + skip
        x_out = self.fusion_conv(x_out)
        return x_out


# ------------------------------------------------------------------
# MACBlock Definition
# ------------------------------------------------------------------
class MACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MACBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.conv1x1(out)
        return out + identity


# ------------------------------------------------------------------
# Dilated Soft Mask Attention (3D)
# ------------------------------------------------------------------
class DilatedSoftMask3D(nn.Module):
    def __init__(self, in_channels, dilations=(1, 2, 4)):
        super().__init__()
        d1, d2, d3 = dilations
        self.trunk_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )
        self.mask_branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=d1, dilation=d1, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=d1, dilation=d1, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )
        self.mask_branch2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=d2, dilation=d2, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=d2, dilation=d2, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )
        self.mask_branch3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=d3, dilation=d3, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=d3, dilation=d3, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )
        self.mask_conv = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels), nn.Sigmoid()
        )

    def forward(self, x):
        t = self.trunk_branch(x)
        m1 = self.mask_branch1(x)
        m2 = self.mask_branch2(x)
        m3 = self.mask_branch3(x)
        concat_out = torch.cat([m1, m2, m3], dim=1)
        fused_mask = self.mask_conv(concat_out)
        return t + fused_mask


# ------------------------------------------------------------------
# ECA + Dilated Spatial Attention (3D)
# ------------------------------------------------------------------
class ECADilatedAttention3D(nn.Module):
    def __init__(self, channel, k_size=3, dilations=(1, 2, 4)):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eca_conv = nn.Conv3d(channel, channel, kernel_size=k_size,
                                  padding=(k_size - 1) // 2, bias=False)
        self.eca_sigmoid = nn.Sigmoid()
        self.dilated_softmask = DilatedSoftMask3D(in_channels=channel, dilations=dilations)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.eca_conv(y)
        y = self.eca_sigmoid(y)
        x_eca = x * y.expand_as(x)
        spatial_attention = self.dilated_softmask(x_eca)
        out = x_eca + (x_eca * spatial_attention)
        return out


# ------------------------------------------------------------------
# VGG Configuration Options
# ------------------------------------------------------------------
vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
          512, 512, 512, 512, "M"],
    "AF": [64, 128, "F", 256, 256, "M", "F", 512, 512, "M", "F", 512, 512, "M", "F"],
    "DF": [64, 64, 128, 128, "F", 256, 256, 256, "M", "F", 512, 512, 512, "M", "F",
           512, 512, 512, "M", "F"],
    "EF": [64, 64, 128, 128, "F", 256, 256, 256, 256, "M", "F", 512, 512, 512, 512, "M",
           "F", 512, 512, 512, 512, "M", "F"]
}


# ------------------------------------------------------------------
# VGG_FPN Backbone with TACA attention modules
# ------------------------------------------------------------------
class VGG_FPN(nn.Module):
    def __init__(self, cfg: str = "D", in_channels: int = 4, batch_norm: bool = True,
                 input_size: int = 256, conv_at_start: bool = False,
                 taca_placement: str = 'all', mvap_kernel_size: int = 5,
                 ddaf_dilations: tuple = (1, 2, 4),
                 vcwaf_window_size: tuple = (2, 2, 2), vcwaf_num_heads: int = 4,
                 macblock_placement: str = 'C5',
                 use_maxpool: bool = False):
        super().__init__()
        self.out_channels = 256
        self.taca_placement = taca_placement
        _in_channels = in_channels if not conv_at_start else 32
        self.layers = self.make_layers(vgg_cfgs[cfg], _in_channels, batch_norm, input_size,
                                       mvap_kernel_size=mvap_kernel_size,
                                       use_maxpool=use_maxpool)
        self.fpn_neck = FPN([128, 256, 512, 512], self.out_channels, 4)

        # Parse TACA placement
        if taca_placement == 'none':
            active = [False, False, False, False]
        elif taca_placement == 'all':
            active = [True, True, True, True]
        elif taca_placement == 'C2':
            active = [True, False, False, False]
        elif taca_placement == 'C3':
            active = [False, True, False, False]
        elif taca_placement == 'C4':
            active = [False, False, True, False]
        elif taca_placement == 'C5':
            active = [False, False, False, True]
        elif taca_placement == 'C4C5':
            active = [False, False, True, True]
        elif taca_placement == 'C2C3C4':
            active = [True, True, True, False]
        else:
            active = [True, True, True, True]

        attn_list = []
        channels = [128, 256, 512, 512]
        for i in range(4):
            if active[i]:
                if i < 3:
                    attn_list.append(ECADilatedAttention3D(channels[i], dilations=ddaf_dilations))
                else:
                    attn_list.append(WMHCA3D(channels[i], num_heads=vcwaf_num_heads,
                                             window_size=vcwaf_window_size))
            else:
                attn_list.append(nn.Identity())
        self.attn_modules = nn.ModuleList(attn_list)

        # Fix SEBlock mismatch in WMHCA3D
        if isinstance(self.attn_modules[3], WMHCA3D):
            wm = self.attn_modules[3]
            wm.se_block_pre = SEBlock3D(wm.num_heads, reduction=1)

        self.conv_at_start = conv_at_start
        self.starting_layers = None
        self.ds_layers = None
        if self.conv_at_start:
            self.starting_layers = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            )
            self.ds_layers = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                nn.Conv3d(32, 128, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            )
        # Replace last Conv3d with MACBlock based on placement config
        if cfg in ["AF", "DF", "EF"] and macblock_placement != 'none':
            if macblock_placement == 'C5':
                mac_indices = [7]  # Only deepest stage
            elif macblock_placement == 'all':
                mac_indices = [4, 5, 6, 7]  # All stages C2-C5
            elif macblock_placement == 'C4C5':
                mac_indices = [6, 7]
            elif macblock_placement == 'C2':
                mac_indices = [4]
            elif macblock_placement == 'C3':
                mac_indices = [5]
            elif macblock_placement == 'C4':
                mac_indices = [6]
            else:
                mac_indices = [7]

            for idx in mac_indices:
                if idx < len(self.layers):
                    block = self.layers[idx]
                    for i in range(len(block) - 1, -1, -1):
                        if isinstance(block[i], nn.Conv3d):
                            conv_layer = block[i]
                            block[i] = MACBlock(conv_layer.in_channels, conv_layer.out_channels)
                            break

    def make_layers(self, cfg, in_channels, batch_norm, input_size, mvap_kernel_size=5, use_maxpool=False) -> nn.Sequential:
        layers: List[nn.Module] = []
        curr_layer: List[nn.Module] = []
        _in_channels = in_channels
        if input_size >= 160:
            layers += [
                nn.Conv3d(_in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                (nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True) if use_maxpool else MultiDimAttentionPooling3d(64, kernel_size=mvap_kernel_size))
            ]
            _in_channels = 64
        else:
            layers += [
                nn.Conv3d(_in_channels, 64, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True)
            ]
            _in_channels = 64
        for v in cfg:
            if v == "M":
                curr_layer.append(nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True) if use_maxpool else MultiDimAttentionPooling3d(_in_channels, kernel_size=mvap_kernel_size))
            elif v == "F":
                if curr_layer:
                    layers.append(nn.Sequential(*curr_layer))
                    curr_layer = []
            else:
                v = cast(int, v)
                conv3d = nn.Conv3d(_in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    curr_layer += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    curr_layer += [conv3d, nn.ReLU(inplace=True)]
                _in_channels = v
        if curr_layer:
            layers.append(nn.Sequential(*curr_layer))
        return nn.Sequential(*layers)

    def forward(self, X):
        features = []
        X_ds = None
        if self.conv_at_start:
            X = self.starting_layers(X)
            X_ds = self.ds_layers(X)
        for layer in self.layers:
            X = layer(X)
            features.append(X)
        if self.conv_at_start:
            features[-4] = features[-4] + X_ds
        features_to_use = features[-4:]
        for i in range(4):
            if isinstance(self.attn_modules[i], WMHCA3D):
                features_to_use[i] = self.attn_modules[i](features_to_use[i], features_to_use[i])
            elif not isinstance(self.attn_modules[i], nn.Identity):
                features_to_use[i] = self.attn_modules[i](features_to_use[i])
        return self.fpn_neck(features_to_use)


# Swin Transformer FPN

def shifted_window_attention( # changed to 3D
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int] = [128, 128, 128],
    num_heads: int = 4,
    shift_size: List[int] = [64, 64, 64],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias for 3D.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, H, W, D, C]): The input tensor or 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
    Returns:
        Tensor[B, H, W, D, C]: The output tensor after shifted window attention.
    """
    B, H, W, D, C = input.shape
    # pad feature maps to multiples of window size -> 3D
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_d = (window_size[2] - D % window_size[2]) % window_size[2]
    x = F.pad(input, (0, 0, 0, pad_d, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, pad_D, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window -> 3D
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0
    if window_size[2] >= pad_D:
        shift_size[2] = 0

    # cyclic shift -> 3D
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows -> 3D
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1]) * (pad_D // window_size[2])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], pad_D // window_size[2], window_size[2], C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B * num_windows, window_size[0] * window_size[1] * window_size[2], C)  # B*nW, Ws*Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask #TODO: change to 3d
        attn_mask = x.new_zeros((pad_H, pad_W, pad_D))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        d_slices = ((0, -window_size[2]), (-window_size[2], -shift_size[2]), (-shift_size[2], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1], d[0] : d[1]] = count
                    count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], pad_D // window_size[2], window_size[2])
        attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(num_windows, window_size[0] * window_size[1] * window_size[2])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows -> 3D
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], pad_D // window_size[2], window_size[0], window_size[1], window_size[2], C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, pad_H, pad_W, pad_D, C)

    # reverse cyclic shift -> 3D
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features -> 3D
    x = x[:, :H, :W, :D, :].contiguous()
    return x


def _get_relative_position_bias( # changed to 3D
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1] * window_size[2]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


class ShiftedWindowAttention(nn.Module): # changed to 3D
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 3")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias -> 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1 * 2*Wd-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window -> 3D
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[2] - 1) * (2 * self.window_size[1] - 1) # problematic
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1) # problematic
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wd*Wh*Ww*Wd
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, D, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, D, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class SwinTransformerBlock(nn.Module): # changed to 3D
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module): # changed to 3D
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm, expand_dim: bool = True):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, dim*2 if expand_dim else dim, bias=False)
        self.norm = norm_layer(8 * dim)
    
    def _patch_merging_pad(self, x: torch.Tensor) -> torch.Tensor: # changed to 3D
        H, W, D, _ = x.shape[-4:]
        x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x1 = x[..., 1::2, 0::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x2 = x[..., 0::2, 1::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x3 = x[..., 1::2, 1::2, 0::2, :]  # ... H/2 W/2 D/2 C
        x4 = x[..., 0::2, 0::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x5 = x[..., 1::2, 0::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x6 = x[..., 0::2, 1::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x7 = x[..., 1::2, 1::2, 1::2, :]  # ... H/2 W/2 D/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # ... H/2 W/2 D/2 8*C
        return x

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, D, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, D/2, C]
        """
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 D/2 2*C
        return x


class SwinTransformer_FPN(nn.Module): # TODO: change to 3D
    """
    Implements the 3D Swin Transformer FPN. 
    Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-5),
        block: Optional[Callable[..., nn.Module]] = SwinTransformerBlock,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        expand_dim: bool = True,
        out_channels: int = 256,
        input_dim: int = 4
    ):
        super().__init__()
        self.out_channels = out_channels
        # split image into non-overlapping patches
        self.patch_partition = nn.Sequential(
            nn.Conv3d(input_dim, embed_dim, kernel_size=(patch_size[0], patch_size[1], patch_size[2]), 
                      stride=(patch_size[0], patch_size[1], patch_size[2])),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(embed_dim),
        )

        self.stages = nn.ModuleList()
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        fpn_in_channels = []

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = nn.ModuleList()
            dim = embed_dim * 2**i_stage if expand_dim else embed_dim
            fpn_in_channels.append(dim)

            # add patch merging layer
            if i_stage > 0:
                input_dim = fpn_in_channels[-2] if len(fpn_in_channels) > 1 else embed_dim
                stage.append(downsample_layer(input_dim, norm_layer, expand_dim))
            
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            self.stages.append(nn.Sequential(*stage))

        self.fpn_neck = FPN(fpn_in_channels, out_channels, len(fpn_in_channels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features : List[torch.Tensor] = []
        x = self.patch_partition(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous()) # [N, C, H, W, D]

        features = self.fpn_neck(features)
        return features
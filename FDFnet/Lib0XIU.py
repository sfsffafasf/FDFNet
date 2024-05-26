import torch
import torch.nn as nn
# from mmcv.cnn import ConvAWS2d, constant_init
# from mmcv.ops.deform_conv import deform_conv2d
import torchvision.models as models
import scipy.stats as st
from torch.nn import functional as F
import numpy as np
import cv2
from torch.nn.parameter import Parameter
from backbone.mix_transformer import mit_b0,mit_b4
# from toolbox.models.DCTMO0.lv import GaborLayer
import time

from toolbox.model.cai.修layer import MultiSpectralAttentionLayer
import math
from toolbox.model.cai.gumbel import GumbleSoftmax
# toolbox/models/DCTMO0/Lib0.pytoolbox/model/cai/gumbel.py

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 卷积
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        # self.conv = Dynamic_conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=dilation, bias=False)  ##改了动态卷积
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def adjust_kernel_size(input_channels, output_channels):
    # 计算基础的卷积核大小
    b = 1
    gamma = 2
    ratio = output_channels / input_channels
    kernel_size = int(abs((math.log(ratio, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    # 计算输入输出通道数的比值
    # 根据通道数比值调整卷积核大小
    return kernel_size
# class ChannelShuffle(nn.Module):
#     def __init__(self, groups):
#         super(ChannelShuffle, self).__init__()
#         self.groups = groups
#
#     def forward(self, x):
#         '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
#         N, C, H, W = x.size()
#         g = self.groups
#         return x.view(N, g, int(C // g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel//4, in_channel, 1),
            BasicConv2d(in_channel, in_channel, 3, padding=1, dilation=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel//4, in_channel, 1),
            BasicConv2d(in_channel, in_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel//4, in_channel, 1),
            BasicConv2d(in_channel, in_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel//4, in_channel, 1),
            BasicConv2d(in_channel, in_channel, 3, padding=7, dilation=7)
        )
        self.conv_p = nn.Sequential( nn.AdaptiveAvgPool2d(1),#最大池化分支
                                        BasicConv2d(in_channel, 4, 1))
        self.inp_gs = GumbleSoftmax()
        kernel_size = adjust_kernel_size(in_channel, out_channel)
        self.conv_cat = BasicConv2d(4 * in_channel, in_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        xx = torch.chunk(x,4, dim=1)
        m = self.conv_p(x)
        m = self.inp_gs(m, temp=1, force_hard=True)#代码
        wx0, wx1, wx2, wx3 = m[:, :1, :, :], m[:, 1:2, :, :], m[:, 2:3, :, :], m[:, 3:4, :, :]
        x0 = self.branch0(xx[0])*wx0
        x1 = self.branch1(xx[1])*wx1
        x2 = self.branch2(xx[2])*wx2
        x3 = self.branch3(xx[3])*wx3
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.upsample_2(self.conv_res(x+x_cat))
        return x

# BatchNorm2d = nn.BatchNorm2d
# BatchNorm1d = nn.BatchNorm1d
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=96):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x,y):
        x = 0.5*(x + torch.mean(x))
        y = 0.5*(y + torch.mean(y))

        identity = x+y
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class MFA0(nn.Module):
    def __init__(self, img1channel):
        super(MFA0, self).__init__()
        #b0 32, 64, 160, 256    64, 128, 320, 512
        # c2wh = dict([(32, 104), (64, 52), (160, 26), (256, 13)])
        c2wh = dict([(64, 104), (128, 52), (320, 26), (512, 13)])

        self.layer_img = BasicConv2d(img1channel*2, img1channel,kernel_size=3,stride=1, padding=1)

        self.MultiSpectralAttentionLayer = MultiSpectralAttentionLayer(img1channel, c2wh[img1channel], c2wh[img1channel], freq_sel_method = 'top16')
        self.MultiSpectralAttentionLayed = MultiSpectralAttentionLayer(img1channel, c2wh[img1channel],
                                                                       c2wh[img1channel], freq_sel_method='top16')
        self.layer_cat1 = BasicConv2d(img1channel*2 , img1channel, kernel_size=3, stride=1, padding=1)

    def forward(self,img1, dep1):
        ################################[2, 32, 28, 28]
        """
        :param ful: 2, 64, 52
        :param img1: 2, 32, 104
        :param dep1:
        :param img: 2,64,52
        :param dep:
        :return:
        """

        img2 = self.MultiSpectralAttentionLayer(img1)+img1
        dep2 = self.MultiSpectralAttentionLayer(dep1)+dep1
        weighting = self.layer_cat1(torch.cat([img2, dep2], dim=1))
        out = weighting
        return out


class MFA(nn.Module):
    def __init__(self, img1channel):
        super(MFA, self).__init__()
        #b0 32, 64, 160, 256    64, 128, 320, 512
        # c2wh = dict([(32, 104), (64, 52), (160, 26), (256, 13)])
        c2wh = dict([(64, 104), (128, 52), (320, 26), (512, 13)])
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.layer_img = BasicConv2d(img1channel*2, img1channel,kernel_size=3,stride=1, padding=1)
        # self.GaborLayer = GaborLayer(self.layer_img,)
        self.MultiSpectralAttentionLayer = MultiSpectralAttentionLayer(img1channel, c2wh[img1channel], c2wh[img1channel], freq_sel_method = 'top16')
        self.MultiSpectralAttentionLayed = MultiSpectralAttentionLayer(img1channel, c2wh[img1channel],
                                                                       c2wh[img1channel], freq_sel_method='top16')
        self.layer_cat1 = BasicConv2d(img1channel*2 , img1channel, kernel_size=3, stride=1, padding=1)

    def forward(self, ful, img1, dep1):
        ################################[2, 32, 28, 28]
        """
        :param ful: 2, 64, 52
        :param img1: 2, 32, 104
        :param dep1:
        :param img: 2,64,52
        :param dep:
        :return:
        """
        img2 = self.MultiSpectralAttentionLayer(img1)+img1*ful
        dep2 = self.MultiSpectralAttentionLayer(dep1)+dep1*ful
        weighting = self.layer_cat1(torch.cat([img2, dep2], dim=1))+ ful
        out = weighting
        return out


"""
rgb和d分别与融合的做乘法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
rgb和d分别与融合的做加法，然后拼接 后和卷积回原来的通道     x_ful_1_2_m
输出就是融合
"""
####################################################自适应1,2,3,6###########################

class LiSPNetx22(nn.Module):
    def __init__(self, channel=32):
        super(LiSPNetx22, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        # Backbone model   In
        # Backbone model32, 64, 160, 256
        # self.layer_dep0 = nn.Conv2d(3, 3, kernel_size=1)
        res_channels = [32, 64, 160, 256, 256]
        channels = [64, 128, 256, 512, 512]

        self.resnet = mit_b4()
        self.resnet_depth = mit_b4()
        self.resnet.init_weights("/home/data/李超/backbone/mit_b4.pth")
        self.resnet_depth.init_weights("/home/data/李超/backbone/mit_b4.pth")
        ###############################################
        # funsion encoders #
        ## rgb64, 128, 320, 512
        # channels = [32, 64, 160, 256]
        channels = [64, 128, 320, 512]
        self.ful_3 = MFA0(channels[3])
        self.ful_2 = MFA(channels[2])
        self.ful_1 = MFA(channels[1])
        self.ful_0 = MFA(channels[0])
        self.GCM_3 = GCM(channels[3], channels[2])
        self.GCM_2 = GCM(channels[2], channels[1])
        self.GCM_1 = GCM(channels[1], channels[0])
        self.GCM_0 = GCM(channels[0], 16)

        # self.conv_img_03 = nn.Sequential(nn.Conv2d(channels[3], channels[2], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        # self.conv_img_02 = nn.Sequential(nn.Conv2d(channels[2], channels[1], 1),
        #                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        # self.conv_img_01 = nn.Sequential(nn.Conv2d(channels[1], channels[0], 1),
        #                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv_out1 = nn.Conv2d(16, 1, 1)
        self.conv_out2 = nn.Conv2d(channels[0], 1, 1)
        self.conv_out3 = nn.Conv2d(channels[1], 1, 1)
        self.conv_out4 = nn.Conv2d(channels[2], 1, 1)
        # self.dualgcn = DualGCN(256)
        # self.CoordAtt = CoordAtt(channels[3])
        # self.BidirectionalAttention = BidirectionalAttention(channels[3])
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        # self.d3to2r = nn.Sequential(nn.Conv2d(channels[3], channels[2], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                        nn.BatchNorm2d(channels[2]), nn.LeakyReLU(inplace=True))
        # self.d3to2d = nn.Sequential(nn.Conv2d(channels[3], channels[2], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                        nn.BatchNorm2d(channels[2]), nn.LeakyReLU(inplace=True))
        # self.d2to1r = nn.Sequential(nn.Conv2d(channels[2], channels[1], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                        nn.BatchNorm2d(channels[1]), nn.LeakyReLU(inplace=True))
        # self.d2to1d = nn.Sequential(nn.Conv2d(channels[2], channels[1], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                        nn.BatchNorm2d(channels[1]), nn.LeakyReLU(inplace=True))
        # self.d1to0r = nn.Sequential(nn.Conv2d(channels[1], channels[0], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                        nn.BatchNorm2d(channels[0]), nn.LeakyReLU(inplace=True))
        # self.d1to0d = nn.Sequential(nn.Conv2d(channels[1], channels[0], 1),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                        nn.BatchNorm2d(channels[0]), nn.LeakyReLU(inplace=True))
        #
        # self.d1to3 = nn.Sequential(nn.Conv2d(channels[2], channels[1], 1),
        #                        nn.BatchNorm2d(channels[1]), nn.LeakyReLU(inplace=True))
        #
        # self.d1to4 = nn.Sequential(nn.Conv2d(channels[1], channels[0], 1),
        #                        nn.BatchNorm2d(channels[0]), nn.LeakyReLU(inplace=True))



    def forward(self, imgs, depths):

        # depths = imgs
        img_0, img_1, img_2, img_3 = self.resnet.forward_features(imgs)
        # print(img_0.shape, img_1.shape, img_2.shape, img_3.shape)
        ####################################################
        ## decoder rgb     ful_2.shape[2, 256, 14, 14]   img_3.shape [2, 256, 7, 7]
        ####################################################
        dep_0, dep_1, dep_2, dep_3 = self.resnet_depth.forward_features(depths)
        # img_03 = self.conv_img_03(img_3+dep_3)
        # img_02 = self.conv_img_02(img_2+dep_2+img_03)
        # img_01 = self.conv_img_01(img_02+img_1+dep_1)
        # x_rgb_0 = self.rgb1(img_01)
        # print(imgs.shape, img_0.shape, img_1.shape, img_2.shape, img_3.shape)torch.Size([2, 3, 416, 416]) torch.Size([2, 32, 104, 104]) torch.Size([2, 64, 52, 52]) torch.Size([2, 160, 26, 26]) torch.Size([2, 256, 13, 13])
        ## dep_2 = self.conv_img2(dep_2)##################################################
        # ful_03 = self.BidirectionalAttention(img_3,dep_3)  #512,13 #此处加了se注意力
        # print(ful_03.shape,'ful_03')
        # ful_03 = self.CoordAtt(img_3, dep_3)
        OUT_3 = self.ful_3(img_3,dep_3)
        ful_3 = self.GCM_3(OUT_3)
        # img_2 = img_2 + self.d3to2r(img_3)
        # dep_2 = dep_2 + self.d3to2d(dep_3)
        OUT_2 = self.ful_2(ful_3, img_2, dep_2)
        ful_2 = self.GCM_2(OUT_2)
        # img_1 = img_1 + self.d2to1r(img_2)
        # dep_1 = dep_1 + self.d2to1d(dep_2)
        OUT_1 = self.ful_1(ful_2, img_1, dep_1)
        ful_1 = self.GCM_1(OUT_1)
        # img_0 = img_0 + self.d1to0r(img_1)
        # dep_0 = dep_0 + self.d1to0d(dep_1)
        OUT_0 = self.ful_0(ful_1, img_0, dep_0)
        ful_0 = self.GCM_0(OUT_0)
        # ful_0 = self.ful_0(ful_1, img_01, dep_01)

        ########2，256，13      32,208     64,104
        ful1 = self.conv_out1(self.upsample_2(ful_0))
        ful2 = self.conv_out2(self.upsample_4(ful_1))
        ful3 = self.conv_out3(self.upsample_8(ful_2))
        ful4 = self.conv_out4(self.upsample_16(ful_3))
        return ful1,ful2,ful3,ful4,OUT_3,OUT_2,OUT_1,OUT_0,img_3,dep_3,img_2, dep_2, img_1, dep_1, img_0, dep_0,ful_3,ful_2,ful_1,ful_0


if __name__ == "__main__":
    rgb = torch.randn(2, 3, 416, 416).cuda()
    t = torch.randn(2, 3, 416, 416).cuda()
    model = LiSPNetx22().cuda()
    out = model(rgb,t)
    for i in range(len(out)):
        print(out[i].shape)#Flops:6.48 GMac  Params:13.88 M



    # from toolbox import compute_speed/
# Speed Time: 28.42 ms / iter   FPS: 35.18Flops:  4.57 GMac
# # Params: 10.03 M
# def contrastive_loss(out, out_aug, batch_size=2, hidden_norm=False, temperature=1.0):
#     if hidden_norm:
#         out = F.normalize(out, dim=-1)
#         out_aug = F.normalize(out_aug, dim=-1)
#     INF = float('inf')
#     labels = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size * 2)  # [batch_size,2*batch_size]
#     masks = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size)  # [batch_size,batch_size]
#     logits_aa = torch.matmul(out, out.transpose(0, 1)) / temperature  # [batch_size,batch_size]
#     logits_bb = torch.matmul(out_aug, out_aug.transpose(0, 1)) / temperature  # [batch_size,batch_size]
#     logits_aa = logits_aa - masks * INF  # remove the same samples in out
#     logits_bb = logits_bb - masks * INF  # remove the same samples in out_aug
#     logits_ab = torch.matmul(out, out_aug.transpose(0, 1)) / temperature
#     logits_ba = torch.matmul(out_aug, out.transpose(0, 1)) / temperature
#     loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
#     loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
#     loss = loss_a + loss_b
#     return loss
# class OFD(nn.Module):
#     '''
#     A Comprehensive Overhaul of Feature Distillation
#     http://openaccess.thecvf.com/content_ICCV_2019/papers/
#     Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
#     '''
#
#     def __init__(self, in_channels, out_channels):
#         super(OFD, self).__init__()
#         self.connector = nn.Sequential(*[
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels)
#         ])
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, fm_s, fm_t):
#         margin = self.get_margin(fm_t)
#         fm_t = torch.max(fm_t, margin)
#         fm_s = self.connector(fm_s)
#
#         mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
#         loss = torch.mean((fm_s - fm_t) ** 2 * mask)
#
#         return loss
#
#     def get_margin(self, fm, eps=1e-6):
#         mask = (fm < 0.0).float()
#         masked_fm = fm * mask
#
#         margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True) + eps)
#
#         return margin


# from toolbox import compute_speed
# from ptflops import get_model_complexity_info
# with torch.cuda.device(0):
#     net = LiSPNetx22().cuda()
#     flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=False)
#     compute_speed(net, input_size=(1, 3, 416, 416), iteration=500)
#     print('Flops:'+flops)
#     print('Params:'+params)
# print(a.shape)
# Flops:33.52 GMac
# Params:190.26 M
# print(a[1].shape)
# print(a[2].shape)
# print(a[3].shape)Elapsed Time: [17.54 s / 500 iter]
# Speed Time: 35.08 ms / iter   FPS: 28.51
# compute_speed(net,input_size=(1, 3, 416, 416), iteration=500)    此处加了se注意力
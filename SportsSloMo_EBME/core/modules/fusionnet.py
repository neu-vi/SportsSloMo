import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.common_op import conv2, conv3, conv4, deconv, deconv2, deconv3
from ..modules.softsplat import softsplat


def downsample_image(img, mask):
    """ down-sample the image [H*2, W*2, 3] -> [H, W, 2] using convex combination """
    N, _, H, W = img.shape
    mask = mask.view(N, 1, 25, H // 2, W // 2)
    mask = torch.softmax(mask, dim=2)

    down_img = F.unfold(img, [5,5], stride=2, padding=2)
    down_img = down_img.view(N, 3, 25, H // 2, W // 2)

    down_img = torch.sum(mask * down_img, dim=2)
    return down_img


class ContextNet(nn.Module):
    def __init__(self):
        c = 16
        super(ContextNet, self).__init__()
        self.conv1 = conv2(3, c)
        self.conv2 = conv2(c, 2*c)
        self.conv3 = conv2(2*c, 4*c)
        self.conv4 = conv2(4*c, 8*c)

    def forward(self, img, flow):
        feat_pyramid = []

        # calculate forward-warped feature pyramid
        feat = img
        feat = self.conv1(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat1 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat1)

        feat = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat2 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat2)

        feat = self.conv3(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat3 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat3)

        feat = self.conv4(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat4 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat4)

        return feat_pyramid



class FusionNet(nn.Module):
    def __init__(self, args):
        super(FusionNet, self).__init__()
        c = 16
        self.high_synthesis = args.high_synthesis if "high_synthesis" in args else False
        self.contextnet = ContextNet()
        self.down1 = conv4(16, 2*c)
        self.down2 = conv2(4*c, 4*c)
        self.down3 = conv2(8*c, 8*c)
        self.down4 = conv2(16*c, 16*c)
        self.up1 = deconv(32*c, 8*c)
        self.up2 = deconv(16*c, 4*c)
        self.up3 = deconv(8*c, 2*c)
        self.up4 = deconv3(4*c, c)
        self.refine_pred = nn.Conv2d(c, 4, 3, 1, 1)
        if self.high_synthesis:
            self.downsample_mask = nn.Sequential(
                nn.Conv2d(c, 2*c, 5, 2, 2),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 2*c, 3, 1, 1),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 25, 1, padding=0))

        # fix the paramters if needed
        if ("fix_pretrain" in args) and (args.fix_pretrain):
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, img0, img1, bi_flow, time_period=0.5, profile_time=False):
        # upsample input images and estimated bi_flow, if using the
        # "high_synthesis" setting.
        if self.high_synthesis:
            img0 = F.interpolate(input=img0, scale_factor=2.0, mode="bilinear", align_corners=False)
            img1 = F.interpolate(input=img1, scale_factor=2.0, mode="bilinear", align_corners=False)
            bi_flow = F.interpolate(
                    input=bi_flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0

        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        c0 = self.contextnet(img0, flow_0t)
        c1 = self.contextnet(img1, flow_1t)

        # feature extraction by u-net
        s0 = self.down1(torch.cat((warped_img0, warped_img1, img0, img1, flow_0t_1t), 1))
        s1 = self.down2(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down3(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down4(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up2(torch.cat((x, s2), 1))
        x = self.up3(torch.cat((x, s1), 1))
        x = self.up4(torch.cat((x, s0), 1))

        # prediction
        refine = self.refine_pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask = torch.sigmoid(refine[:, 3:4])
        merged_img = warped_img0 * refine_mask + warped_img1 * (1 - refine_mask)
        interp_img = merged_img + refine_res
        interp_img = torch.clamp(interp_img, 0, 1)

        # convex down-sampling, if using "high_synthesis" setting.
        if self.high_synthesis:
            downsample_mask = self.downsample_mask(x)
            interp_img = downsample_image(interp_img, downsample_mask)

        return interp_img



if __name__ == "__main__":
    pass

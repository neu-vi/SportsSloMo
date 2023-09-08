import torch
import math
import numpy
import torch.nn.functional as F
import torch.nn as nn

from ..utils import correlation
from ..modules.softsplat import softsplat


backwarp_tenGrid = {}
backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    """Backward warping based on grid_sample

    Args:
        tenInput: data tensor of shape N, C, H, W
        tenFlow: optical flow tensor of shape N, 2, H, W

    Returns:
        A new tensor of shape N, C, H, W, which is sampled from tenInput according to the coordinates defined by
        tenFlow.
    """
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]),
                tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]),
                tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
                [ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = F.grid_sample(input=tenInput,
            grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
            mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask


#**************************************************************************************************#
# => Feature Pyramid
#**************************************************************************************************#
class FeatPyramid(nn.Module):
    """Two-level feature pyramid
    1) remove high-level feature pyramid (compared to PWC-Net), and add more conv layers to stage 2;
    2) do not increase the output channel of stage 2, in order to keep the cost of corr volume under control.
    """
    def __init__(self):
        super(FeatPyramid, self).__init__()
        c = 24
        self.conv_stage1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))

    def forward(self, img):
        C1 = self.conv_stage1(img)
        C2 = self.conv_stage2(C1)

        return [C1, C2]



#**************************************************************************************************#
# => Estimator
#**************************************************************************************************#
class Estimator(nn.Module):
    """A 6-layer flow estimator, with correlation-injected features
    1) construct partial cost volume with the CNN features from stage 2 of the feature pyramid;
    2) estimate bi-directional flows, by feeding cost volume, CNN features for both warped images,
    CNN feature and estimated flow from previous iteration.
    """
    def __init__(self):
        super(Estimator, self).__init__()
        corr_radius = 4
        image_feat_channel = 48
        last_flow_feat_channel = 64
        in_channels = (corr_radius*2 + 1) ** 2 + image_feat_channel * 2 \
                + last_flow_feat_channel + 4

        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=160,
                    kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=128,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=112,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=112, out_channels=96,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=64,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer6 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=4,
                    kernel_size=3, stride=1, padding=1))


    def forward(self, feat0, feat1, last_feat, last_flow):
        corr_fn=correlation.FunctionCorrelation
        volume = F.leaky_relu(
                input=corr_fn(tenFirst=feat0, tenSecond=feat1), negative_slope=0.1, inplace=False)
        input_feat = torch.cat([volume, feat0, feat1, last_feat, last_flow], 1)
        feat = self.conv_layer1(input_feat)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)
        feat = self.conv_layer4(feat)
        feat = self.conv_layer5(feat)
        flow = self.conv_layer6(feat)

        return flow, feat




#**************************************************************************************************#
# => BiFlowNet
#**************************************************************************************************#
class BiFlowNet(nn.Module):
    """Our bi-directional flownet
    In general, we combine image pyramid, middle-oriented forward warping,
    lightweight feature encoder and cost volume for simultaneous bi-directional
    motion estimation.
    """
    def __init__(self, args):
        super(BiFlowNet, self).__init__()
        self.pyr_level = args.pyr_level if "pyr_level" in args else 3
        self.warp_type = args.warp_type if "warp_type" in args else "middle-forward"
        self.feat_pyramid = FeatPyramid()
        self.flow_estimator = Estimator()

        # fix the paramters if needed
        if ("fix_pretrain" in args) and (args.fix_pretrain):
            for p in self.parameters():
                p.requires_grad = False

    def forward_one_iteration(self, img0, img1, last_feat, last_flow, warp_type=None):
        """ estimate flows for one image pyramid level

        Before feature extraction, we perform warping for input images to
        compensate estimated motion.  We have three options for warping, which is
        specialized by the `warp_type` parameter.

        warp_type:
            0) None: do not perform warping for input images
            1) "backward": backward warping input frames towards each other
            2) "middle-forward": forward warping input frames towards the hidden
            middle frame
            3) "forward": forward warping input frames towards each other
        """
        assert warp_type in [None, "backward", "middle-forward", "forward"]

        up_flow = F.interpolate(input=last_flow, scale_factor=4.0, mode="bilinear", align_corners=False)
        if warp_type == "backward":
            img0 = backwarp(tenInput=img0, tenFlow=up_flow[:, 2:])
            img1 = backwarp(tenInput=img1, tenFlow=up_flow[:, :2])
        if warp_type == "middle-forward":
            img0 = softsplat.FunctionSoftsplat(
                    tenInput=img0, tenFlow=up_flow[:, :2]*0.5, tenMetric=None, strType='average')
            img1 = softsplat.FunctionSoftsplat(
                    tenInput=img1, tenFlow=up_flow[:, 2:]*0.5, tenMetric=None, strType='average')
        if warp_type == "forward":
            img0 = softsplat.FunctionSoftsplat(
                    tenInput=img0, tenFlow=up_flow[:, :2]*1.0, tenMetric=None, strType='average')
            img1 = softsplat.FunctionSoftsplat(
                    tenInput=img1, tenFlow=up_flow[:, 2:]*1.0, tenMetric=None, strType='average')

        feat0 = self.feat_pyramid(img0)[-1]
        feat1 = self.feat_pyramid(img1)[-1]

        flow, feat = self.flow_estimator(feat0, feat1, last_feat, last_flow)
        return flow, feat

    def forward(self, img0, img1):
        N, _, H, W = img0.shape
        last_flow_feat_channel = 64

        for level in list(range(self.pyr_level))[::-1]:
            scale_factor = 1 / 2 ** level
            img0_down = F.interpolate(input=img0, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False)
            img1_down = F.interpolate(input=img1, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False)

            if level == self.pyr_level - 1:
                last_flow = torch.zeros((N, 4, H // (2 ** (level+2)), W //(2 ** (level+2)))).to(img0.device)
                last_feat = torch.zeros((N, last_flow_feat_channel,
                    H // (2 ** (level+2)), W // (2 ** (level+2)))).to(img0.device)
                warp_type = None
            else:
                last_flow = F.interpolate(input=flow, scale_factor=2.0,
                        mode="bilinear", align_corners=False) * 2
                last_feat = F.interpolate(input=feat, scale_factor=2.0,
                        mode="bilinear", align_corners=False)
                warp_type = self.warp_type

            flow, feat = self.forward_one_iteration(img0_down, img1_down,
                    last_feat, last_flow, warp_type)

        # directly up-sample estimated flow to full resolution with bi-linear interpolation
        output_flow = F.interpolate(input=flow, scale_factor=4.0, mode="bilinear", align_corners=False)

        return output_flow





if __name__ == "__main__":
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ResUNet import ResUNet

'''
Referred to https://github.com/Mhaiyang/CVPR2021_PFNet
'''

################### Channel Attention Block ######################

class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width, length = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, length)

        out = self.gamma * out + x
        return out


################## Spatial Attention Block ######################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width, length = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height * length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * length)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height * length)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, length)

        out = self.gamma * out + x
        return out


################### ASPP ####################

class ASPP_Block(nn.Module):
    def __init__(self, input_channels):
        super(ASPP_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv3d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv3d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv3d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv3d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv3d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm3d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv3d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm3d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


##################### Identify Module ########################

class Identify(nn.Module):
    def __init__(self, channel):
        super(Identify, self).__init__()
        self.channel = channel
        self.channel_attention = CA_Block(self.channel)
        self.spatial_attention = SA_Block(self.channel)
        self.map_prediction = nn.Conv3d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        spatial_attention = self.sab(channel_attention)
        prediction = self.map_prediction(spatial_attention)

        return spatial_attention, prediction


######################## Discern Module ##########################

class Discern(nn.Module):
    def __init__(self, channel1, channel2):
        super(Discern, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.feature_up_sample = nn.Sequential(nn.Conv3d(self.channel2, self.channel1, 3, 1, 1),
                                nn.BatchNorm3d(self.channel1), nn.ReLU(),
                                nn.ConvTranspose3d(self.channel1, self.channel1, 2, 2))

        self.input_prediction = nn.Sequential(nn.ConvTranspose3d(1, 1, 2, 2), nn.Sigmoid())
        self.output_prediction = nn.Conv3d(self.channel1, 1, 3, 1, 1)

        self.false_positive = ASPP_Block(self.channel1)
        self.false_negative = ASPP_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm3d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, last_prediction):
        # x; current-level encoder features
        # y: higher-level decoder features
        # last_prediction: higher-level decoder prediction

        up = self.feature_up_sample(y)

        last_prediction = self.input_prediction(last_prediction)
        f_feature = x * last_prediction
        b_feature = x * (1 - last_prediction)

        false_positive = self.false_positive(f_feature)
        false_negative = self.false_negative(b_feature)

        refined_feature = up - (self.alpha * false_positive)
        refined_feature = self.bn1(refined_feature)
        refined_feature = self.relu1(refined_feature)

        refined_feature = refined_feature + (self.beta * false_negative)
        refined_feature = self.bn2(refined_feature)
        refined_feature = self.relu2(refined_feature)

        output_prediction = self.output_prediction(refined_feature)

        return refined_feature, output_prediction


###################################################################
# ########################## Identify-to-Discern NETWORK ##############################
###################################################################
class IDNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(IDNet, self).__init__()
        # params

        # backbone
        resUNet = ResUNet(training=True)
        self.layer0 = nn.Sequential(nn.Conv3d(3, 16, 3, 1, padding=1),
                                    nn.InstanceNorm3d(16),
                                    nn.LeakyReLU(), )
        self.layer1 = nn.Sequential(nn.Conv3d(16, 16, 2, 2),
                                    nn.Conv3d(16, 32, 3, 1, padding=1),
                                    nn.InstanceNorm3d(32),
                                    nn.LeakyReLU(), )
        self.layer2 = resUNet.encoder_stage1
        self.layer3 = resUNet.encoder_stage2
        self.layer4 = resUNet.encoder_stage3

        # channel reduction |  skip connection
        self.cr4 = nn.Sequential(nn.Conv3d(512, 512, 3, 1, 1), nn.BatchNorm3d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv3d(256, 256, 3, 1, 1), nn.BatchNorm3d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU())

        # positioning
        self.identify = Identify(512)

        # focus Focus(x, y) x: encode feature channels; y: former decode feature channels.
        self.discern3 = Discern(256, 512)
        self.discern2 = Discern(64, 256)
        self.discern1 = Discern(32, 64)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x_t1, x_t2, x_flair):
        x = torch.cat((x_t1, x_t2), 1)
        x = torch.cat((x, x_flair), 1)
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # positioning
        attention_features, predict4 = self.identify(cr4)

        # focus
        decoder_features3, predict3 = self.discern3(cr3, attention_features, predict4)
        decoder_features2, predict2 = self.discern2(cr2, decoder_features3, predict3)
        decoder_features1, predict1 = self.discern1(cr1, decoder_features2, predict2)

        predict0 = F.interpolate(predict1, size=x.size()[2:], mode='trilinear', align_corners=True)

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1), torch.sigmoid(predict0)

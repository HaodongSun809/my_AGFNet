import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Cross_Fusion_Module(nn.Module):
    def __init__(self,channel_list, out_channel):
        super(Cross_Fusion_Module, self).__init__()
        self.conv_level_1__3 = nn.Sequential(nn.Conv2d(channel_list[0],out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_1__1 = nn.Sequential(nn.Conv2d(channel_list[0],out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.up_level_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.parameter_level_1_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_1_beta = nn.Parameter(torch.ones(1))

        self.conv_level_2__3 = nn.Sequential(nn.Conv2d(channel_list[1] + out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_2__1 = nn.Sequential(nn.Conv2d(channel_list[1]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.parameter_level_2_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_2_beta = nn.Parameter(torch.ones(1))

        self.up_level_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_level_3__3 = nn.Sequential(nn.Conv2d(channel_list[2]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_3__1 = nn.Sequential(nn.Conv2d(channel_list[2]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.parameter_level_3_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_3_beta = nn.Parameter(torch.ones(1))

        self.up_level_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_level_4__3 = nn.Sequential(nn.Conv2d(channel_list[3]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_4__1 = nn.Sequential(nn.Conv2d(channel_list[3]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.parameter_level_4_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_4_beta = nn.Parameter(torch.ones(1))

    def forward(self, features):
        features_level_1 = self.parameter_level_1_alpha * self.conv_level_1__3(features[0]) + self.parameter_level_1_beta * self.conv_level_1__1(features[0])
        
        features[1] = torch.cat((features[1], self.up_level_1(features_level_1)), dim=1)
        features_level_2 = self.parameter_level_2_alpha * self.conv_level_2__3(features[1]) + self.parameter_level_2_beta * self.conv_level_2__1(features[1])
        
        features[2] = torch.cat((features[2], self.up_level_2(features_level_2)), dim=1)
        features_level_3 = self.parameter_level_3_alpha * self.conv_level_3__3(features[2]) + self.parameter_level_3_beta * self.conv_level_3__1(features[2])
        
        features[3] = torch.cat((features[3], self.up_level_3(features_level_3)), dim=1)
        features_level_4 = self.parameter_level_4_alpha * self.conv_level_4__3(features[3]) + self.parameter_level_4_beta * self.conv_level_4__1(features[3])

        out_features = features_level_4 

        return out_features


class Cross_Fusion_Module_v2(nn.Module):
    def __init__(self,channel_list, out_channel):
        super(Cross_Fusion_Module_v2, self).__init__()
        self.conv_level_1__3 = nn.Sequential(nn.Conv2d(channel_list[0],out_channel,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_1__1 = nn.Sequential(nn.Conv2d(channel_list[0],out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_1__5 = nn.Sequential(nn.Conv2d(channel_list[0],out_channel,5,1,2), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.up_level_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.parameter_level_1_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_1_beta = nn.Parameter(torch.ones(1))
        self.parameter_level_1_theta = nn.Parameter(torch.ones(1))

        self.conv_level_2__3 = nn.Sequential(nn.Conv2d(channel_list[1]+ out_channel,out_channel,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_2__1 = nn.Sequential(nn.Conv2d(channel_list[1]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_2__5 = nn.Sequential(nn.Conv2d(channel_list[1]+ out_channel,out_channel,5,1,2), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.parameter_level_2_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_2_beta = nn.Parameter(torch.ones(1))
        self.parameter_level_2_theta = nn.Parameter(torch.ones(1))

        self.up_level_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_level_3__3 = nn.Sequential(nn.Conv2d(channel_list[2]+ out_channel,out_channel,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_3__1 = nn.Sequential(nn.Conv2d(channel_list[2]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_3__5 = nn.Sequential(nn.Conv2d(channel_list[2]+ out_channel,out_channel,5,1,2), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.parameter_level_3_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_3_beta = nn.Parameter(torch.ones(1))
        self.parameter_level_3_theta = nn.Parameter(torch.ones(1))


        self.up_level_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_level_4__3 = nn.Sequential(nn.Conv2d(channel_list[3]+ out_channel,out_channel,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_4__1 = nn.Sequential(nn.Conv2d(channel_list[3]+ out_channel,out_channel,3,1,1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv_level_4__5 = nn.Sequential(nn.Conv2d(channel_list[3]+ out_channel,out_channel,5,1,2), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.parameter_level_4_alpha = nn.Parameter(torch.ones(1))
        self.parameter_level_4_beta = nn.Parameter(torch.ones(1))
        self.parameter_level_4_theta = nn.Parameter(torch.ones(1))
    
    def forward(self, features):
        features_level_1 = self.parameter_level_1_alpha * self.conv_level_1__3(features[0]) + self.parameter_level_1_beta * self.conv_level_1__1(features[0]) + self.parameter_level_1_theta * self.conv_level_1__5(features[0])
        
        features[1] = torch.cat((features[1], self.up_level_1(features_level_1)), dim=1)
        features_level_2 = self.parameter_level_2_alpha * self.conv_level_2__3(features[1]) + self.parameter_level_2_beta * self.conv_level_2__1(features[1]) + self.parameter_level_2_theta * self.conv_level_2__5(features[1])
        
        features[2] = torch.cat((features[2], self.up_level_2(features_level_2)), dim=1)
        features_level_3 = self.parameter_level_3_alpha * self.conv_level_3__3(features[2]) + self.parameter_level_3_beta * self.conv_level_3__1(features[2]) + self.parameter_level_3_theta * self.conv_level_3__5(features[2])
        
        features[3] = torch.cat((features[3], self.up_level_3(features_level_3)), dim=1)
        features_level_4 = self.parameter_level_4_alpha * self.conv_level_4__3(features[3]) + self.parameter_level_4_beta * self.conv_level_4__1(features[3]) + self.parameter_level_4_theta * self.conv_level_4__5(features[3])

        out_features = features_level_4 

        return out_features

class LocationGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(LocationGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

class Information_Guide_Module(nn.Module):
    def __init__(self, channel,num_classes):
        super(Information_Guide_Module, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.out = nn.Sequential(BasicConv2d(channel,channel,3,1,1),BasicConv2d(channel,channel,3,1,1),nn.Conv2d(channel,num_classes,1))

    def forward(self, features, map):
        map =  F.sigmoid(map)
        # fore_map = torch.zeros_like(map)
        # back_map = torch.zeros_like(map)
        # fore_map[map < 0.5] = 0.5
        # fore_map[map >= 0.5] = 1

        # back_map[map < 0.5] = 1
        # back_map[map >= 0.5] = 0.5

        fore_features = map *features
        back_features = (1 - map) *features

        final_features = self.alpha * fore_features + self.beta * back_features + features

        out = self.out(final_features)

        return out





class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1




class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SCNet(nn.Module):
    def __init__(self, channel=32):
        super(SCNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


    def forward(self, img, depth):

        # backbone
        pvt = self.backbone(img)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention


        # CFM
        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        return prediction1_8, prediction2_8


class SCNet_multi(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.Translayer2_0_d = BasicConv2d(64, channel, 1)
        self.Translayer2_1_d = BasicConv2d(128, channel, 1)
        self.Translayer3_1_d = BasicConv2d(320, channel, 1)
        self.Translayer4_1_d = BasicConv2d(512, channel, 1)

        self.CFM = CFM(channel)
        self.CFM_d = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.ca_d = ChannelAttention(64)
        self.sa_d = SpatialAttention()

        self.SAM = SAM()

        self.SAM_d = SAM()
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


    def forward(self, img, depth):

        # backbone
        pvt = self.backbone(img)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        pvt_d = self.backbone(depth)
        d1 = pvt_d[0]
        d2 = pvt_d[1]
        d3 = pvt_d[2]
        d4 = pvt_d[3]

        
        # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention

        d1 = self.ca(d1) * d1 # channel attention
        cim_feature_d = self.sa(d1) * d1 # spatial attention

        x2_t_d = self.Translayer2_1_d(d2)  
        x3_t_d = self.Translayer3_1_d(d3)  
        x4_t_d = self.Translayer4_1_d(d4)

        cfm_feature_d = self.CFM_d(x4_t_d, x3_t_d, x2_t_d)


        # CFM
        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        cfm_feature = cfm_feature + cfm_feature_d
        cim_feature = cim_feature + cim_feature_d

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        return prediction1_8, prediction2_8


class SCNet_multi_V2(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi_V2, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.alpha_a_1 = nn.Parameter(torch.ones(1))
        self.alpha_b_1 = nn.Parameter(torch.ones(1))

        self.alpha_a_2 = nn.Parameter(torch.ones(1))
        self.alpha_b_2 = nn.Parameter(torch.ones(1))

        self.alpha_a_3 = nn.Parameter(torch.ones(1))
        self.alpha_b_3 = nn.Parameter(torch.ones(1))

        self.alpha_a_4 = nn.Parameter(torch.ones(1))
        self.alpha_b_4 = nn.Parameter(torch.ones(1))

        self.concat_reduce_1 = BasicConv2d(2*64, 64, 3, 1, 1)
        self.concat_reduce_2 = BasicConv2d(2*128, 128, 3, 1, 1)
        self.concat_reduce_3 = BasicConv2d(2*320, 320, 3, 1, 1)
        self.concat_reduce_4 = BasicConv2d(2*512, 512, 3, 1, 1)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)


        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()


        self.SAM = SAM()

        self.SAM_d = SAM()
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


    def forward(self, img, depth):

        # backbone
        pvt_d = self.backbone(depth)
        pvt = self.backbone(img)
        x1 = self.alpha_a_1 * pvt[0] + self.alpha_b_1 * pvt_d[0]
        x2 = self.alpha_a_2 * pvt[1] + self.alpha_b_2 * pvt_d[1]
        x3 = self.alpha_a_3 * pvt[2] 
        x4 = self.alpha_a_4 * pvt[3]


        
        # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention



        # CFM
        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)


        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        return prediction1_8, prediction2_8


class SCNet_multi_V3(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi_V3, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.alpha_a_1 = nn.Parameter(torch.ones(1))
        self.alpha_b_1 = nn.Parameter(torch.ones(1))

        self.alpha_a_2 = nn.Parameter(torch.ones(1))
        self.alpha_b_2 = nn.Parameter(torch.ones(1))

        self.CFM = Cross_Fusion_Module([512,320,128,64],channel)

        self.out_1 = nn.Conv2d(channel,1,3,1)


    def forward(self, img, depth):

        # backbone
        pvt_d = self.backbone(depth)
        pvt = self.backbone(img)
        x1 = self.alpha_a_1 * pvt[0] + self.alpha_b_1 * pvt_d[0]
        x2 = self.alpha_a_2 * pvt[1] + self.alpha_b_2 * pvt_d[1]
        x3 = pvt[2] 
        x4 = pvt[3]
        features = [x4,x3,x2,x1]

        features = self.CFM(features)

        out_1 = self.out_1(features)


        return F.interpolate(out_1, img.size()[2:], mode='bilinear', align_corners=True)



class SCNet_multi_V4(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi_V4, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.alpha_a_1 = nn.Parameter(torch.ones(1))
        self.alpha_b_1 = nn.Parameter(torch.ones(1))

        self.alpha_a_2 = nn.Parameter(torch.ones(1))
        self.alpha_b_2 = nn.Parameter(torch.ones(1))

        self.CFM = Cross_Fusion_Module([512,320,128,64],channel)

        self.out_1 = nn.Conv2d(channel,1,3,1,1)

        self.IGM = Information_Guide_Module(channel,1)


    def forward(self, img, depth):

        # backbone
        pvt_d = self.backbone(depth)
        pvt = self.backbone(img)
        x1 = self.alpha_a_1 * pvt[0] + self.alpha_b_1 * pvt_d[0]
        x2 = self.alpha_a_2 * pvt[1] + self.alpha_b_2 * pvt_d[1]
        x3 = pvt[2] 
        x4 = pvt[3]
        features = [x4,x3,x2,x1]

        features = self.CFM(features)

        out_1 = self.out_1(features)

        out_2 = self.IGM(features, out_1)


        return F.interpolate(out_1, img.size()[2:], mode='bilinear', align_corners=True), F.interpolate(out_2, img.size()[2:], mode='bilinear', align_corners=True)


class SCNet_multi_V5(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi_V5, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.alpha_a_1 = nn.Parameter(torch.ones(1))
        self.alpha_b_1 = nn.Parameter(torch.ones(1))

        self.alpha_a_2 = nn.Parameter(torch.ones(1))
        self.alpha_b_2 = nn.Parameter(torch.ones(1))

        self.CFM_1 = Cross_Fusion_Module([512,320,128,64],channel)

        self.LGM_1 = LocationGuideModule(512)
        self.LGM_2 = LocationGuideModule(320)

        self.CFM_2 = Cross_Fusion_Module([512,320,128,64],channel)

        self.out_1 = nn.Conv2d(channel,1,3,1,1)
        self.out_2 = nn.Conv2d(channel,1,3,1,1)


    def forward(self, img, depth):

        # backbone
        pvt_d = self.backbone(depth)
        pvt = self.backbone(img)
        x1 = self.alpha_a_1 * pvt[0] + self.alpha_b_1 * pvt_d[0]
        x2 = self.alpha_a_2 * pvt[1] + self.alpha_b_2 * pvt_d[1]
        x3 = pvt[2] 
        x4 = pvt[3]
        features = [x4,x3,x2,x1]

        init_features = self.CFM_1(features)
        out_1 = self.out_1(init_features)

        level_0 = self.LGM_1(pvt[3], F.interpolate(out_1, features[0].size()[2:], mode='bilinear', align_corners=True))

        level_1 = self.LGM_2(pvt[2], F.interpolate(out_1, features[1].size()[2:], mode='bilinear', align_corners=True))

        new_features = [level_0, level_1, x2, x1]

        final_features = self.CFM_2(new_features)

        out_2 = self.out_2(final_features)


        return F.interpolate(out_1, img.size()[2:], mode='bilinear', align_corners=True), F.interpolate(out_2, img.size()[2:], mode='bilinear', align_corners=True)


class SCNet_multi_V6(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi_V6, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.alpha_a_1 = nn.Parameter(torch.ones(1))
        self.alpha_b_1 = nn.Parameter(torch.ones(1))

        self.alpha_a_2 = nn.Parameter(torch.ones(1))
        self.alpha_b_2 = nn.Parameter(torch.ones(1))

        self.CFM_1 = Cross_Fusion_Module([512,320,128,64],channel)

        self.LGM = LocationGuideModule(channel)

        self.out_1 = nn.Conv2d(channel,1,3,1,1)
        self.out_2 = nn.Conv2d(channel,1,3,1,1)


    def forward(self, img, depth):

        # backbone
        pvt_d = self.backbone(depth)
        pvt = self.backbone(img)
        x1 = self.alpha_a_1 * pvt[0] + self.alpha_b_1 * pvt_d[0]
        x2 = self.alpha_a_2 * pvt[1] + self.alpha_b_2 * pvt_d[1]
        x3 = pvt[2] 
        x4 = pvt[3]
        features = [x4,x3,x2,x1]

        init_features = self.CFM_1(features)
        out_1 = self.out_1(init_features)

        final_features = self.LGM(init_features,out_1)
        out_2 = self.out_2(final_features)


        return F.interpolate(out_1, img.size()[2:], mode='bilinear', align_corners=True), F.interpolate(out_2, img.size()[2:], mode='bilinear', align_corners=True)


class SCNet_multi_V7(nn.Module):
    def __init__(self, channel=32):
        super(SCNet_multi_V7, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer_0 = BasicConv2d(64, channel, 1)
        self.Translayer_1 = BasicConv2d(128, channel, 1)
        self.Translayer_2 = BasicConv2d(320, channel, 1)
        self.Translayer_3 = BasicConv2d(512, channel, 1)

        self.alpha_a_1 = nn.Parameter(torch.ones(1))
        self.alpha_b_1 = nn.Parameter(torch.ones(1))

        self.alpha_a_2 = nn.Parameter(torch.ones(1))
        self.alpha_b_2 = nn.Parameter(torch.ones(1))

        self.CFM_1 = Cross_Fusion_Module_v2([channel,channel,channel,channel],channel)

        self.LGM = LocationGuideModule(channel)

        self.out_1 = nn.Conv2d(channel,1,3,1,1)
        self.out_2 = nn.Conv2d(channel,1,3,1,1)


    def forward(self, img, depth):

        # backbone
        pvt_d = self.backbone(depth)
        pvt = self.backbone(img)
        x1 = self.Translayer_0(self.alpha_a_1 * pvt[0] + self.alpha_b_1 * pvt_d[0])
        x2 = self.Translayer_1(self.alpha_a_2 * pvt[1] + self.alpha_b_2 * pvt_d[1])
        x3 = self.Translayer_2(pvt[2])
        x4 = self.Translayer_3(pvt[3])
        features = [x4,x3,x2,x1]

        init_features = self.CFM_1(features)
        out_1 = self.out_1(init_features)

        final_features = self.LGM(init_features,out_1)
        out_2 = self.out_2(final_features)


        return F.interpolate(out_1, img.size()[2:], mode='bilinear', align_corners=True), F.interpolate(out_2, img.size()[2:], mode='bilinear', align_corners=True)
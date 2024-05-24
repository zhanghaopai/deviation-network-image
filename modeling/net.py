import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.networks import build_feature_extractor, NET_OUT_DIM

class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.feature_extractor = build_feature_extractor(self.args.backbone)
        self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)


    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        # n_scales：提取多少个特征
        for s in range(self.args.n_scales):
            image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            # 使用resnet-18提取特征--------->特征学习器
            feature = self.feature_extractor(image_scaled)

            # 获得特征分数
            scores = self.conv(feature)
            if self.args.topk > 0:
                # 只取分数topK
                scores = scores.view(int(scores.size(0)), -1)   # scores:[48, 1, 14, 14] ---> scores[48, 196]
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0] # score:[48, topk]
                scores = torch.mean(scores, dim=1).view(-1, 1)         # score:[48, 1]
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)
        _, y_pred = torch.max(scores, dim=1)
        score = torch.mean(scores, dim=1)
        score = score.view(-1, 1)
        return score, y_pred

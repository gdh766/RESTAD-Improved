#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn


class RBFLayer(nn.Module):
    def __init__(self, centers_dim):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(torch.Tensor(*centers_dim))
        self.log_gamma = nn.Parameter(torch.Tensor([1.0]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centers, mean=0, std=1)
        nn.init.normal_(self.log_gamma, mean=0, std=1)

    def forward(self, x):
        x = x.unsqueeze(1) - self.centers.unsqueeze(0).unsqueeze(2)
        x = x ** 2
        x = torch.sum(x, dim=-1)
        output = torch.exp(-0.5 * torch.exp(self.log_gamma) * x )  

        final_output = output.permute(0, 2, 1)
        return final_output

    # ------------------------------------------------------------------
    # 【毕设修改 M2】RBF 参数正则项（附加方法，不影响上面的 forward）
    #
    # 动机：原作者对 centers 用 N(0,1) 随机初始化、对 log_gamma 用 N(0,1)，
    # 且训练过程中两者都不受任何约束。这会带来两种退化：
    #   (1) 中心坍缩：多个中心收敛到几乎同一点，等价于中心数变少，
    #       论文图 6 里"中心数增加到一定程度后性能不再提升"与此吻合；
    #   (2) 尺度饱和：exp(log_gamma) 过大时 RBF 变成近似阶跃函数，
    #       过小时所有中心给出几乎相同的高相似度，两者都会让
    #       z 失去区分能力（εs 退化为常数）。
    # 这里给出两个可选正则：
    #   center_div_lambda : 惩罚中心两两余弦相似度（把中心推开）
    #   gamma_reg_lambda  : 把 log_gamma 拉向目标值 gamma_reg_target
    # 两个系数默认都是 0，此时返回 0，原训练行为完全不变。
    # ------------------------------------------------------------------
    def regularization(self, center_div_lambda=0.0, gamma_reg_lambda=0.0,
                       gamma_reg_target=-2.0):
        zero = self.centers.new_zeros(())
        details = {"center_div": 0.0, "gamma_reg": 0.0}
        loss = zero

        if center_div_lambda > 0 and self.centers.size(0) > 1:
            unit = torch.nn.functional.normalize(self.centers, dim=1)
            sim = unit @ unit.t()
            m = self.centers.size(0)
            off_diag_mean = (sim.sum() - sim.diag().sum()) / (m * (m - 1))
            loss = loss + center_div_lambda * off_diag_mean
            details["center_div"] = float(off_diag_mean.detach())

        if gamma_reg_lambda > 0:
            deviation = (self.log_gamma - gamma_reg_target) ** 2
            loss = loss + gamma_reg_lambda * deviation.mean()
            details["gamma_reg"] = float(deviation.mean().detach())

        return loss, details
    


    
    
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        normalization_fn,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(normalization_fn(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(
        self,
        n_colors=3,
        n_resblocks=16,
        n_feats=64,
        res_scale=1,
        conv=default_conv,
        bn=False,
        normalization_fn=None,
    ):
        super(EDSR, self).__init__()
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        m_head = [conv(n_colors, n_feats, kernel_size)]

        m_body = [
            ResBlock(
                conv,
                n_feats,
                kernel_size,
                act=act,
                res_scale=res_scale,
                bn=bn,
                normalization_fn=normalization_fn,
            )
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        return res

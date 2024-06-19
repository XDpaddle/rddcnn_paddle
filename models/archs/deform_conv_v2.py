import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DeformConv2d(nn.Layer):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.Pad2D(padding)
        self.conv = nn.Conv2D(inc, outc, kernel_size=kernel_size, stride=kernel_size, padding=0, bias_attr=bias)
        self.p_conv = nn.Conv2D(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2D(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.initializer.Constant(0.0)(self.p_conv.weight)
        if self.modulation:
            nn.initializer.Constant(0.0)(self.m_conv.weight)

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = F.sigmoid(self.m_conv(x))
        dtype = offset.dtype
        ks = self.kernel_size
        N = offset.shape[1] // 2
        if self.padding:
            x = self.zero_padding(x)
        # Generating the sampling grid
        p = self._get_p(offset, dtype)
        p = p.transpose([0, 2, 3, 1])
        q_lt = p.floor()
        q_rb = q_lt + 1
        q_lt = paddle.concat([paddle.clip(q_lt[..., :N], 0, x.shape[2] - 1), paddle.clip(q_lt[..., N:], 0, x.shape[3] - 1)], axis=-1).astype('int32')
        q_rb = paddle.concat([paddle.clip(q_rb[..., :N], 0, x.shape[2] - 1), paddle.clip(q_rb[..., N:], 0, x.shape[3] - 1)], axis=-1).astype('int32')
        q_lb = paddle.concat([q_lt[..., :N], q_rb[..., N:]], axis=-1)
        q_rt = paddle.concat([q_rb[..., :N], q_lt[..., N:]], axis=-1)
        # Bilinear interpolation weights
        g_lt = (1 + (q_lt[..., :N].astype(dtype) - p[..., :N])) * (1 + (q_lt[..., N:].astype(dtype) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].astype(dtype) - p[..., :N])) * (1 - (q_rb[..., N:].astype(dtype) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].astype(dtype) - p[..., :N])) * (1 - (q_lb[..., N:].astype(dtype) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].astype(dtype) - p[..., :N])) * (1 + (q_rt[..., N:].astype(dtype) - p[..., N:]))
        # Sampled image values
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # Weighted sum to get the output values
        x_offset = g_lt.unsqueeze(1) * x_q_lt + g_rb.unsqueeze(1) * x_q_rb + g_lb.unsqueeze(1) * x_q_lb + g_rt.unsqueeze(1) * x_q_rt
        if self.modulation:
            m = m.transpose([0, 2, 3, 1]).unsqueeze(1)
            m = paddle.concat([m for _ in range(x_offset.shape[1])], axis=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = paddle.meshgrid(paddle.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                       paddle.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = paddle.concat([p_n_x.flatten(), p_n_y.flatten()], axis=0)
        p_n = p_n.reshape([1, 2 * N, 1, 1]).astype(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = paddle.meshgrid(paddle.arange(1, h * self.stride + 1, self.stride),
                                       paddle.arange(1, w * self.stride + 1, self.stride))
        p_0_x = p_0_x.flatten().reshape([1, 1, h, w]).tile([1, N, 1, 1])
        p_0_y = p_0_y.flatten().reshape([1, 1, h, w]).tile([1, N, 1, 1])
        p_0 = paddle.concat([p_0_x, p_0_y], axis=1).astype(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.shape[1] // 2, offset.shape[2], offset.shape[3]
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.shape
        padded_w = x.shape[3]
        c = x.shape[1]

        # 扁平化x为（b, c, H*W）
        x = x.reshape([b, c, -1])

        # 计算索引
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x * w + offset_y
        index = index.reshape([b, -1])  # 展平为1D
        index = paddle.unsqueeze(index, axis=1).tile([1, c, 1])  # 复制c次以匹配通道维度

        # 确保所有索引都是int64
        index = index.astype('int64')
        batch_indices = paddle.arange(b, dtype='int64').reshape([-1, 1, 1]).tile([1, c, h * w * N])  # 生成批次索引
        channel_indices = paddle.arange(c, dtype='int64').reshape([1, -1, 1]).tile([b, 1, h * w * N])  # 生成通道索引

        # 使用gather_nd来收集数据
        gather_indices = paddle.stack([batch_indices, channel_indices, index], axis=-1)  # 组合索引
        x_offset = paddle.gather_nd(x, gather_indices)  # 使用gather_nd获取数据
        x_offset = x_offset.reshape([b, c, h, w, N])  # 重新塑形

        return x_offset



    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, N = x_offset.shape
        x_offset = paddle.concat([x_offset[..., s:s + ks].reshape([b, c, h, w * ks]) for s in range(0, N, ks)], axis=-1)
        x_offset = x_offset.reshape([b, c, h * ks, w * ks])
        return x_offset

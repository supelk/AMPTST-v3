import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 对应论文3.3.1节"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放点积注意力"""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)

        # 线性变换并分头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 输出投影
        output = self.w_o(attn_output)
        return output, attn_weights


class FrequencyDomainAttention(nn.Module):
    """频域注意力机制 - 对应论文3.3.2节"""

    def __init__(self, d_model, n_heads, seq_len, modes=64, dropout=0.1):
        super(FrequencyDomainAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.modes = min(modes, seq_len // 2)  # 频率模式数
        self.d_k = d_model // n_heads

        # 频域注意力
        self.freq_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 频域投影层（将复数转换为实数表示）
        self.freq_projection = nn.Linear(2, d_model)  # 实部和虚部

        # 逆投影层
        self.inv_freq_projection = nn.Linear(d_model, 2)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        频域注意力前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # 1. 傅里叶变换到频域
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')  # [batch_size, seq_len//2+1, d_model]

        # 2. 将复数分解为实部和虚部
        real_part = x_ft.real
        imag_part = x_ft.imag
        freq_input = torch.stack([real_part, imag_part], dim=-1)  # [batch_size, seq_len//2+1, d_model, 2]

        # 3. 频域投影
        freq_input_flat = freq_input.view(batch_size, -1, 2)  # [batch_size, (seq_len//2+1)*d_model, 2]
        projected_freq = self.freq_projection(freq_input_flat)  # [batch_size, (seq_len//2+1)*d_model, d_model]

        # 4. 频域自注意力
        freq_attn_out, freq_attn_weights = self.freq_attention(
            projected_freq, projected_freq, projected_freq
        )
        freq_attn_out = self.norm(projected_freq + self.dropout(freq_attn_out))

        # 5. 逆投影回复数表示
        freq_output = self.inv_freq_projection(freq_attn_out)  # [batch_size, (seq_len//2+1)*d_model, 2]
        freq_output = freq_output.view(batch_size, seq_len // 2 + 1, d_model, 2)

        # 6. 重构复数
        real_output = freq_output[..., 0]
        imag_output = freq_output[..., 1]
        out_ft = torch.complex(real_output, imag_output)

        # 7. 逆傅里叶变换回时域
        freq_out = torch.fft.irfft(out_ft, n=seq_len, dim=1, norm='ortho')

        return freq_out, freq_attn_weights


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合机制 - 对应论文3.3.3节"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # 时域到频域的交叉注意力
        self.time2freq_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 频域到时域的交叉注意力
        self.freq2time_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, time_feat, freq_feat):
        """
        交叉注意力融合

        参数:
            time_feat: 时域特征 [batch_size, seq_len, d_model]
            freq_feat: 频域特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = time_feat.shape

        # 1. 时域-频域交叉注意力
        time2freq_out, _ = self.time2freq_attention(time_feat, freq_feat, freq_feat)

        # 2. 频域-时域交叉注意力
        freq2time_out, _ = self.freq2time_attention(freq_feat, time_feat, time_feat)

        # 3. 门控融合
        combined = torch.cat([time2freq_out, freq2time_out], dim=-1)
        gate_weights = self.gate(combined)  # [batch_size, seq_len, d_model]

        fused = gate_weights * time2freq_out + (1 - gate_weights) * freq2time_out
        fused = self.norm1(fused)

        # 4. 前馈网络
        ff_out = self.feed_forward(fused)
        output = self.norm2(fused + self.dropout(ff_out))

        return output


class TimeFrequencyCrossAttentionLayer(nn.Module):
    """时域-频域交叉注意力层 - 对应论文3.3节完整实现"""

    def __init__(self, d_model, n_heads, seq_len, modes=64, dropout=0.1):
        super(TimeFrequencyCrossAttentionLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len

        # 时域自注意力路径
        self.time_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.time_norm = nn.LayerNorm(d_model)

        # 频域自注意力路径
        self.freq_attention = FrequencyDomainAttention(d_model, n_heads, seq_len, modes, dropout)
        self.freq_norm = nn.LayerNorm(d_model)

        # 交叉注意力融合
        self.cross_fusion = CrossAttentionFusion(d_model, n_heads, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        时域-频域交叉注意力层前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        # 时域路径
        time_attn_out, time_attn_weights = self.time_attention(x, x, x)
        time_out = self.time_norm(x + self.dropout(time_attn_out))

        # 频域路径
        freq_out, freq_attn_weights = self.freq_attention(x)
        freq_out = self.freq_norm(x + self.dropout(freq_out))

        # 交叉注意力融合
        fused_output = self.cross_fusion(time_out, freq_out)

        return fused_output, time_attn_weights, freq_attn_weights


class Model(nn.Module):
    """完整的频域-时域交叉注意力网络 - 对应论文第3章整体架构"""

    def __init__(self, configs, modes=64, dropout=0.1, use_auxiliary_loss=False):
        """
        频域-时域交叉注意力网络

        参数:
            input_dim: 输入特征维度N
            d_model: 模型隐藏层维度
            n_heads: 注意力头数
            seq_len: 输入序列长度L
            pred_len: 预测步长D
            num_layers: 网络层数
            modes: 频率模式数
            dropout: dropout率
            use_auxiliary_loss: 是否使用辅助损失（中继监督）
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.input_dim = configs.enc_in
        self.num_layers = configs.e_layers
        self.d_model = configs.d_model
        self.use_auxiliary_loss = use_auxiliary_loss

        # 输入嵌入层 - 对应论文3.2节
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(self.d_model, self.seq_len, dropout)

        # 时域-频域交叉注意力层堆叠
        self.layers = nn.ModuleList([
            TimeFrequencyCrossAttentionLayer(self.d_model, configs.n_heads, self.seq_len, modes, dropout)
            for _ in range(self.num_layers)
        ])

        # 中继监督输出层（如果使用辅助损失）
        if use_auxiliary_loss:
            self.auxiliary_outputs = nn.ModuleList([
                nn.Linear(self.d_model, 1) for _ in range(self.num_layers - 1)
            ])

        # 输出投影层 - 对应论文3.4节
        self.output_projection = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),  # 序列长度投影
            nn.Dropout(dropout)
        )

        # 最终输出层
        self.final_output = nn.Linear(self.d_model, 1)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,x_mark_enc=None,dec_inp=None, batch_y_mark=None):
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, input_dim]

        返回:
            如果use_auxiliary_loss为True: (main_output, auxiliary_outputs)
            否则: main_output
        """
        batch_size = x.shape[0]
        auxiliary_outputs = []

        # 输入投影和位置编码
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)

        # 通过多层时域-频域交叉注意力
        for i, layer in enumerate(self.layers):
            x, time_attn, freq_attn = layer(x)

            # 中继监督（除最后一层外）
            if self.use_auxiliary_loss and i < self.num_layers - 1:
                aux_out = self.auxiliary_outputs[i](x)  # [batch_size, seq_len, 1]
                aux_out = aux_out.transpose(1, 2)  # [batch_size, 1, seq_len]
                aux_out = self.output_projection(aux_out)  # [batch_size, 1, pred_len]
                auxiliary_outputs.append(aux_out.squeeze(1))  # [batch_size, pred_len]

        # 最终输出投影
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.output_projection(x)  # [batch_size, d_model, pred_len]
        main_output = x.transpose(1, 2)  # [batch_size, pred_len, d_model]

        # main_output = self.final_output(x).squeeze(-1)  # [batch_size, pred_len]

        if self.use_auxiliary_loss:
            return main_output, auxiliary_outputs
        else:
            return main_output


class PositionalEncoding(nn.Module):
    """位置编码 - 补充实现"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)





# 使用示例和测试代码
# if __name__ == "__main__":
#     # 参数设置
#     batch_size = 32
#     seq_len = 100
#     pred_len = 10
#     input_dim = 5
#     d_model = 128
#     n_heads = 8
#     num_layers = 3
#
#     # 创建模型
#     model = model(
#         input_dim=input_dim,
#         d_model=d_model,
#         n_heads=n_heads,
#         seq_len=seq_len,
#         pred_len=pred_len,
#         num_layers=num_layers,
#         use_auxiliary_loss=True
#     )
#
#     # 示例输入
#     x = torch.randn(batch_size, seq_len, input_dim)
#     targets = torch.randn(batch_size, pred_len)
#
#     # 前向传播
#     main_output, auxiliary_outputs = model(x)
#
#     print("=== FTCAN模型测试 ===")
#     print(f"输入形状: {x.shape}")
#     print(f"主输出形状: {main_output.shape}")
#     print(f"辅助输出数量: {len(auxiliary_outputs)}")
#     for i, aux_out in enumerate(auxiliary_outputs):
#         print(f"辅助输出 {i + 1} 形状: {aux_out.shape}")
#
#     # 损失计算
#     criterion = FTCANLoss(lambda_aux=0.3)
#     total_loss, main_loss, aux_loss = criterion((main_output, auxiliary_outputs), targets)
#
#     print(f"\n损失计算:")
#     print(f"总损失: {total_loss.item():.4f}")
#     print(f"主损失: {main_loss.item():.4f}")
#     print(f"辅助损失: {aux_loss.item():.4f}")
#     print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
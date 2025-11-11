import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import Embedding_forAMPTST
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.1):
        super().__init__()
        self.nf = nf
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # print(f"x.shape: {x.shape},nf: {self.nf}")
        x = self.linear(x)
        x = self.dropout(x)
        return x
class AMPTST(nn.Module):
    def __init__(self, configs):
        super(AMPTST, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.enc_in = configs.enc_in
        self.layers = configs.e_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_layers = configs.down_sampling_layers
        self.channel_independence = configs.channel_independence
        self.preprocess = series_decomp(configs.moving_avg)
        self.pf = configs.pf
        self.Multi_PTST_period = nn.ModuleList(
            [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=False), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(self.layers)
                    ],
                    norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
                )
                for _ in range(self.down_sampling_layers + 1)
            ]
        )
        self.Multi_PTST_frequency = nn.ModuleList(
            [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=False), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(self.layers)
                    ],
                    norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
                )
                for _ in range(self.down_sampling_layers + 1)
            ]
        )

    def forward(self, x_list):  # [B,T,dm]
        res_list = []
        for i,x in enumerate(x_list):
            # print(f"input of AMPSTS.block {x.shape}")
            B, T, N = x.size()
            period_list, period_weight = FFT_for_Period(x, self.k)  # 自适应周期
            res = []
            # seasons =[]
            # trends =[]
            for k in range(self.k):
                period = period_list[k]
                # padding
                if T % period != 0:
                    length = ((T // period) + 1) * period
                    padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
                    block_in = torch.cat([x, padding], dim=1)
                else:
                    length = T
                    block_in = x
                # reshape
                block_in = block_in.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()  # [B,dm,f,t]

                block_in_p = block_in.permute(0, 2, 3, 1).contiguous()  # [B,f,t,dm], seasons
                block_in_p = torch.reshape(block_in_p, (B * block_in_p.shape[1], -1, N))  # [B*f,t,dm]
                block_out_p, _ = self.Multi_PTST_period[i](block_in_p)
                block_season = block_out_p.reshape(B, -1, N)  # [B,f,t,dm] > [B,T,dm]
                block_in_f = block_in.permute(0, 3, 2, 1).contiguous()  # [B,t,f,dm], trends
                block_in_f = torch.reshape(block_in_f, (
                block_in_f.shape[0] * block_in_f.shape[1], block_in_f.shape[2], block_in_f.shape[3]))  # [B*t,f,dm]
                block_out_f, _ = self.Multi_PTST_frequency[i](block_in_f)
                block_trend = block_out_f.reshape(B, -1, N)  # [B,f,t,dm] > [B,T,dm]
                out = block_season + block_trend

                # seasons.append(block_season)
                # trends.append(block_trend)
                res.append(out[:,:T,:])
            # seasons = torch.cat(seasons, dim=-1)
            # trends = torch.cat(trends, dim=-1)
            # adaptive aggregation
            res = torch.stack(res, dim=-1)
            period_weight = F.softmax(period_weight, dim=1)
            period_weight = period_weight.unsqueeze(
                1).unsqueeze(1).repeat(1, T, N, 1)
            res = torch.sum(res * period_weight, -1)
            # residual connection
            res = res + x
            res_list.append(res)
        return res_list

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layers = configs.e_layers
        self.enc_in = configs.enc_in
        self.channel_independence = configs.channel_independence
        if self.channel_independence:
            self.enc_embedding = Embedding_forAMPTST(1, configs.d_model,configs.dropout)
        else:
            self.enc_embedding = Embedding_forAMPTST(configs.enc_in, configs.d_model,configs.dropout)
        self.model = AMPTST(configs)
        # self.model = nn.ModuleList(
        #     [AMPTST(configs) for _ in range(self.layers)]
        # )

        self.head_nf_list = []
        for i in range(configs.down_sampling_layers + 1):
            self.head_nf_list.append(
                int(configs.seq_len // (configs.down_sampling_window ** i))
            )
            # print("hf_list:")
            # print(int(((configs.seq_len // (configs.down_sampling_window ** i)) - patch_len) / stride + 2))
        self.head = nn.ModuleList(
            [
                FlattenHead(self.enc_in, hf, self.pred_len, head_dropout=configs.dropout) for hf in self.head_nf_list
            ]
        )
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )
        # if self.channel_independence:
        #     self.projection = nn.Linear(configs.d_model, 1)
        # else:
        #     self.projection = nn.Linear(configs.d_model, configs.enc_in)
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc
    def forward(self, x_enc, x_mark_enc,dec_inp, batch_y_mark):
        # B, T, N = x_enc.size()
        # if self.channel_independence:
        #     x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # x_enc = self.enc_embedding(x_enc)  # [B*N,T,dm] or [B,T,dm]
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                # if self.channel_independence:
                #     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                #     x_mark = x_mark.repeat(N, 1, 1)
                # x = self.enc_embedding(x)  # [B*N,T,dm] or [B,T,dm]
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):    # 对尺度序列进行投影
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                # if self.channel_independence:
                #     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                # x = self.enc_embedding(x)  # [B*N,T,dm] or [B,T,dm]   在特征维度上进行嵌入
                x_list.append(x)  # [B*N,T,dm] or [B,T,dm]
        # print(f"len of x_list: {len(x_list)}")
        out_list = self.model(x_list)  # [B*N,T,dm] or [B,T,dm]
        oc_list = []
        # print(f"len of out_list: {len(out_list)}")
        # for i,x in enumerate(out_list):
        #     print(f"{i} of outlist:shape {x.shape}")
        for i,x in enumerate(out_list):
            x = x.permute(0, 2, 1).contiguous()
            # print(f"{i} of outlist")
            y = self.head[i](x)  # [B*N,dm,target_w] or [B,dm,target_w]
            y = y.permute(0, 2, 1).contiguous()
            # oc = self.projection(y) # [B*N,target_w,1] or [B,target_w,N]
            # if self.channel_independence:
            #     oc = torch.reshape(oc,(B,self.pred_len,self.enc_in))
            oc_list.append(y)
        output = torch.stack(oc_list,dim=-1).sum(-1)
        output = self.normalize_layers[0](output,'denorm')
        return output
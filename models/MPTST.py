import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_MutiPTST
from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x




class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        padding = stride
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.layer = configs.e_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_layers = configs.down_sampling_layers
        self.channel_independence = configs.channel_independence
        self.preprocess = series_decomp(configs.moving_avg)
        self.patch_embedding = PatchEmbedding_MutiPTST(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.Multi_PTST = nn.ModuleList(
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
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
                )
                for _ in range(self.down_sampling_layers+1)
            ]
        )
        # Prediction Head
        self.head_nf_list = []
        for i in range(configs.down_sampling_layers+1):
            self.head_nf_list.append(
                configs.d_model * int(((configs.seq_len // (configs.down_sampling_window ** i)) - patch_len) / stride + 2)
            )
            # print("hf_list:")
            # print(int(((configs.seq_len // (configs.down_sampling_window ** i)) - patch_len) / stride + 2))
        self.head = nn.ModuleList(
            [
                FlattenHead(self.enc_in,hf,self.pred_len,head_dropout=configs.dropout) for hf in self.head_nf_list
            ]
        )


        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
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
    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def forward(self, x_enc, x_mark_enc,dec_inp, batch_y_mark):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc) # B,T,C
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        # [B,T,N] > [B,n,pl,pn] > [B*n,pn,dm]
        enc_out_list = []
        patch_num_list = []
        x_list = self.pre_enc(x_list)   # channelindpendence or decomp
        for i, x in zip(range(len(x_list[0])), x_list[0]):
            enc_out = x.permute(0, 2, 1)  # [B*N,1,T]
            # print("the x_{} patch_embedding".format(i))
            enc_out = self.patch_embedding(enc_out) # [B*N,1,T]>[B*n,pn,pl]>[B*n,pn,dm]
            enc_out_list.append(enc_out)
        # PatchTST
        MultiPTST_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # print(i,len(enc_out_list))
            t = self.Multi_PTST[i](enc_out)  # [Bs*n, pn, dm]
            tst_bb_out, _ = t
            tst_bb_out = torch.reshape(tst_bb_out, (-1, self.enc_in,tst_bb_out.shape[-2], tst_bb_out.shape[-1]))
            MultiPTST_out_list.append(tst_bb_out)
        list = []
        for i,x in enumerate(MultiPTST_out_list):
            # print(i,x.shape[-2])
            y = self.head[i](x)
            y = y.permute(0, 2, 1)
            list.append(y)
        dec_out = torch.stack(list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out





import torch
import torch.nn as nn
"""
Linear
"""
class Patch(nn.Module):
    def __init__(self, patch_len, stride, embedding, padding, d_model):
        super(Patch, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.embedding = embedding
        if embedding:
            self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs,n,pn,p_len]
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        if self.embedding:
            x = self.value_embedding(x)
        return x

class Patch_block(nn.Module):
    # channel和混合patch并行特征提取
    def __init__(self, d_model, seq_len, pred_len, channel, patch_len=16, stride=16, embedding=False):
        super(Patch_block, self).__init__()


        p_model = patch_len * 2
        padding = stride
        patch_num = int((seq_len - patch_len) / stride + 2)
        n_model = patch_num * 2
        # print("patch_num c:",patch_num)
        self.patch = Patch(patch_len, stride, embedding, padding, d_model=d_model)
        # self.inter_block = nn.Sequential(
        #     nn.Linear(in_features=patch_len, out_features=p_model),
        #     nn.GELU(),
        #     nn.Linear(in_features=p_model, out_features=patch_len),
        # )
        self.inter_block = nn.Linear(pred_len, pred_len)
        self.intar_block = nn.Sequential(
            nn.Linear(in_features=patch_num, out_features=n_model),
            nn.GELU(),
            nn.Linear(in_features=n_model, out_features=patch_num),
        )
        self.channel_block = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.GELU(),
            nn.Linear(d_model, channel),
        )
        self.head = nn.Linear(patch_len*patch_num,seq_len)
        self.flatten = nn.Flatten(start_dim=-2)
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.Dropout(0.1)
        )
        # self.projector = nn.Sequential(
        #     nn.Linear(in_features=configs.seq_len, out_features=configs.s_model),
        #     nn.GELU(),
        #     nn.Linear(in_features=configs.s_model, out_features=configs.pred_len),
        # )
    def forward(self, x):
        x_c = self.channel_block(x.permute(0,2,1)).permute(0,2,1) + x  # shape remain
        x_p = self.patch(x)  # [bs,n,pn,p_len]
        b,n,pn,p_len = x_p.shape
        # print("patch_num actual:",pn)
        x_p_1 = self.intar_block(x_p.permute(0,1,3,2)).permute(0,1,3,2)
        # x_p_1 = self.inter_block(x_p_1+x_p)
        x_p_2 = self.inter_block(x_p)
        # x_p_2 = self.intar_block((x_p_2+x_p).permute(0,1,3,2))
        x_p_out = self.flatten(x_p_1)+self.flatten(x_p_2)
        y = x_c + self.head(x_p_out)
        return self.projection(y)

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.m1_type = configs.m1_type
        if self.m1_type == 0:
            self.block = nn.Linear(configs.seq_len, configs.pred_len)
            self.block.weight = nn.Parameter(
                (1 / configs.seq_len) * torch.ones([configs.pred_len, configs.seq_len]))
        elif self.m1_type == 1:
            self.block = nn.Sequential(
                nn.Linear(in_features=configs.seq_len, out_features=configs.d_model),
                nn.GELU(),
                nn.Linear(in_features=configs.d_model, out_features=configs.pred_len),
            )
        elif configs.m1_type == 2:
            self.block = Patch_block(configs.d_model,configs.seq_len,configs.pred_len,configs.enc_in)


    def forward(self, x,a,b,c):
        x = x.permute(0, 2, 1)  # [b,n,L]

        return self.block(x).permute(0, 2, 1)

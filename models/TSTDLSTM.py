import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# class LSTM(nn.Module):
#     def __init__(
#             self,
#             input_size,  # 输入特征维度（对应 features:12）
#             lstm_hidden,  # LSTM隐藏层维度
#             output_len,  # 输出时间步（对应 output_len:24）
#             num_layers
#     ):
#         super().__init__()
#         # LSTM层：提取时序特征
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=lstm_hidden,
#             batch_first=True,
#             num_layers=num_layers,
#         )

#         # 卷积层：将时间维度从168压缩到24，同时输出1个特征
#         self.conv = nn.Conv1d(
#             in_channels=lstm_hidden,
#             out_channels,  # 输出特征数（对应 features:1）
#             kernel_size=168 - output_len + 1  # 计算卷积核大小：168-24+1=145
#         )

#         # 激活函数（可选）
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # 输入形状: (bs, 168, 12)

#         # LSTM处理
#         lstm_out, _ = self.lstm(x)  # 输出形状: (bs, 168, lstm_hidden)

#         # 调整维度以适应Conv1d输入格式 (bs, channels, length)
#         lstm_out = lstm_out.permute(0, 2, 1)  # 形状: (bs, lstm_hidden, 168)

#         # 卷积压缩时间维度
#         conv_out = self.conv(lstm_out)  # 输出形状: (bs, 1, 24)

#         # 调整维度为 (bs, 24, 1)
#         output = conv_out.permute(0, 2, 1)

#         return output  # 形状: (bs, 24, 1)

class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.decompsition = configs.decompsition
        # Decompsition Kernel Size
        kernel_size = 25
        if configs.decompsition:
            self.decompsition = series_decomp(kernel_size)
            self.individual = configs.individual
            self.channels = configs.enc_in
            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()

                for i in range(self.channels):
                    self.Linear_Seasonal.append(nn.LSTM(input_size=1,hidden_size=self.pred_len,num_layers=7,batch_first=True))
                    self.Linear_Trend.append(nn.LSTM(input_size=1,hidden_size=self.pred_len,num_layers=7,batch_first=True))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            else:
                self.Linear_Seasonal = nn.LSTM(input_size=self.channels,hidden_size=self.pred_len,num_layers=7,batch_first=True)
                self.Linear_Trend = nn.LSTM(input_size=self.channels,hidden_size=self.pred_len,num_layers=7,batch_first=True)
                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.model = nn.LSTM(input_size=self.channels,hidden_size=self.pred_len,num_layers=7,batch_first=True)

        
    def forward(self, x, x_mask=None,dec_input=None,y_mark=None):
        # x: [Batch, Input length, Channel]
        if self.decompsition:
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            if self.individual:
                seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                              dtype=seasonal_init.dtype).to(seasonal_init.device)
                trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                           dtype=trend_init.dtype).to(trend_init.device)
                for i in range(self.channels):
                    seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                    trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            else:
                seasonal_output = self.Linear_Seasonal(seasonal_init)
                trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
        else:
            # x = x.permute(0,2,1)
            x,_ = self.model(x)
        
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
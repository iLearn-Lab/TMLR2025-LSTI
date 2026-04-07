import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

    def RevIN_Cal(self, x_in, mask_in):
        # x_in:[L,D]
        if torch.all(mask_in == 0):     # 该区间无效
            return torch.zeros((1,1), device=x_in.device), torch.zeros((1,1), device=x_in.device)
        cnt = torch.sum(mask_in == 1, dim=0)
        cnt[cnt==0] = 1
        means_o = (torch.sum(x_in, dim=0) / cnt).unsqueeze(0).detach()
        stdev_o = (torch.sqrt(torch.sum((x_in - means_o) * (x_in - means_o), dim=0) /
                        cnt + 1e-5)).unsqueeze(0).detach()
        return means_o, stdev_o

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, x_before, x_after):   
        
        mean_list = []
        stdev_list = []
        for b in range(mask.shape[0]):      # 每个batch单独处理
            zero_rate = torch.sum(mask[b]).item() / (mask.shape[1] * mask.shape[2])
            if zero_rate >= 1:          # 该区间全是缺失值，应用过去和未来的的mean和stdev
                # 计算过去的RevIN
                x_in = x_before[b]
                mask_in = (~torch.isnan(x_in)).to(torch.int)
                x_in = x_in.masked_fill(mask_in == 0, 0)
                l_mean, l_stdev = self.RevIN_Cal(x_in, mask_in)

                # 计算未来的RevIN
                x_in = x_after[b]
                mask_in = (~torch.isnan(x_in)).to(torch.int)
                x_in = x_in.masked_fill(mask_in == 0, 0)
                r_mean, r_stdev = self.RevIN_Cal(x_in, mask_in)

                # 合并两个RevIN
                print(l_mean.shape)
                if l_mean.shape[1] == 1:
                    if r_mean.shape[1] == 1:
                        print("!!!!!!")
                        means_o = torch.zeros((1, x_in.shape[1]), device=x_enc.device)
                        stdev_o = torch.ones((1, x_in.shape[1]), device=x_enc.device)
                    else:
                        print("!!!!!!")
                        means_o = r_mean
                        stdev_o = r_stdev
                else:
                    if r_mean.shape[1] == 1:
                        print("!!!!!!")
                        means_o = l_mean
                        stdev_o = l_stdev
                    else:
                        means_o = (l_mean + r_mean) / 2
                        stdev_o = (l_stdev + r_stdev) / 2
                
                mean_list.append(means_o)
                stdev_list.append(stdev_o)

            else:               # 该区间存在可用数据，用自己的RevIN
                x_in = x_enc[b]
                mask_in = mask[b]
                means_o, stdev_o = self.RevIN_Cal(x_in, mask_in)
                mean_list.append(means_o)
                stdev_list.append(stdev_o)
                    

        # Normalization from Non-stationary Transformer
        means = torch.stack(mean_list)
        stdev = torch.stack(stdev_list)
        x_enc = x_enc - means
        x_enc /= stdev
        # x_enc = x_enc.masked_fill(mask == 0, 0)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_before=None, x_after=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, x_before, x_after)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

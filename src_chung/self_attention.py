import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import csv


# class GELU(nn.Module):
#    def forward(self, input):
#        return F.gelu(input)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)

        #        self.gelu = GELU()
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        # print('q_shape: ',q.shape)

        d_k = self.att_size
        d_v = self.att_size
        batch_size_q = q.size(0)

        q = self.linear_q(q).view(batch_size_q, -1, self.num_heads, d_k)
        # print('q_shape: ',q.shape)
        batch_size_k=k.size(0)
        k = self.linear_k(k)
        # print('k_shape: ',k.shape)
        k=k.view(batch_size_k, -1, self.num_heads, d_k)
        # print('k_shape ver2: ',k.shape)
        v = self.linear_v(v).view(batch_size_k, -1, self.num_heads, d_v)
        # print('v_shape: ',v.shape)

        q = q.transpose(0, 2)  # [q_len, h, b_q, d_q]
        v = v.transpose(0, 2)  # [v_len, h, b_kv, d_v]
        k = k.transpose(0, 2).transpose(2, 3)  # [k_len, h, d_k, b_kv]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [q_len(k_len), h, b_q, b_kv]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn] [q_len(k_len), h, b_q, d_v]
        
        x = x.transpose(0, 2).transpose(1,2).contiguous()  # [b, q_len, h, attn] [b_q, q_len(k_len),h, d_v]
        x = x.view(batch_size_q, -1, self.num_heads * d_v)
        x = self.output_layer(x)

        x=x.squeeze(1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size,ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        # print('x_shape: ',x.shape)
        y = self.self_attention_norm(x)
        # print('y_shape: ',y.shape)
        kv = self.self_attention_norm(kv)
        # print('kv_shape: ',kv.shape)
        y = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x
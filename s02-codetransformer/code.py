# simple attention
# import torch

# inputs = torch.tensor(
#     [
#         [0.43, 0.15, 0.89], # Your (x^1)
#         [0.55, 0.87, 0.66], # journey (x^2)
#         [0.57, 0.85, 0.64], # starts (x^3)
#         [0.22, 0.58, 0.33], # with (x^4)
#         [0.77, 0.25, 0.10], # one (x^5)
#         [0.05, 0.80, 0.55]] # step (x^6
# )
# query = inputs[1]
# print(inputs.shape)
# attn_scores_2 = torch.empty(inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query)

# print(attn_scores_2)

# #归一化
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

# attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

# query = inputs[1]
# context_vec_2 = torch.zeros(query.shape)
# for i, x_i in enumerate(inputs):
#     context_vec_2 += attn_weights_2[i] * x_i
# print(context_vec_2)

# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i][j] = torch.dot(x_i, x_j)

# print(attn_scores)

# attn_scores = inputs @ inputs.T # 矩阵乘法
# attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

# all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)
# ---------------------------------------------------------------

# # 带有权重的self-attention
# import torch
# inputs = torch.tensor(
#     [
#         [0.43, 0.15, 0.89], # Your (x^1)
#         [0.55, 0.87, 0.66], # journey (x^2)
#         [0.57, 0.85, 0.64], # starts (x^3)
#         [0.22, 0.58, 0.33], # with (x^4)
#         [0.77, 0.25, 0.10], # one (x^5)
#         [0.05, 0.80, 0.55]] # step (x^6
# )

# x_2 = inputs[1]
# d_in = inputs.shape[1]
# d_out = 2

# torch.manual_seed(123)
# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# query_2 = x_2 @ W_query
# key_2 = x_2 @ W_key
# value_2 = x_2 @ W_value
# print(query_2)

# keys = inputs @ W_key
# values = inputs @ W_value
# print("keys.shape:", keys.shape)
# print("values.shape", values.shape)

# keys_2  = keys[1]
# attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)

# attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

# d_k = keys.shape[-1]
# attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)
# ---------------------------------------------------------------

# # 抽象为class
# import torch.nn as nn
# import torch
# class SelfAttention_v1(nn.Module):
#     def __init__(self, d_in, d_out):
#         super().__init__()
#         self.W_query = nn.Parameter(torch.rand(d_in, d_out))
#         self.W_key = nn.Parameter(torch.rand(d_in, d_out))
#         self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
#     def forward(self, x):
#         keys = x @ self.W_key
#         queries = x @ self.W_query
#         values = x @ self.W_value

#         attn_scores = queries @ keys.T
#         attn_weights = torch.softmax(
#             attn_scores / keys.shape[-1]**0.5, dim=-1
#         )
#         context_vec = attn_weights @ values

#         return context_vec

# d_in = 3
# d_out = 2
# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)
# inputs = torch.tensor(
#     [
#         [0.43, 0.15, 0.89], # Your (x^1)
#         [0.55, 0.87, 0.66], # journey (x^2)
#         [0.57, 0.85, 0.64], # starts (x^3)
#         [0.22, 0.58, 0.33], # with (x^4)
#         [0.77, 0.25, 0.10], # one (x^5)
#         [0.05, 0.80, 0.55]] # step (x^6
# )
# print(sa_v1(inputs))
# class SelfAttention_v2(nn.Module):
#     def __init__(self, d_in, d_out, qkv_bias=False):
#         super().__init__()
#         self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
#     def forward(self, x):
#         keys = self.W_key(x)
#         queries = self.W_query(x)
#         values = self.W_value(x)
#         attn_scores = queries @ keys.T
#         attn_weights = torch.softmax(
#             attn_scores / keys.shape[-1]**0.5, dim=-1
#         )

#         context_vec = attn_weights @ values

#         return context_vec

# torch.manual_seed(789)
# sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))

# with torch.no_grad():
#     sa_v1.W_query.copy_(sa_v2.W_query.weight.T)
#     sa_v1.W_key.copy_(sa_v2.W_key.weight.T)
#     sa_v1.W_value.copy_(sa_v2.W_value.weight.T)
# print(torch.allclose(sa_v1(inputs), sa_v2(inputs)))
# print(sa_v1(inputs))
# print(sa_v2(inputs))
# ---------------------------------------------------------------

# # 掩码注意力
# import torch
# import torch.nn as nn

# class SelfAttention_v2(nn.Module):
#     def __init__(self, d_in, d_out, qkv_bias=False):
#         super().__init__()
#         self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
#     def forward(self, x):
#         keys = self.W_key(x)
#         queries = self.W_query(x)
#         values = self.W_value(x)
#         attn_scores = queries @ keys.T
#         attn_weights = torch.softmax(
#             attn_scores / keys.shape[-1]**0.5, dim=-1
#         )

#         context_vec = attn_weights @ values

#         return context_vec

# d_in = 3
# d_out = 2
# torch.manual_seed(789)
# sa_v2 = SelfAttention_v2(d_in, d_out)
# inputs = torch.tensor(
#     [
#         [0.43, 0.15, 0.89], # Your (x^1)
#         [0.55, 0.87, 0.66], # journey (x^2)
#         [0.57, 0.85, 0.64], # starts (x^3)
#         [0.22, 0.58, 0.33], # with (x^4)
#         [0.77, 0.25, 0.10], # one (x^5)
#         [0.05, 0.80, 0.55]] # step (x^6
# )
# queries = sa_v2.W_query(inputs)
# keys = sa_v2.W_key(inputs)
# attn_scores = queries @ keys.T
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# context_length = attn_scores.shape[0]
# mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

# masked_simple = attn_weights * mask_simple
# print(masked_simple)

# row_sums = masked_simple.sum(dim=-1, keepdim=True)
# print(row_sums)

# masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

# # better mask attention, 将掩码的部分设置为-inf, 这样在算softmax时候,概率就为零
# mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

# attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# # droup mask attention
# torch.manual_seed(123)
# dropout = torch.nn.Dropout(0.5)
# example = torch.ones(6, 6)
# print(dropout(example))

# torch.manual_seed(123)
# print(dropout(attn_weights))
# ---------------------------------------------------------------

# 抽象 drop-mask attention
import torch
import torch.nn as nn

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6
)
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

# 输入: [batch, num_tokens, d_in] 输出: [batch, num_tokens, d_out] 为所有的token embedding生成对应的context embedding
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # 对注意力得分使用掩码
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], 
            -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        # drop 防止过拟合
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
d_in = 3
d_out = 2
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
print(context_vec.shape)
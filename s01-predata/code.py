# # 加载数据
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# print("Total number of charater:", len(raw_text))
# print(raw_text[:99])
# ---------------------------------------------------------------

# # 简单的对文本进行划分，不一定非得要去掉空格，例如python代码对于空格是非常敏感的
# import re
# text = "Hello, world. This, is a test."
# result = re.split(r'([,.]|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# print(result)
# ---------------------------------------------------------------

# # 对数据进行分词处理，text ---> token
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# import re
# import token
# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

# # 将token转换成tokenId
# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)
# print(vocab_size)

# # 创建词典
# vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

# # 构建分词器类
# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text) 
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
    
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # 去掉标签前的空格
#         return text

# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

# # 词典之外的词，会报错
# # text = "Hello, do you like tea?"
# # print(tokenizer.encode(text))

# # 扩展词汇表（增加额外的token， 未知单词token和文本边界的token）
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item if item in self.str_to_int
#                         else "<|unk|>" for item in preprocessed if item.strip()]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
    
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids]) # 这里会给标点符号前面添加上空格
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)

#         return text

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))

# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))
# ---------------------------------------------------------------

# # BPE，字节对编码
# from importlib.metadata import version
# import tiktoken
# print("tiktoken version:", version("tiktoken"))

# tokenizer = tiktoken.get_encoding("gpt2")
# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#     "of someunknowPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)

# # exercise 2.1
# text = "Akwirw ier"
# ids = tokenizer.encode(text)
# print(tokenizer.encode(text))
# for id in ids:
#     print(tokenizer.decode([id]))
# print(tokenizer.decode(ids))
# ---------------------------------------------------------------

# # 构建训练样本对
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# import token

# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

# enc_sample = enc_text[50:]
# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y:      {y}")

# for i in range(1, context_size + 1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))
# ---------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk)) # 将python list 转换为 tensor
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()

# dataloader = create_dataloader_v1(
#     raw_txt, batch_size=1, max_length=4, stride=1, shuffle=False
# )
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)

# dataloader = create_dataloader_v1(
#     raw_txt, batch_size=8, max_length=4, stride=4,
#     shuffle=False
# )

# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
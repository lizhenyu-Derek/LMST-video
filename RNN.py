import torch
import torch.nn as nn
from torchtext.vocab import Vectors

# 词嵌入文件路径（根据您下载的文件路径进行更改）
glove_path = 'glove.6B.100d.txt'

# 定义词嵌入维度
embedding_dim = 100

# 创建一个加载GloVe词嵌入的词汇表
vectors = Vectors(name=glove_path, cache="./.vector_cache")
vocab = vectors.stoi

print(vectors.vectors.shape)

# # 创建一个嵌入层，并加载预训练的GloVe词嵌入权重
# embedding = nn.Embedding.from_pretrained(vectors, padding_idx=0)
#
# # 定义输入文本序列（示例）
input_sequence = ["hello", "world", "example"]
#
# # 将输入序列转换为词嵌入
# embedded_input = embedding(torch.LongTensor([vocab[word] for word in input_sequence]))
#
# # 输出embedded_input包含了输入序列中每个单词的GloVe词嵌入
for word in input_sequence:
    print(vectors.vectors[vocab[word]])
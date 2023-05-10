import os
import random
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from torchtext import transforms as T

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from transformers import BertModel
from transformers.optimization import get_linear_schedule_with_warmup


# 读入acimbd库
def read_acimbd(is_train, path='./aclImdb_v1/aclImdb'):
    review_list = []
    label_list = []

    # 定义一个标记器去分割英语单词
    tokenizer = get_tokenizer('basic_english')

    # 定义路径
    if is_train:
        valid_file = os.path.join(path, 'train')
    else:
        valid_file = os.path.join(path, 'test')

    valid_file_pos = os.path.join(valid_file, 'pos')
    valid_file_neg = os.path.join(valid_file, 'neg')

    # 遍历所有的positive文件，用tokenizer进行分割
    for filename in os.listdir(valid_file_pos):
        with open(os.path.join(valid_file_pos, filename), 'r', encoding='utf-8') as file_content:
            review_list.append(tokenizer(file_content.read()))
            label_list.append(1)

    # 遍历所有的negative文件
    for filename in os.listdir(valid_file_neg):
        with open(os.path.join(valid_file_neg, filename), 'r', encoding='utf-8') as file_content:
            review_list.append(tokenizer(file_content.read()))
            label_list.append(0)

    return review_list, label_list


# 将review_list与label_list经过vocab的index转化后用TensorDataset打包
def build_dataset(review_list, label_list, _vocab, max_len=256):
    # 建立一个词表转化,vocab里存储词汇与它的唯一标签，利用VocabTransform将词汇转化为对应的数字
    # 利用Truncate将所有的句子的最长长度限制在了max_len
    # 利用ToTensor将所有的句子按照此时最长的句子进行填充为张量，填充的内容为'[PAD]'对应的数字
    # 利用PadTransform将ToTensor里所有的句子长度均填充为max_len
    # 与第二次实验不同的点是要将句子的开头与结尾加入开始[CLS]以及结束[SEP],需要按照bert模型的vocab创建
    seq_to_tensor = T.Sequential(
        T.VocabTransform(vocab=_vocab),
        T.Truncate(max_seq_len=max_len-2),
        T.AddToken(token=_vocab['[CLS]'], begin=True),
        T.AddToken(token=_vocab['[SEP]'], begin=False),
        T.ToTensor(padding_value=_vocab['[PAD]']),
        T.PadTransform(max_length=max_len, pad_value=_vocab['[PAD]'])
    )
    dataset = TensorDataset(seq_to_tensor(review_list), torch.tensor(label_list))
    return dataset


# 从bert模型中读取字典
def load_acimdb(trained_bert_vocab):
    review_train_list, label_train_list = read_acimbd(is_train=True)
    review_test_list, label_test_list = read_acimbd(is_train=False)

    with open(trained_bert_vocab, 'r', encoding='utf-8') as vocab_file:
        token_list = []
        for token in vocab_file.readlines():
            token_list.append(token.strip())
    _vocab = vocab(OrderedDict([(token, 1) for token in token_list]))
    # 设置未登录词的索引
    _vocab.set_default_index(_vocab['[UNK]'])

    dataset_train = build_dataset(review_train_list, label_train_list, _vocab=_vocab)
    dataset_test = build_dataset(review_test_list, label_test_list, _vocab=_vocab)
    return dataset_train, dataset_test, _vocab


# 设置随机数种子
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MyBERT(nn.Module):
    def __init__(self, vocab, bert_model):
        super().__init__()
        self.vocab = vocab
        self.bert = BertModel.from_pretrained(bert_model)

        self.bert_config = self.bert.config
        out_dim = self.bert_config.hidden_size
        self.classifier = nn.Linear(out_dim, 2)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 2)
        )

    def forward(self, input_ids):
        attention_mask = (input_ids != self.vocab['[PAD]']).long().float()
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        result = self.classifier(output.pooler_output)
        return result


# 设置模型保存路径
def set_model_save_path(learning_rate, epoch_num, batch_size, bert_model):
    path = './src/bert_result/bert_' + 'rate' + str(learning_rate) + '_epoch' + str(epoch_num) + \
           '_batch' + str(batch_size) + '_' + bert_model + '.pth'
    return path


# 设置图片保存路径
def set_pic_save_path(learning_rate, epoch_num, batch_size, bert_model):

    path = './src/draw_result/bert_' + 'rate' + str(learning_rate) + '_epoch' + str(epoch_num) + \
           '_batch' + str(batch_size) + '_' + bert_model + '.png'
    return path


init_seeds(0)
# 设置跑的平台是CPU还是GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 18
learning_rate = 5e-5
epoch_num = 4

# 读入base_uncased的bert模型的vocab表，也可以读入别的
bert_model = "bert-base-uncased"
if bert_model == "bert-base-uncased":
    bert_model_vocab = "./src/trained_bert_vocab/vocab_base_uncased.txt"
elif bert_model == "bert-base-cased":
    bert_model_vocab = "./src/trained_bert_vocab/vocab_base_cased.txt"
elif bert_model == "bert-base-chinese":
    bert_model_vocab = "./src/trained_bert_vocab/vocab_base_chinese.txt"

dataset_train, dataset_test, _vocab = load_acimdb(bert_model_vocab)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

model = MyBERT(vocab=_vocab, bert_model=bert_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 在bert_base_uncased模型中作者提到使用了linear的scheduler
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0,
                                            num_training_steps=epoch_num * len(dataloader_train))
loss_func = nn.CrossEntropyLoss()

# 设置模型存储地址以及画图存储地址
path_model = set_model_save_path(learning_rate, epoch_num, batch_size, bert_model)
path_pic = set_pic_save_path(learning_rate, epoch_num, batch_size, bert_model)

# 训练模型
use_trained_model = False
if use_trained_model:
    model.load_state_dict(torch.load(path_model))
else:
    train_loss_per_epoch = []
    for epoch in range(epoch_num):
        print(f'epoch {epoch + 1}:')
        total_loss = 0
        batch_idx = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader_train):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predict_y = model(batch_x)
            loss = loss_func(predict_y, batch_y)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 5 == 0:
                print(f"loss on batch_idx {batch_idx} is: {loss:.6f}")

        total_loss /= (batch_idx + 1)
        train_loss_per_epoch.append(total_loss.item())
        print(f"loss on train set: {total_loss:.6f}\n")

    torch.save(model.state_dict(), path_model)

    # 画图展示训练的epoch过程中的loss变化
    draw_x = list(range(1, len(train_loss_per_epoch) + 1))
    plt.plot(draw_x, train_loss_per_epoch)
    plt.title('loss change in training set')
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.savefig(path_pic)
    plt.show()

# 在测试集上的结果
acc = 0
for batch_x, batch_y in dataloader_test:
    with torch.no_grad():
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predict_y = model(batch_x)
        acc += (torch.argmax(predict_y, dim=1) == batch_y).sum().item()
print(f"accuracy: {acc / len(dataset_test):.6f}")

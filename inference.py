import logging
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import MyDataset
from model.transformer import Transformer
from scheduler import Scheduler
from config import Config
import tqdm
from datetime import datetime
from model.decoder import TransformerDecoder

from model.encoder import TransformerEncoder
np.random.seed(37)
torch.manual_seed(73)

cfg = Config()
cfg.DEVICE = "cpu"

data_link = "data/fra.txt"
with open(data_link, "r") as f:
    data = f.read()
sen_list = data.split("\n")
print(len(sen_list))
dataset_dict = {"en": [], "fr": []}
for sen in sen_list:
    temp = sen.split("\t")
    if len(temp) == 3:
        dataset_dict["en"].append(temp[0])
        dataset_dict["fr"].append(temp[1])
dataset_df = pd.DataFrame(dataset_dict)

len_dts = len(dataset_df)
indices = np.arange(len_dts)
np.random.shuffle(indices)
len_using = 10
part_of_indices = indices[:len_using]
valid_fraction = 0.1
indices_train = part_of_indices[int(len_using*valid_fraction):]
indices_valid = part_of_indices[:int(len_using*valid_fraction)]
dataset_train_df = dataset_df.iloc[list(indices_train)].reset_index().drop('index', axis=1)
dataset_valid_df = dataset_df.iloc[list(indices_valid)].reset_index().drop('index', axis=1)
print("Len dts: ", len_dts)
print("Len train: ", len(dataset_train_df))
print("Len valid: ", len(dataset_valid_df))


train_dts = MyDataset(cfg, dataset_train_df)
valid_dts = MyDataset(cfg, dataset_train_df, dataset_valid_df)
cfg.ENG_VOCAB_SIZE = train_dts.vocab_src.vocab_size
cfg.FR_VOCAB_SIZE = train_dts.vocab_trg.vocab_size

model = Transformer(cfg)
model.load_state_dict(torch.load("checkpoint/22_08_2022 06_53_44.pkl"))

def inference(model, dataset, indices):
    for index in indices:
        src, src_mask, trg, trg_mask, labels, loss_mask = dataset[index]
        print("Translating sentence ({}): {} - {}".format(dataset.src_lan, dataset.df.iloc[index][dataset.src_lan], src))
        print("To sentence ({}): {} - {}".format(dataset.trg_lan, dataset.df.iloc[index][dataset.trg_lan], trg))
        src = src[:int(torch.sum(src_mask))].unsqueeze(0)
        encoder_output = model.encoder(src, None)
        predicted_sentence = ["<SOS>"]
        predicted_sentence_num = torch.Tensor([dataset.vocab_src.stoi["<SOS>"]])
        while True:
            decoder_output = model.decoder(encoder_output, predicted_sentence_num.unsqueeze(0), None, None)
            print(decoder_output)
            decoder_output_num = torch.argmax(decoder_output)
            decoder_output_string = dataset.vocab_trg.itos[int(decoder_output_num)]
            predicted_sentence.append(decoder_output_string)
            print(predicted_sentence_num, decoder_output_num)
            predicted_sentence_num = torch.cat((predicted_sentence_num, decoder_output_num.unsqueeze(0)))
            print(predicted_sentence)
            if decoder_output_string == "<EOS>":
                break
        print("*"*17)
inference(model, train_dts, [1,2])
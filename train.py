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
np.random.seed(37)
torch.manual_seed(73)

cfg = Config()

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
train_loader = DataLoader(train_dts, batch_size=cfg.BATCH_SIZE)
valid_loader = DataLoader(valid_dts, batch_size=cfg.BATCH_SIZE)

model = Transformer(cfg)
# model.load_state_dict(torch.load("checkpoint/22_08_2022 02_13_37.pkl"))
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1.0e-9)
scheduler = Scheduler(optimizer, cfg)
loss_fnc = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="sum")

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
logging.basicConfig(filename="checkpoint/logger",
                    format='%(asctime)s %(process)d,%(threadName)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train(train_loader, valid_loader, model, optimizer, scheduler, loss_fnc, logger):
    logger.info("*"*17)
    logger.info("Start training: " + str(datetime.now().strftime("%d_%m_%Y %H_%M_%S")))
    logger.info("*"*17)
    model.train()
    n_epochs = 100
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for _, (src, src_mask, trg, trg_mask, labels, loss_mask) in enumerate(tqdm.tqdm(train_loader)):
            logits = model(src, src_mask, trg, trg_mask) * loss_mask
            labels = labels * loss_mask
            # loss calculation
            loss = loss_fnc(logits.transpose(-2, -1), labels.transpose(-2, -1))/torch.sum(trg_mask)
            epoch_loss += loss.item()

            # back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # learning rate scheduler
            if scheduler is not None:
                scheduler.step()
        logger.info("Epoch: {} - Total loss: {}".format(epoch + 1, epoch_loss))
        print("Epoch: {} - Total loss: {}".format(epoch + 1, epoch_loss))
        if epoch % 5 == 0:
            now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
            torch.save(model.state_dict(), "checkpoint/"+str(now)+".pkl")
            logger.info("Save successful!")
    return 

train(train_loader, valid_loader, model, optimizer, scheduler, loss_fnc, logger)
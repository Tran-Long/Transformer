import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from .vocabulary import Vocabulary

class MyDataset(Dataset):
    def __init__(self, config, train_df, valid_df=None, src_lan="en", trg_lan="fr", transform=None) -> None:
        # If create valid dts, valid_df must not None
        super(MyDataset, self).__init__()
        self.max_seq_len = config.MAX_SEQ_LEN
        self.df = train_df if valid_df is None else valid_df
        self.src_lan = src_lan
        self.trg_lan = trg_lan
        self.transform = transform
        self.vocab_src = Vocabulary(train_df, src_lan)
        self.vocab_trg = Vocabulary(train_df, trg_lan)
        self.device = config.DEVICE
    
    def __len__(self):
        return len(self.df)

    def padding(self, num_list):
        if len(num_list) >= self.max_seq_len:
            mask = [True for _ in range(self.max_seq_len)]
            return num_list[:self.max_seq_len], mask
        else:
            mask = [True for _ in range(len(num_list))] + [False for _ in range(self.max_seq_len-len(num_list))]
            for _ in range(self.max_seq_len - len(num_list)):
                num_list.append(self.vocab_src.stoi["<PAD>"])
            return num_list, mask
    
    def one_hot_encode(self, num_list, vocab):
        num_tensor = torch.Tensor(num_list).long()
        return F.one_hot(num_tensor, num_classes=vocab.vocab_size)

    def create_loss_mask(self, trg_mask):
        loss_mask = torch.Tensor(trg_mask).unsqueeze(0).repeat(self.vocab_trg.vocab_size, 1).T
        return loss_mask

    def __getitem__(self, index):
        src_sen = self.df.iloc[index][self.src_lan]
        trg_sen = self.df.iloc[index][self.trg_lan]
        src_num = self.vocab_src.convert_sen2num(src_sen)
        src_num, src_mask = self.padding(src_num)
        trg_num = self.vocab_trg.convert_sen2num(trg_sen)
        trg_num, trg_mask = self.padding(trg_num)
        return torch.Tensor(src_num).to(self.device), torch.Tensor(src_mask).to(self.device), torch.Tensor(trg_num).to(self.device), torch.Tensor(trg_mask).to(self.device), self.one_hot_encode(trg_num, self.vocab_trg).to(self.device), self.create_loss_mask(trg_mask).to(self.device)

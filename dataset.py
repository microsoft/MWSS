'''
Copyright (c) Microsoft Corporation, Yichuan Li and Kai Shu.
Licensed under the MIT license.
Authors: Guoqing Zheng (zheng@microsoft.com), Yichuan Li and Kai Shu
'''

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from itertools import chain
import os
import pickle
# from snorkel_process import weak_supervision
class FakeNewsDataset(Dataset):
    def __init__(self, file_name, tokenizer, is_weak, max_length, weak_type="", overwrite=False, balance_weak=False):
        super(FakeNewsDataset, self).__init__()
        tokenizer_name = type(tokenizer).__name__
        # if tokenizer_name == "BertTokenizer" or tokenizer_name == "RobertaTokenizer":
        pickle_file = file_name.replace(".csv", '_{}_{}.pkl'.format(max_length, tokenizer_name))
        self.weak_label_count = 3
        if os.path.exists(pickle_file) and overwrite is False:
            load_data = pickle.load(open(pickle_file, "rb"))
            for key, value in load_data.items():
                setattr(self, key, value)
        else:
                save_data = {}
                data = pd.read_csv(file_name)
                # if tokenizer_name == "BertTokenizer" or tokenizer_name == "RobertaTokenizer":

                self.news = [tokenizer.encode(i, max_length=max_length, pad_to_max_length=True) for i in data['news'].values.tolist()]
                self.attention_mask = [[1] * (i.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in i else len(i))
                                       for i in self.news]
                self.attention_mask = [mask + [0] * (max_length - len(mask)) for mask in self.attention_mask]

                if is_weak:
                    self.weak_labels = []
                    assert "label" not in data.columns, "noise data should not contain the clean label"
                    self.weak_labels = data.iloc[:, list(range(1, len(data.columns)))].values.tolist()
                    save_data.update({"weak_labels": data.iloc[:, list(range(1, len(data.columns)))].values.tolist()})
                else:
                    self.labels = data['label'].values.tolist()

                save_data.update({"news": self.news, "attention_mask": self.attention_mask})
                if is_weak is False:
                    save_data.update({"labels": self.labels})

                pickle.dump(save_data, open(pickle_file, "wb"))
        if is_weak:
            if weak_type == "most_vote":
                self.weak_labels = [1 if np.sum(i) > 1 else 0 for i in self.weak_labels]
            elif weak_type == "flat":
                self.weak_labels = list(chain.from_iterable(self.weak_labels))
                self.news = list(chain.from_iterable([[i] * self.weak_label_count for i in self.news]))
                self.attention_mask = list(
                    chain.from_iterable([[i] * self.weak_label_count for i in self.attention_mask]))
            #"credit_label","polarity_label","bias_label"
            elif weak_type == "cred":
                self.weak_labels = [i[0] for i in self.weak_labels]
            elif weak_type == "polar":
                self.weak_labels = [i[1] for i in self.weak_labels]
            elif weak_type == "bias":
                self.weak_labels = [i[2] for i in self.weak_labels]
        self.is_weak = is_weak
        self.weak_type = weak_type
        if self.is_weak and balance_weak:
            self.__balance_helper()
        if self.is_weak:
            self.__instance_shuffle()

    def __bert_tokenizer(self, tokenizer, max_length, data):
        encode_output = [tokenizer.encode_plus(i, max_length=max_length, pad_to_max_length=True) for i in
                     data['news'].values.tolist()]
        self.news = [i['input_ids'] for i in encode_output]
        self.attention_mask = [i['attention_mask'] for i in encode_output]

        self.token_type_ids = [i['token_type_ids'] for i in encode_output]

    def __instance_shuffle(self):
        index_array = np.array(list(range(len(self))))
        np.random.shuffle(index_array)
        self.news = np.array(self.news)[index_array]
        self.weak_labels = np.array(self.weak_labels)[index_array]
        self.attention_mask = np.array(self.attention_mask)[index_array]

    def __balance_helper(self):
        self.weak_labels = np.array(self.weak_labels)
        # minority_count = min(int(np.sum(self.weak_labels)), len(self.weak_labels) - int(np.sum(self.weak_labels)))
        majority_count = max(int(np.sum(self.weak_labels)), len(self.weak_labels) - int(np.sum(self.weak_labels)))
        one_index = np.argwhere(self.weak_labels == 1)
        zero_index = np.argwhere(self.weak_labels == 0)
        zero_index = list(zero_index.reshape(-1,)) + list(np.random.choice(len(zero_index), majority_count-len(zero_index)))
        one_index = list(one_index.reshape(-1,)) + list(np.random.choice(len(one_index), majority_count-len(one_index)))
        self.weak_labels = self.weak_labels[one_index + zero_index]
        self.news = np.array(self.news)[one_index + zero_index]
        self.attention_mask = np.array(self.attention_mask)[one_index + zero_index]
        if hasattr(self, "token_type_ids"):
            self.token_type_ids = np.array(self.token_type_ids)[one_index+zero_index]


    def __len__(self):
        return len(self.news)

    def __getitem__(self, item):
        if self.is_weak:
            return torch.tensor(self.news[item]), torch.tensor(self.attention_mask[item]), torch.tensor(
                self.weak_labels[item])
        else:
            return torch.tensor(self.news[item]), torch.tensor(self.attention_mask[item]), torch.tensor(self.labels[item])





class SnorkelDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length, overwrite=False):
        super(SnorkelDataset, self).__init__()
        tokenizer_name = type(tokenizer).__name__
        pickle_file = file_name.replace(".csv", '_{}_{}.pkl'.format(max_length, tokenizer_name))
        assert os.path.exists(pickle_file), "please run loadFakeNewsDataset first"
        snorkel_file = pickle_file + "_snorkel"
        if os.path.exists(snorkel_file) and overwrite is False:
            snorkel_data = pickle.load(open(snorkel_file, "rb"))

        else:
            snorkel_data = weak_supervision(pickle_file, snorkel_file)


        for key, value in snorkel_data.items():
            setattr(self, key, value)


    def __len__(self):
        return len(self.news)

    def __getitem__(self, item):
        return torch.tensor(self.news[item]), torch.tensor(self.attention_mask[item]), torch.tensor(self.snorkel_weak[item])







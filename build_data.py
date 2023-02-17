import torch 
import transformers
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings(action='ignore')

def train_test_split(data):
    # Split data
    df_train = data.iloc[:int(len(data)*0.6), :]
    df_val = data.iloc[int(len(data)*0.6):int(len(data)*0.8), :]
    df_test = data.iloc[int(len(data)*0.8):int(len(data)*0.9), :]
    df_infer = data.iloc[int(len(data)*0.9):, :]

    return df_train, df_val, df_test, df_infer

def align_label(texts, labels, tokenizer, labels_to_ids):

    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True) # output of tokenizer
    word_ids = tokenized_inputs.word_ids() # index of word

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, labels_to_ids):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j, tokenizer, labels_to_ids) for i,j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
import pandas as pd
import torch 
import numpy as np
import transformers
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import logging
logging.set_verbosity_error()
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
import argparse
import warnings
warnings.filterwarnings(action='ignore')

from Labeling import make_data
from build_data import train_test_split, align_label, DataSequence


def parse_args():
    parser = argparse.ArgumentParser(description='training BERT')
    parser.add_argument(
        '--api_url', type = str, default = None)
    parser.add_argument(
        '--secret_key', type = str, default = None)
    parser.add_argument(
        '--json_path', type = str, default = 'data/json_result')
    parser.add_argument(
        '--max_length', type = int, default = 512)
    parser.add_argument(
        '--batch_size', type = int, default = 2)
    parser.add_argument(
        '--learning_rate', type = int, default = 5e-3)
    parser.add_argument(
        '--epochs', type = int, default = 10)
    
    args = parser.parse_args()

    return args


class BertModel(torch.nn.Module):
    def __init__(self, unique_labels):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output


def train_model(model, df_train, df_val, tokenizer, labels_to_ids, batch_size, learning_rate, epochs):
    print('##### Train BERT #####')
    train_dataset = DataSequence(df_train, tokenizer, labels_to_ids)
    val_dataset = DataSequence(df_val, tokenizer, labels_to_ids)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    
    optimizer = SGD(model.parameters(), lr=learning_rate)

    model = model.to(device)

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(epochs):
        print('##### Epochs {} #####'.format(epoch_num))
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)
            
            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()
              
            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()
        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')
        
        PATH = 'pretrained_model/NER_{}.pth'.format(epoch_num)
        torch.save(model.state_dict(), PATH)


def main():
    args = parse_args()

    data = make_data(args.json_path)
    print('##### Finish Load Data #####')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    labels = [i.split() for i in data['labels'].values.tolist()]

    # Value of Label
    unique_labels = set() 
    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    # Get corpus
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

    df_train, df_val, _ , _ = train_test_split(data)

    model = BertModel(unique_labels)

    train_model(model, df_train, df_val, tokenizer, labels_to_ids, args.batch_size, args.learning_rate, args.epochs)


if __name__=='__main__':
    main()
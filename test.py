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
from evaluate import evaluate

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
    parser.add_argument(
        '--check_point', type = str, default = 'NER_3.pth' # accuracy 0.839
    )
    
    args = parser.parse_args()

    return args


class BertModel(torch.nn.Module):
    def __init__(self, unique_labels):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output


def main():
    args = parse_args()
    print('##### Load Dataset #####')
    data = pd.read_csv('data/df.csv')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    labels = [i.split() for i in data['labels'].values.tolist()]

    # Value of Label
    unique_labels = set() 
    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    # Get corpus
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

    _, _, df_test, _ = train_test_split(data)
    print('##### Load Checkpoint #####')
    model = BertModel(unique_labels)
    PATH = 'pretrained_model/{}'.format(args.check_point)
    model.load_state_dict(torch.load(PATH), strict=False)

    print('##### Check Test Accuracy')
    evaluate(model, df_test, tokenizer, labels_to_ids)


if __name__=='__main__':
    main()
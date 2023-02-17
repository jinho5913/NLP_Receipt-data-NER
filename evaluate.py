import torch
from torch.utils.data import DataLoader

from build_data import train_test_split, align_label, DataSequence


def evaluate(model, df_test, tokenizer, labels_to_ids):

    test_dataset = DataSequence(df_test, tokenizer, labels_to_ids)

    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)

            input_id = test_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, test_label)
            # logits.shape = (1,512,7)
            for i in range(logits.shape[0]):
              print(i)
              print(test_label[i])
              print(logits[i])
              print('\n')
              logits_clean = logits[i][test_label[i] != -100]
              print(logits_clean)
              print(logits_clean.shape)
              print('\n')
              label_clean = test_label[i][test_label[i] != -100]
              print(label_clean)
              print(label_clean.shape)
              print('\n')
              predictions = logits_clean.argmax(dim=1)
              #print(predictions)
              #print(predictions.shape)
              acc = (predictions == label_clean).float().mean()
              total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')
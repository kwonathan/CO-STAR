import torch
from torch.utils.data import Dataset

# Dataset class

class DatasetGen(Dataset):
    def __init__(self, encodings, labels, for_eval):
        self.encodings = encodings
        self.labels = labels
        self.for_eval = for_eval

    def __getitem__(self, idx):
        if self.for_eval:
            text = torch.tensor(self.encodings['input_ids'][idx])
            attention_text = torch.tensor(self.encodings['attention_mask'][idx])
        else:
            text = torch.tensor(self.encodings['input_ids'][idx] + self.labels['input_ids'][idx])
            attention_text = torch.tensor(self.encodings['attention_mask'][idx] + self.labels['attention_mask'][idx])

        labels = torch.tensor(self.labels['input_ids'][idx])
        item = {'input_ids': text, 'attention_mask': attention_text, 'labels': labels}

        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

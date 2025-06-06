import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 将多个列的文本组合成一个字符串
        text = " [SEP] ".join([str(col) for col in self.texts[idx]])
        
        # 处理缺失值
        text = text.replace('nan', '').strip()
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 将薪资转换为浮点数
        try:
            label = float(self.labels[idx])
        except (ValueError, TypeError):
            # 处理特殊情况，如"Less than 1 year"或缺失值
            if isinstance(self.labels[idx], str) and "Less than 1 year" in self.labels[idx]:
                label = 0.5
            else:
                label = 0.0

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        } 

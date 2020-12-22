import torch
from torch.utils.data import Dataset, DataLoader


class MBTIDataset(Dataset):
    
    def __init__(self, posts, targets, tokenizer, max_len):
        self.posts = posts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, item):
        post = str(self.posts[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            post,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'post_text': post,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = MBTIDataset(
        posts=df["posts"].to_numpy(),
        targets=df["type"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )
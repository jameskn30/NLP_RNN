import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, content, tokenizer, max_len, stride = 1):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(content)

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i: i + max_len]
            target_chunk = token_ids[i + 1: i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self) -> int: 
        return len(self.input_ids)

    def __getitem__(self, index) -> torch.Tensor:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(
    txt, batch_size = 8, max_len = 256, stride = 128, 
    shuffle = True, drop_last = True, num_workers = 0):

    assert 1 <= stride <= max_len, "Stride must be between 1 and max_len"

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_len, stride)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle = shuffle, 
        drop_last=drop_last, #drop last non-full batch, prevents dataloader issue
        num_workers=num_workers
    )
    return dataloader, tokenizer, dataset
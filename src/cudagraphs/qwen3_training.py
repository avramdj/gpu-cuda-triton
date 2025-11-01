import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

DS = load_dataset("roneneldan/TinyStories")

BATCH_SIZE = 16

train_ds = DS["train"].select(range(10000))
val_ds = DS["train"].select(range(10000, 11000))

train_ds = train_ds.map(lambda x: tokenizer(x["text"]), batched=True)
val_ds = val_ds.map(lambda x: tokenizer(x["text"]), batched=True)

print(train_ds[0])

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model.train()
optimizer = AdamW(model.parameters(), lr=1e-4)

for batch in tqdm(train_dataloader):
    x = batch.to(model.device)
    loss = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["input_ids"]).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
for batch in tqdm(val_dataloader):
    x = batch.to(model.device)
    loss = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["input_ids"]).loss
    print(loss)
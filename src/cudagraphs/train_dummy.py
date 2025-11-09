import os
from itertools import islice

import torch
from datasets import Dataset, load_dataset
from torch import optim
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_text_dataset(
    tokenizer, max_length, split="train", num_samples=1000, seed=42
):
    stream = load_dataset("allenai/c4", "en", split=split, streaming=True)
    sampled = list(islice(stream, num_samples))
    dataset = Dataset.from_list(sampled)
    dataset = dataset.shuffle(seed=seed)

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        return tokenized

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def main():
    device = torch.device("cuda")
    batch_size = 4
    max_seq_length = 128
    precision = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=True, use_fast=True
    )
    dataset = prepare_text_dataset(tokenizer, max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            trust_remote_code=True,
            torch_dtype=precision,
        )
        .to(device)
        .train()
    )
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    opt = optim.AdamW(model.parameters(), lr=1e-5, capturable=True)
    cudagraph = torch.cuda.CUDAGraph()
    cudagraph_stream = torch.cuda.Stream(device=device)

    static_input_ids = torch.empty(
        batch_size, max_seq_length, dtype=torch.long, device=device
    )
    static_attention_mask = torch.empty(
        batch_size, max_seq_length, dtype=torch.long, device=device
    )
    static_labels = torch.empty(
        batch_size, max_seq_length, dtype=torch.long, device=device
    )
    loader_iter = iter(loader)

    def fetch_batch():
        nonlocal loader_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        return batch["input_ids"], batch["attention_mask"], batch["labels"]

    def copy_to_static(input_ids, attention_mask, labels):
        static_input_ids.copy_(input_ids, non_blocking=True)
        static_attention_mask.copy_(attention_mask, non_blocking=True)
        static_labels.copy_(labels, non_blocking=True)

    input_ids0, attention_mask0, labels0 = fetch_batch()
    copy_to_static(input_ids0, attention_mask0, labels0)

    # warm tf up
    for _ in range(3):
        with torch.cuda.stream(cudagraph_stream):
            opt.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=static_input_ids,
                attention_mask=static_attention_mask,
                labels=static_labels,
            )
            loss = outputs.loss
            loss.backward()

    copy_to_static(input_ids0, attention_mask0, labels0)
    torch.cuda.synchronize()

    with torch.cuda.graph(cudagraph, stream=cudagraph_stream):
        opt.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=static_input_ids,
            attention_mask=static_attention_mask,
            labels=static_labels,
        )
        loss = outputs.loss
        loss.backward()
        opt.step()

    torch.cuda.synchronize()

    total_steps = 70

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            warmup=5,
            wait=50,
            active=15,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    for step in tqdm(range(1, total_steps + 1)):
        input_ids, attention_mask, labels = fetch_batch()
        copy_to_static(input_ids, attention_mask, labels)
        cudagraph.replay()
        prof.step()

    prof.export_chrome_trace("cuda_graph_profile.json")


if __name__ == "__main__":
    main()

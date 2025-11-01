import os
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

torch.backends.cudnn.fp32_precision = "tf32"

PROFILE = os.getenv("PROFILE", "0") == "1"

BATCH_SIZE = 1024
MAX_PROMPT_LENGTH = 32
MAX_TOKENS = 64  # 17026.65 tok/s RTX 4090 + torch 2.9.0
N_INFERENCE_STEPS = MAX_TOKENS - MAX_PROMPT_LENGTH

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda"
)
model = torch.compile(
    model, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=False
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps: {MAX_TOKENS}")
print(f"Inference steps: {N_INFERENCE_STEPS}")
print(f"Max prompt length: {MAX_PROMPT_LENGTH}")
print(f"Device: {model.device}")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


def sample(logits, temperature=0.7, top_k=50):
    logits_fp32 = logits.float()
    scaled_logits = logits_fp32[:, -1, :] / temperature
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
    next_token = torch.gather(top_k_indices, -1, sampled_indices)
    return next_token


def forward(
    model, input_ids, position_ids, past_key_values, cache_position, use_cache=True
):
    return sample(
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        ).logits
    )


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def run_model(model):
    model.eval()

    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=MAX_TOKENS,
        device=model.device,
        dtype=torch.float16,
    )

    static_input_ids = torch.zeros(BATCH_SIZE, 1, dtype=torch.long, device=model.device)
    static_position_ids = torch.zeros(
        BATCH_SIZE, 1, dtype=torch.long, device=model.device
    )
    static_cache_position = torch.zeros(1, dtype=torch.long, device=model.device)

    seq_len = past_key_values.get_seq_length()
    static_position_ids.fill_(seq_len)
    static_cache_position.fill_(seq_len)

    # warmup boy
    for _ in range(3):
        forward(
            model=model,
            input_ids=static_input_ids,
            position_ids=static_position_ids,
            past_key_values=past_key_values,
            cache_position=static_cache_position,
            use_cache=True,
        )
    torch.cuda.synchronize()

    cudagraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cudagraph):
        next_token = forward(
            model=model,
            input_ids=static_input_ids,
            position_ids=static_position_ids,
            past_key_values=past_key_values,
            cache_position=static_cache_position,
            use_cache=True,
        )

    def f():
        prompt = "Hello, this is a prefill"

        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(
            model.device
        )
        prompt_tokens = prompt_tokens.repeat(BATCH_SIZE, 1)

        assert prompt_tokens.shape[-1] <= MAX_PROMPT_LENGTH, "prompt too long"

        with torch.no_grad():
            outputs = model(
                prompt_tokens, past_key_values=past_key_values, use_cache=True
            )
            next_token.copy_(sample(outputs.logits))

        torch.cuda.synchronize()
        t0 = time.time()

        sentence = prompt_tokens[0].tolist()
        seq_len = past_key_values.get_seq_length()

        static_position_ids.fill_(seq_len - 1)
        static_cache_position.fill_(seq_len - 1)

        for _ in tqdm(range(N_INFERENCE_STEPS), disable=True):
            static_input_ids.copy_(next_token)
            static_position_ids.add_(1)
            static_cache_position.add_(1)
            cudagraph.replay()
            sentence.extend(next_token.tolist()[0])

        torch.cuda.synchronize()
        time_per_iter = (time.time() - t0) / N_INFERENCE_STEPS

        print(f"\nTokens/sec: {1 / time_per_iter * BATCH_SIZE:.2f}")
        print(tokenizer.decode(sentence))

    if PROFILE:
        with torch.profiler.profile(
            record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            f()
        prof.export_chrome_trace("qwen3.json")
        times = "\n".join(
            prof.key_averages()
            .table(sort_by="self_cuda_time_total", row_limit=0)
            .strip()
            .split("\n")[-2:]
        )
        print(times)
    else:
        f()


run_model(model)

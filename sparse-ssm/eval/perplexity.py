import torch
import time
import math
import statistics
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Cache tokenizer to avoid repeated loading (~1s each time)
_tokenizer_cache = {}


def _get_tokenizer(model):
    """Extract or load the tokenizer matching the model (cached)."""
    name = getattr(model.config, "_name_or_path", None)
    if not name:
        name = "state-spaces/mamba-130m-hf"
    if name not in _tokenizer_cache:
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _tokenizer_cache[name] = tok
    return _tokenizer_cache[name]


@torch.no_grad()
def _eval_perplexity(model, dataset_name, dataset_config, split,
                     max_samples=50):
    """Per-token perplexity matching the paper's methodology.

    Concatenates all tokens from the split, divides into non-overlapping
    chunks of 1024, and computes per-token negative log-likelihood.
    Returns a dict: {"ppl": float, "ci_low": float, "ci_high": float}
    where ci_low/ci_high form a 95% bootstrap CI over per-chunk losses.
    """
    tokenizer = _get_tokenizer(model)

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    model.eval()

    # Concatenate all text and tokenize once
    all_text = "\n\n".join(ex["text"].strip() for ex in dataset
                           if ex["text"].strip())
    encodings = tokenizer(all_text, return_tensors="pt")
    all_ids = encodings["input_ids"][0]               # [total_tokens]

    seq_len = 1024
    n_chunks = max(1, all_ids.size(0) // seq_len)
    all_ids = all_ids[: n_chunks * seq_len]           # trim to full chunks

    chunk_losses = []
    for i in tqdm(range(n_chunks), desc=f"PPL ({dataset_name})", leave=False):
        chunk = all_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
        chunk = chunk.to(model.device)
        outputs = model(chunk, labels=chunk)
        loss = outputs.loss.item()
        if math.isnan(loss) or math.isinf(loss):
            continue
        chunk_losses.append(loss)

    if len(chunk_losses) == 0:
        return {"ppl": float('inf'), "ci_low": float('inf'),
                "ci_high": float('inf')}
    return _bootstrap_ppl(chunk_losses)


def _bootstrap_ppl(losses, n_bootstrap=200):
    """Compute perplexity with 95% bootstrap confidence interval.

    Uses numpy for fast vectorized bootstrap resampling.
    """
    import numpy as np
    losses_arr = np.array(losses)
    avg_loss = losses_arr.mean()
    ppl = math.exp(float(avg_loss))

    n = len(losses)
    rng = np.random.RandomState(42)
    # Vectorized: sample all at once → [n_bootstrap, n]
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = losses_arr[indices].mean(axis=1)
    boot_ppls = np.exp(boot_means)
    boot_ppls.sort()
    ci_low = float(boot_ppls[int(0.025 * n_bootstrap)])
    ci_high = float(boot_ppls[int(0.975 * n_bootstrap)])
    return {"ppl": ppl, "ci_low": ci_low, "ci_high": ci_high}


@torch.no_grad()
def _eval_perplexity_streaming(model, dataset_name, dataset_config, split,
                               max_samples=50):
    """Perplexity eval using streaming (avoids downloading full dataset).

    Returns dict with "ppl", "ci_low", "ci_high".
    """
    tokenizer = _get_tokenizer(model)

    dataset = load_dataset(dataset_name, dataset_config, split=split,
                           streaming=True)
    model.eval()
    losses = []

    for example in tqdm(dataset, desc=f"PPL ({dataset_name})", leave=False,
                        total=max_samples):
        text = example["text"].strip()
        if len(text) < 100:
            continue

        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if inputs["input_ids"].shape[1] < 20:
            continue

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

        if math.isnan(loss) or math.isinf(loss):
            continue

        losses.append(loss)
        if len(losses) >= max_samples:
            break

    if len(losses) == 0:
        return {"ppl": float('inf'), "ci_low": float('inf'),
                "ci_high": float('inf')}
    return _bootstrap_ppl(losses)


def evaluate_wikitext_perplexity(model, max_samples=None):
    """WikiText-2 Perplexity (full validation set by default).

    Returns dict with ppl, ci_low, ci_high.
    """
    return _eval_perplexity(model, "wikitext", "wikitext-2-raw-v1",
                            "validation", max_samples=max_samples)


def evaluate_c4_perplexity(model, max_samples=20):
    """C4 validation Perplexity (streaming to avoid 300GB download)."""
    return _eval_perplexity_streaming(
        model, "allenai/c4", "en", "validation", max_samples=max_samples
    )


def evaluate_all_perplexity(model, max_samples=None):
    """Evaluate perplexity on WikiText-2 and C4.

    Returns dict mapping dataset name to {"ppl", "ci_low", "ci_high"}.
    PTB is excluded — its HF dataset script is deprecated and no longer loads.

    Args:
        max_samples: cap on dataset samples (None = full validation set for
            WikiText-2, 50 docs for streaming C4).
    """
    results = {}
    results["wikitext2"] = evaluate_wikitext_perplexity(model, max_samples)
    try:
        c4_samples = max_samples if max_samples is not None else 20
        results["c4"] = evaluate_c4_perplexity(model, c4_samples)
    except Exception as e:
        results["c4"] = f"error: {e}"
    return results


@torch.no_grad()
def benchmark_inference(model, n_tokens=128, n_runs=5):
    """Measure inference latency (with std dev) and memory usage."""
    tokenizer = _get_tokenizer(model)

    device = model.device
    model.eval()

    input_ids = torch.randint(0, tokenizer.vocab_size, (1, n_tokens),
                              device=device)

    # Warm-up
    for _ in range(2):
        model(input_ids)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    latency_ms = statistics.mean(times) * 1000
    latency_std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0

    mem_mb = 0.0
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    total_params = 0
    nonzero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        nonzero_params += p.count_nonzero().item()

    return {
        "latency_ms": round(latency_ms, 2),
        "latency_std_ms": round(latency_std_ms, 2),
        "memory_mb": round(mem_mb, 2),
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity_pct": round((1 - nonzero_params / total_params) * 100, 2),
    }

#!/usr/bin/env python
"""
SparseSSM Replication -- Lab Assignment

Implements the pruning framework from:
  "SparseSSM: Efficient Selective Structured State Space Models
   Can Be Pruned in One-Shot" (Tuo & Wang, arXiv:2506.09613)
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from prune.sparsessm import SparseSSMPruner
from eval.perplexity import (evaluate_wikitext_perplexity,
                              evaluate_all_perplexity,
                              benchmark_inference)


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def run_single(args):
    """Single pruning run at a given sparsity."""
    torch.manual_seed(args.seed)
    print(f"\n{'='*60}")
    print(f"SparseSSM | Model: {args.model} | "
          f"Sparsity: {args.sparsity*100:.0f}% | Mode: {args.prune_mode}")
    print(f"{'='*60}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="train")
    model, tokenizer = load_model(args.model)

    # -- Benchmark BEFORE pruning --
    if args.skip_before_eval:
        print("\nSkipping before-pruning evaluation (--skip_before_eval)")
        ppl_before = {"wikitext2": {"ppl": 0, "ci_low": 0, "ci_high": 0}}
        bench_before = {"latency_ms": 0, "latency_std_ms": 0,
                        "memory_mb": 0, "total_params": 0,
                        "nonzero_params": 0, "sparsity_pct": 0}
    else:
        print("\nEvaluation BEFORE pruning...")
        ppl_before = evaluate_all_perplexity(model, max_samples=args.max_eval_samples)
        bench_before = benchmark_inference(model)
        wiki_b = ppl_before['wikitext2']
        print(f"  WikiText-2 PPL = {wiki_b['ppl']:.2f} "
              f"(95% CI: [{wiki_b['ci_low']:.2f}, {wiki_b['ci_high']:.2f}])")
        for ds, val in ppl_before.items():
            if ds != "wikitext2" and isinstance(val, dict):
                print(f"  {ds} PPL = {val['ppl']:.2f} "
                      f"(95% CI: [{val['ci_low']:.2f}, {val['ci_high']:.2f}])")
        print(f"  Latency = {bench_before['latency_ms']:.1f} "
              f"+/- {bench_before['latency_std_ms']:.1f} ms | "
              f"Memory = {bench_before['memory_mb']:.1f} MB | "
              f"Sparsity = {bench_before['sparsity_pct']:.2f}%")

    # -- Pruning --
    pruner = SparseSSMPruner(model, dataset, args)
    pruned_model = pruner.prune()

    # -- Save --
    os.makedirs(args.output_dir, exist_ok=True)
    pruned_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to ./{args.output_dir}")

    # -- Benchmark AFTER pruning --
    print("\nEvaluation AFTER pruning...")
    ppl_after = evaluate_all_perplexity(pruned_model, max_samples=args.max_eval_samples)
    bench_after = benchmark_inference(pruned_model)
    wiki_a = ppl_after['wikitext2']
    print(f"  WikiText-2 PPL = {wiki_a['ppl']:.2f} "
          f"(95% CI: [{wiki_a['ci_low']:.2f}, {wiki_a['ci_high']:.2f}])")
    for ds, val in ppl_after.items():
        if ds != "wikitext2" and isinstance(val, dict):
            print(f"  {ds} PPL = {val['ppl']:.2f} "
                  f"(95% CI: [{val['ci_low']:.2f}, {val['ci_high']:.2f}])")
    print(f"  Latency = {bench_after['latency_ms']:.1f} "
          f"+/- {bench_after['latency_std_ms']:.1f} ms | "
          f"Memory = {bench_after['memory_mb']:.1f} MB | "
          f"Sparsity = {bench_after['sparsity_pct']:.2f}%")

    # -- Summary --
    wiki_before = ppl_before["wikitext2"]["ppl"]
    wiki_after = ppl_after["wikitext2"]["ppl"]
    speedup = bench_before["latency_ms"] / bench_after["latency_ms"] \
        if bench_after["latency_ms"] > 0 else 0

    print(f"\n{'='*60}")
    print(f"SUMMARY (sparsity={args.sparsity*100:.0f}%)")
    if wiki_before > 0:
        print(f"  PPL:     {wiki_before:.2f} -> {wiki_after:.2f} "
              f"({(wiki_after/wiki_before - 1)*100:+.1f}%)")
    else:
        print(f"  PPL (after):  {wiki_after:.2f}")
    if bench_before['latency_ms'] > 0:
        print(f"  Latency: {bench_before['latency_ms']:.1f} -> "
              f"{bench_after['latency_ms']:.1f} ms "
              f"({speedup:.2f}x)")
    else:
        print(f"  Latency (after): {bench_after['latency_ms']:.1f} ms")
    print(f"  Params:  {bench_after['sparsity_pct']:.2f}% sparse")
    print(f"{'='*60}")

    return {
        "sparsity": args.sparsity,
        "mode": args.prune_mode,
        "method": args.ssm_method,
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "bench_before": bench_before,
        "bench_after": bench_after,
        "speedup": round(speedup, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="SparseSSM Pruning")
    parser.add_argument("--model", type=str,
                        default="state-spaces/mamba-130m-hf")
    parser.add_argument("--sparsity", type=float, default=0.50)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="pruned_mamba")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prune_mode", type=str, default="ssm",
                        choices=["ssm", "full", "structured", "structured+ffn"],
                        help="ssm = A_log unstructured; full = A_log + FFN; "
                             "structured = remove SSM columns; "
                             "structured+ffn = structured SSM + FFN pruning")
    parser.add_argument("--ssm_method", type=str, default="algorithm1",
                        choices=["algorithm1", "l2"],
                        help="algorithm1 = full Algorithm 1; l2 = simplified")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Cap evaluation samples (None = full validation)")
    parser.add_argument("--skip_before_eval", action="store_true",
                        help="Skip before-pruning evaluation (baseline is "
                             "always the same for a given model)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run sparsity sweep (20%%-60%%)")
    args = parser.parse_args()

    result = run_single(args)
    os.makedirs("results", exist_ok=True)
    fname = f"results/{args.model.split('/')[-1]}_sp{int(args.sparsity*100)}pct.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to {fname}")

if __name__ == "__main__":
    main()

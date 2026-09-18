#!/usr/bin/env python3
"""
tools/eval/hf_perplexity_baseline.py

Compute HuggingFace Transformers perplexity baseline for comparison with XInfer.

This script tokenizes the exact same input text using the same tokenizer,
runs the HF model in eval mode, and records:
  - Overall perplexity
  - Per-token NLL losses

The output JSON is consumed by eval_perplexity --hf-baseline to enforce
the fail-loud divergence contract.

Usage:
    python hf_perplexity_baseline.py \
        --model Qwen/Qwen3.8-27B \
        --input input.txt \
        --output hf_baseline.json \
        [--max-tokens 4096] \
        [--device cpu|cuda|auto]

Requirements:
    pip install torch transformers
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Compute HF Transformers perplexity baseline for XInfer comparison"
    )
    parser.add_argument("--model", required=True, help="HF model name or local checkpoint path")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to evaluate")
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: float16)")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: This script requires 'torch' and 'transformers'.", file=sys.stderr)
        print("Install with: pip install torch transformers", file=sys.stderr)
        sys.exit(2)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    print(f"[hf_baseline] Loading model: {args.model}")
    print(f"[hf_baseline] Device: {device}, dtype: {args.dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[hf_baseline] Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"[hf_baseline] Model loaded")

    # Read and tokenize input
    input_text = Path(args.input).read_text(encoding="utf-8")
    encodings = tokenizer(input_text, return_tensors="pt", truncation=True,
                          max_length=args.max_tokens)
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]
    print(f"[hf_baseline] Input tokenized: {seq_len} tokens")

    if seq_len <= 1:
        print("ERROR: Need at least 2 tokens for perplexity evaluation", file=sys.stderr)
        sys.exit(2)

    # Forward pass to get logits
    print("[hf_baseline] Running forward pass...")
    t_start = time.time()

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    t_fwd = time.time() - t_start
    print(f"[hf_baseline] Forward pass complete in {t_fwd:.2f}s")

    # Compute per-token cross-entropy loss (causal: logits[i] predicts token[i+1])
    # Shift: logits[0..seq_len-2] predict tokens[1..seq_len-1]
    shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:].contiguous()     # [1, seq_len-1]

    num_positions = seq_len - 1

    # Compute log_softmax + NLL per token
    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)  # [1, N, V]

    # Gather the log-prob of the ground truth token at each position
    per_token_log_probs = log_probs[0].gather(
        dim=-1, index=shift_labels[0].unsqueeze(-1)
    ).squeeze(-1)  # [N]

    per_token_nll = -per_token_log_probs  # [N]
    total_nll = per_token_nll.sum().item()
    mean_nll = total_nll / num_positions
    perplexity = math.exp(mean_nll)

    per_token_loss_list = per_token_nll.cpu().tolist()
    token_ids_list = input_ids[0].cpu().tolist()

    # Report
    print(f"\n{'='*40}")
    print(f" HF Baseline Perplexity Results")
    print(f"{'='*40}")
    print(f"  Model:           {args.model}")
    print(f"  Tokens evaluated: {num_positions}")
    print(f"  Total NLL loss:  {total_nll:.4f}")
    print(f"  Mean NLL loss:   {mean_nll:.4f}")
    print(f"  Perplexity:      {perplexity:.4f}")
    print(f"{'='*40}")

    # Write output JSON
    result = {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "input_file": args.input,
        "seq_len": seq_len,
        "num_positions": num_positions,
        "total_nll_loss": total_nll,
        "mean_nll_loss": mean_nll,
        "perplexity": perplexity,
        "per_token_loss": per_token_loss_list,
        "token_ids": token_ids_list,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[hf_baseline] Results written to: {args.output}")


if __name__ == "__main__":
    main()

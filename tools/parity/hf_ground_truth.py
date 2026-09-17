#!/usr/bin/env python3
"""
tools/parity/hf_ground_truth.py
Extract real ground-truth logits and top-k predictions from official Qwen3.8-27B HF checkpoint.
Uses single-active-shard streaming on CPU with torch.inference_mode().
Peak RAM stays manageable via shard rotation. Runs layer-by-layer to avoid OOM.
"""

import os
import sys
import gc
import json
import time
import ctypes
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5RMSNorm,
)


def trim_ram():
    gc.collect()
    if sys.platform == "win32":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass


class ShardManager:
    """Manages at most 2 open safetensors shard handles to minimize RAM while avoiding re-opens."""
    def __init__(self, ckpt_dir: Path, device: str):
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.open_shards: Dict[str, Any] = {}
        self.max_open = 2

    def get_tensor(self, shard_file: str, key: str) -> torch.Tensor:
        if shard_file not in self.open_shards:
            if len(self.open_shards) >= self.max_open:
                # Close the oldest shard
                oldest = next(iter(self.open_shards))
                del self.open_shards[oldest]
                trim_ram()
            self.open_shards[shard_file] = safe_open(
                self.ckpt_dir / shard_file, framework="pt", device=self.device
            )
        return self.open_shards[shard_file].get_tensor(key)

    def close_all(self):
        self.open_shards.clear()
        trim_ram()


def run_ground_truth(
    checkpoint_dir: str,
    prompt: str,
    output_json: str,
    top_k: int = 10,
    device_str: str = "xpu:0",
):
    ckpt_path = Path(checkpoint_dir)
    print("=" * 72, flush=True)
    print("  Qwen3.8-27B Official HF Checkpoint: Ground-Truth Logits Extractor", flush=True)
    print("=" * 72, flush=True)
    print(f" Checkpoint:  {ckpt_path}", flush=True)
    print(f" Device:      {device_str}", flush=True)
    print(f" Prompt:      '{prompt}'", flush=True)
    print("=" * 72, flush=True)

    device = torch.device(device_str)

    # 1. Load config & tokenizer
    print("[1/5] Loading config and tokenizer...", flush=True)
    text_cfg = AutoConfig.from_pretrained(ckpt_path).text_config
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Tokenize prompt without chat template
    token_ids = tokenizer.encode(prompt)
    token_pieces = [tokenizer.decode([t]) for t in token_ids]
    print(f"      Prompt Tokens ({len(token_ids)}): {token_ids}", flush=True)
    print(f"      Token Pieces: {token_pieces}", flush=True)

    # 2. Index safetensors shards
    print("[2/5] Indexing safetensors shards...", flush=True)
    index_path = ckpt_path / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shards = ShardManager(ckpt_path, device_str)

    with torch.inference_mode():
        # 3. Embedding lookup
        print("[3/5] Performing embedding lookup directly to GPU...", flush=True)
        embed_shard = weight_map["model.language_model.embed_tokens.weight"]
        embed_w = shards.get_tensor(embed_shard, "model.language_model.embed_tokens.weight")
        input_ids_tensor = torch.tensor([token_ids], device=device)
        hidden_states = F.embedding(input_ids_tensor, embed_w)
        del embed_w
        trim_ram()

        seq_len = input_ids_tensor.shape[1]

        # Precompute rotary embeddings
        rotary_emb = Qwen3_5TextRotaryEmbedding(text_cfg).to(device)
        pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(4, 1, -1)
        text_pos_ids = pos_ids[0]
        mrope_pos_ids = pos_ids[1:]
        position_embeddings = rotary_emb(hidden_states, mrope_pos_ids)
        del rotary_emb
        trim_ram()

        # 4. Stream all 64 layers sequentially
        print(f"[4/5] Executing all {text_cfg.num_hidden_layers} layers sequentially...", flush=True)
        t_start = time.time()

        for l in range(text_cfg.num_hidden_layers):
            l_t0 = time.time()
            prefix = f"model.language_model.layers.{l}."
            needed_keys = [k for k in weight_map if k.startswith(prefix)]

            layer_state = {
                k[len(prefix):]: shards.get_tensor(weight_map[k], k)
                for k in needed_keys
            }

            layer = Qwen3_5DecoderLayer(text_cfg, layer_idx=l).to(device, dtype=torch.bfloat16)
            layer.load_state_dict(layer_state, strict=True)

            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=text_pos_ids,
            )

            del layer, layer_state
            trim_ram()

            if (l + 1) % 8 == 0 or l == text_cfg.num_hidden_layers - 1:
                elapsed = time.time() - t_start
                import psutil
                ram_mb = psutil.Process().memory_info().rss / (1024**2)
                print(f"      Completed layer {l+1:2d}/{text_cfg.num_hidden_layers} in {elapsed:5.1f}s | RAM: {ram_mb:6.1f} MB", flush=True)

        # 5. Final Norm & LM Head
        print("[5/5] Computing final RMSNorm and LM Head projection...", flush=True)
        norm_shard = weight_map["model.language_model.norm.weight"]
        norm_w = shards.get_tensor(norm_shard, "model.language_model.norm.weight")
        norm = Qwen3_5RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).to(device, dtype=torch.bfloat16)
        norm.weight.data.copy_(norm_w)
        hidden_states = norm(hidden_states)
        del norm, norm_w

        lm_head_shard = weight_map["lm_head.weight"]
        lm_head_w = shards.get_tensor(lm_head_shard, "lm_head.weight")
        logits = F.linear(hidden_states, lm_head_w).float() # [1, seq_len, vocab_size]
        del lm_head_w

        shards.close_all()

        total_time = time.time() - t_start
        print(f"      Full 64-layer forward pass completed in {total_time:.2f} seconds.", flush=True)

    # 6. Extract predictions & format output
    positions_data = []
    print("\n" + "=" * 72, flush=True)
    print("  OFFICIAL HF GROUND-TRUTH PREDICTIONS PER POSITION", flush=True)
    print("=" * 72, flush=True)

    for pos in range(seq_len):
        pos_logits = logits[0, pos] # [vocab_size]
        probs = F.softmax(pos_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        top_logits_vals = pos_logits[top_indices]

        top_tokens = []
        for i in range(top_k):
            tid = top_indices[i].item()
            tok_str = tokenizer.decode([tid])
            top_tokens.append({
                "rank": i + 1,
                "token_id": tid,
                "token_str": tok_str,
                "logit": round(top_logits_vals[i].item(), 4),
                "probability": round(top_probs[i].item(), 6),
            })

        pos_input_tok = token_ids[pos]
        pos_input_str = tokenizer.decode([pos_input_tok])
        target_desc = f"Position {pos}: After '{pos_input_str}' (ID {pos_input_tok})"
        if pos == seq_len - 1:
            target_desc += " -> PREDICTION FOR NEXT TOKEN"

        print(f"\n{target_desc}", flush=True)
        print(f"{'Rank':<5} {'Token ID':<10} {'Token String':<20} {'Logit':<12} {'Probability':<12}", flush=True)
        print("-" * 65, flush=True)
        for t in top_tokens:
            print(f"{t['rank']:<5} {t['token_id']:<10} {repr(t['token_str']):<20} {t['logit']:<12.4f} {t['probability']:<12.6f}", flush=True)

        positions_data.append({
            "position": pos,
            "input_token_id": pos_input_tok,
            "input_token_str": pos_input_str,
            "top_predictions": top_tokens,
            "top1_token_id": top_indices[0].item(),
            "top1_token_str": tokenizer.decode([top_indices[0].item()]),
            "top1_logit": round(top_logits_vals[0].item(), 4),
            "top1_prob": round(top_probs[0].item(), 6),
        })

    # Save to json
    result_record = {
        "model_name": "Qwen/Qwen3.8-27B",
        "checkpoint_dir": str(ckpt_path),
        "precision": "bfloat16 (unquantized)",
        "prompt": prompt,
        "prompt_tokens": token_ids,
        "prompt_token_strings": token_pieces,
        "forward_time_sec": round(total_time, 2),
        "positions": positions_data,
        "next_predicted_token_id": positions_data[-1]["top1_token_id"],
        "next_predicted_token_str": positions_data[-1]["top1_token_str"],
        "next_predicted_prob": positions_data[-1]["top1_prob"],
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)

    # Save raw logits vector for all prompt positions to numpy binary file
    all_logits_np = logits[0].cpu().numpy() # [seq_len, vocab_size]
    np_path = out_file.with_suffix(".npy")
    np.save(str(np_path), all_logits_np)

    print("\n" + "=" * 72, flush=True)
    print(f" [GROUND TRUTH GENERATED SUCCESSFULLY]", flush=True)
    print(f" Output JSON:  {out_file}", flush=True)
    print(f" Output Numpy: {np_path} (shape: {all_logits_np.shape}, dtype: {all_logits_np.dtype})", flush=True)
    print(f" Next Predicted Token: ID {positions_data[-1]['top1_token_id']} ({repr(positions_data[-1]['top1_token_str'])}) with probability {positions_data[-1]['top1_prob']:.4f}", flush=True)
    print("=" * 72, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract HF ground-truth logits for Qwen3.8-27B")
    parser.add_argument("--checkpoint-dir", type=str, default=r"H:\Models\Qwen3.8-27B", help="Path to HF checkpoint")
    parser.add_argument("--prompt", type=str, default="The sky is", help="Prompt text")
    parser.add_argument("--output-json", type=str, default=r"tools\parity\hf_ground_truth.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top predictions to record")
    args = parser.parse_args()

    run_ground_truth(
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
        output_json=args.output_json,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

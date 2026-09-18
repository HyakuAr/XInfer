#!/usr/bin/env python3
"""
tools/convert/qwen3_8_27b/convert.py
Offline Hugging Face checkpoint -> .xinfer converter for Qwen3.8-27B with INT4 quantization.
"""

import os
import sys
import json
import time
import struct
import ctypes
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import functools
print = functools.partial(print, flush=True)

import torch
import safetensors.torch

# Container constants matching src/artifact/container.h
MAGIC_BYTES = b"XINFER\0\0"
FOOTER_MAGIC_BYTES = b"XINFFOOT"
CURRENT_FORMAT_VERSION = 1
SECTION_ALIGNMENT = 64
MAX_SECTION_NAME_LEN = 96

# Section types
SECTION_TYPE_RAW_BLOB = 0x0000
SECTION_TYPE_METADATA_JSON = 0x0001
SECTION_TYPE_TOKENIZER_DATA = 0x0002
SECTION_TYPE_TENSOR_WEIGHTS = 0x0003
SECTION_TYPE_TENSOR_SCALES = 0x0004
SECTION_TYPE_CHAT_TEMPLATE = 0x0005


class FastCRC64:
    def __init__(self, dll_path: Optional[str] = None):
        self.dll = None
        if dll_path and os.path.exists(dll_path):
            try:
                self.dll = ctypes.CDLL(dll_path)
                self.dll.xinfer_crc64.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
                self.dll.xinfer_crc64.restype = ctypes.c_uint64
                self.dll.xinfer_crc64_file.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
                self.dll.xinfer_crc64_file.restype = ctypes.c_uint64
            except Exception as e:
                print(f"[Warning] Failed to load DLL for fast CRC64: {e}")
                self.dll = None

        if self.dll is None:
            # Precompute fallback table
            POLY64 = 0xC96C5795D7870F42
            self.table = []
            for b in range(256):
                crc = b
                for _ in range(8):
                    if crc & 1:
                        crc = (crc >> 1) ^ POLY64
                    else:
                        crc >>= 1
                self.table.append(crc)

    def calculate_bytes(self, data: bytes) -> int:
        if self.dll:
            return self.dll.xinfer_crc64(data, len(data))
        crc = 0xFFFFFFFFFFFFFFFF
        for byte in data:
            crc = (crc >> 8) ^ self.table[(crc ^ byte) & 0xFF]
        return crc ^ 0xFFFFFFFFFFFFFFFF

    def calculate_file(self, filepath: str, max_bytes: int = 0) -> int:
        if self.dll:
            p = filepath.replace("\\", "/").encode("utf-8")
            return self.dll.xinfer_crc64_file(p, max_bytes)
        crc = 0xFFFFFFFFFFFFFFFF
        with open(filepath, "rb") as f:
            bytes_left = max_bytes if max_bytes > 0 else os.path.getsize(filepath)
            chunk_size = 4 * 1024 * 1024
            while bytes_left > 0:
                to_read = min(bytes_left, chunk_size)
                chunk = f.read(to_read)
                if not chunk:
                    break
                for byte in chunk:
                    crc = (crc >> 8) ^ self.table[(crc ^ byte) & 0xFF]
                bytes_left -= len(chunk)
        return crc ^ 0xFFFFFFFFFFFFFFFF


def quantize_int4_symmetric(tensor: torch.Tensor, group_size: int = 128) -> Tuple[bytes, bytes, torch.Tensor]:
    """
    Quantize 2D weight matrix to symmetric INT4 per group.
    Returns (packed_bytes, scales_bytes, scales_tensor_float32).
    """
    orig_shape = tensor.shape
    x = tensor.float().view(-1, group_size)
    
    # Symmetric scale = max(|x|) / 7.0
    max_abs = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 7.0, min=1e-8)
    
    # Quantize to [-8, 7]
    q = torch.clamp(torch.round(x / scale), -8, 7).to(torch.int8)
    
    # Pack two 4-bit values into one uint8: low nibble = even, high nibble = odd
    low = (q[:, 0::2] & 0x0F).to(torch.uint8)
    high = ((q[:, 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = (low | high).contiguous()
    
    # FP16 scales
    scales_fp16 = scale.squeeze(-1).to(torch.float16).contiguous()
    
    packed_bytes = packed.numpy().tobytes()
    scales_bytes = scales_fp16.numpy().tobytes()
    
    return packed_bytes, scales_bytes, scale.squeeze(-1)


def get_tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def dequantize_int4_symmetric(packed_bytes: bytes, scales: torch.Tensor, orig_shape: torch.Size, group_size: int = 128) -> torch.Tensor:
    """
    Numerical Oracle: reference FP32 dequantization of INT4 packed weights.
    """
    import numpy as np
    arr = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
    packed = torch.from_numpy(arr).view(-1, group_size // 2)
    
    unpacked_low = (packed & 0x0F).to(torch.int8)
    unpacked_low = torch.where(unpacked_low >= 8, unpacked_low - 16, unpacked_low).float()
    
    unpacked_high = ((packed >> 4) & 0x0F).to(torch.int8)
    unpacked_high = torch.where(unpacked_high >= 8, unpacked_high - 16, unpacked_high).float()
    
    unpacked = torch.empty((packed.shape[0], group_size), dtype=torch.float32)
    unpacked[:, 0::2] = unpacked_low
    unpacked[:, 1::2] = unpacked_high
    
    dequant = unpacked * scales.view(-1, 1).float()
    return dequant.view(orig_shape)


def compute_cosine_similarity(t1: torch.Tensor, t2: torch.Tensor, chunk_size: int = 10_000_000) -> float:
    """
    Numerically stable cosine similarity using chunked float64 accumulation.
    Avoids float32 reduction underflow on large tensors (e.g. lm_head with ~1.27B elements)
    which causes PyTorch's native F.cosine_similarity to produce impossible values > 1.0.
    """
    flat1 = t1.reshape(-1)
    flat2 = t2.reshape(-1)
    n = flat1.numel()

    dot_sum = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c1 = flat1[start:end].double()
        c2 = flat2[start:end].double()
        dot_sum += torch.dot(c1, c2).item()
        norm1_sq += torch.dot(c1, c1).item()
        norm2_sq += torch.dot(c2, c2).item()

    denom = math.sqrt(norm1_sq) * math.sqrt(norm2_sq)
    if denom == 0.0:
        return 0.0
    return float(dot_sum / denom)


def should_quantize_tensor(name: str, tensor: torch.Tensor, group_size: int) -> bool:
    """
    Identify 2D projection linear weights that must be INT4-quantized.
    """
    if tensor.ndim != 2:
        return False
    # Check if divisible by group size
    if tensor.shape[1] % group_size != 0:
        return False
    
    # Do not quantize embedding tokens (retained in BF16 for high token precision)
    if "embed_tokens" in name:
        return False
    
    linear_patterns = [
        "proj", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
        "lm_head"
    ]
    return any(p in name for p in linear_patterns)


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.8-27B BF16 checkpoint to .xinfer INT4 format")
    parser.add_argument("--checkpoint-dir", type=str, default=r"H:\Models\Qwen3.8-27B", help="Path to HF checkpoint")
    parser.add_argument("--output-path", type=str, default=r"out\qwen3_8_27b.xinfer", help="Output .xinfer file path")
    parser.add_argument("--group-size", type=int, default=128, help="INT4 group size")
    parser.add_argument("--parity-samples", type=int, default=16, help="Number of tensors to verify for numerical parity")
    parser.add_argument("--dll-path", type=str, default=r"build\xinfer_artifact_c.dll", help="Path to fast CRC64 DLL")
    parser.add_argument("--parity-report", type=str, default=r"tools\convert\qwen3_8_27b\parity_report.json", help="Report output")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.parity_report).parent.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print(" xinfer Model Converter: Qwen3.8-27B BF16 -> INT4 (.xinfer)")
    print(f" Source: {ckpt_dir}")
    print(f" Output: {out_path}")
    print(f" Group Size: {args.group_size}")
    print("==========================================================")

    if not ckpt_dir.exists():
        print(f"[Error] Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    crc_engine = FastCRC64(args.dll_path)
    print(f"[Info] CRC-64 Engine: {'Native C++ DLL' if crc_engine.dll else 'Python Fallback'}")

    # 1. Load config
    config_path = ckpt_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    text_cfg = config.get("text_config", config)

    # 2. Build metadata JSON
    metadata = {
        "model_name": "Qwen/Qwen3.8-27B",
        "quant_scheme": f"INT4-G{args.group_size}-SYM",
        "tokenizer_type": "qwen3_8_tiktoken",
        "properties": {
            "architectures": str(config.get("architectures", ["Qwen3_5ForConditionalGeneration"])),
            "model_type": str(config.get("model_type", "qwen3_5")),
            "num_hidden_layers": str(text_cfg.get("num_hidden_layers", 64)),
            "hidden_size": str(text_cfg.get("hidden_size", 5120)),
            "intermediate_size": str(text_cfg.get("intermediate_size", 17408)),
            "num_attention_heads": str(text_cfg.get("num_attention_heads", 24)),
            "num_key_value_heads": str(text_cfg.get("num_key_value_heads", 4)),
            "head_dim": str(text_cfg.get("head_dim", 256)),
            "vocab_size": str(text_cfg.get("vocab_size", 248320)),
            "max_position_embeddings": str(text_cfg.get("max_position_embeddings", 262144)),
            "rms_norm_eps": str(text_cfg.get("rms_norm_eps", 1e-6)),
            "rope_theta": str(text_cfg.get("rope_parameters", {}).get("rope_theta", 10000000)),
            "full_attention_interval": str(text_cfg.get("full_attention_interval", 4)),
            "layer_types": json.dumps(text_cfg.get("layer_types", [])),
        }
    }
    meta_json_bytes = json.dumps(metadata, indent=2).encode("utf-8")

    # 3. Read index to find all shards
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)["weight_map"]
        shards_to_tensors = {}
        for tname, shard_file in idx_data.items():
            shards_to_tensors.setdefault(shard_file, []).append(tname)
        shard_files = sorted(shards_to_tensors.keys())
    else:
        # Single file
        single = ckpt_dir / "model.safetensors"
        shard_files = [single.name] if single.exists() else []

    print(f"[Info] Found {len(shard_files)} safetensors shards.")

    # Target parity sampling: choose sample across layers and tensor kinds
    parity_candidates = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.1.linear_attn.out_proj.weight",
        "model.language_model.layers.10.mlp.down_proj.weight",
        "model.language_model.layers.20.mlp.gate_proj.weight",
        "model.language_model.layers.30.self_attn.q_proj.weight",
        "model.language_model.layers.40.linear_attn.out_proj.weight",
        "model.language_model.layers.50.mlp.up_proj.weight",
        "model.language_model.layers.63.self_attn.o_proj.weight",
        "lm_head.weight"
    ]
    parity_results = []

    # Section table entries list: (name, type_tag, flags, offset, size, checksum)
    section_entries = []
    
    t_start = time.time()
    current_offset = 64 # Size of FileHeader

    with open(out_path, "wb") as out_f:
        # Write placeholder header (64 bytes)
        out_f.write(b"\0" * 64)

        # Write metadata JSON block
        meta_offset = current_offset
        meta_size = len(meta_json_bytes)
        out_f.write(meta_json_bytes)
        current_offset += meta_size

        def write_section(name: str, type_tag: int, data: bytes):
            nonlocal current_offset
            # Align to 64 bytes
            pad = (SECTION_ALIGNMENT - (current_offset % SECTION_ALIGNMENT)) % SECTION_ALIGNMENT
            if pad > 0:
                out_f.write(b"\0" * pad)
                current_offset += pad

            offset = current_offset
            size = len(data)
            checksum = crc_engine.calculate_bytes(data)

            out_f.write(data)
            current_offset += size

            section_entries.append((name, type_tag, 0, offset, size, checksum))

        # Iterate over shards and stream tensors
        total_tensors_processed = 0
        total_linear_quantized = 0

        for s_idx, shard_file in enumerate(shard_files):
            shard_path = ckpt_dir / shard_file
            print(f"[{s_idx+1}/{len(shard_files)}] Processing {shard_file} ...")
            tensors_in_shard = safetensors.torch.load_file(str(shard_path), device="cpu")

            for tname, tensor in tensors_in_shard.items():
                total_tensors_processed += 1

                if should_quantize_tensor(tname, tensor, args.group_size):
                    total_linear_quantized += 1
                    packed_bytes, scales_bytes, scales = quantize_int4_symmetric(tensor, args.group_size)

                    # Parity check if candidate
                    if tname in parity_candidates or len(parity_results) < args.parity_samples:
                        if tname not in [r["tensor_name"] for r in parity_results]:
                            # Run dequantization parity check oracle
                            dequant = dequantize_int4_symmetric(packed_bytes, scales, tensor.shape, args.group_size)
                            diff = torch.abs(dequant - tensor.float())
                            max_err = float(torch.max(diff))
                            mean_err = float(torch.mean(diff))
                            max_bound = float(torch.max(scales / 2.0)) + 1e-4
                            cos_sim = compute_cosine_similarity(tensor.float(), dequant)

                            # Sanity check: cosine similarity must be mathematically bounded to [-1.0, 1.0]
                            # Allow small floating-point tolerance (1e-4) for float64 accumulation
                            small_tolerance = 1e-4
                            assert abs(cos_sim) <= 1.0 + small_tolerance, (
                                f"FATAL: Cosine similarity out of mathematical bounds [-1.0, 1.0]: "
                                f"{cos_sim} for {tname} (violates sanity check)"
                            )
                            cos_sim = min(1.0, max(-1.0, cos_sim))
                            passed = bool(max_err <= max_bound and abs(cos_sim) <= 1.0 + small_tolerance and cos_sim >= 0.95)


                            parity_results.append({
                                "tensor_name": tname,
                                "shape": list(tensor.shape),
                                "group_size": args.group_size,
                                "max_error": max_err,
                                "theoretical_max_bound": max_bound,
                                "mean_error": mean_err,
                                "cosine_similarity": cos_sim,
                                "parity_passed": passed
                            })
                            print(f"    -> Parity Check [{tname}]: CosineSim={cos_sim:.5f}, MaxErr={max_err:.4f} <= Bound={max_bound:.4f} [{'PASS' if passed else 'FAIL'}]")

                    # Write INT4 weights
                    write_section(tname, SECTION_TYPE_TENSOR_WEIGHTS, packed_bytes)
                    # Write FP16 scales
                    write_section(f"{tname}.scales", SECTION_TYPE_TENSOR_SCALES, scales_bytes)
                else:
                    # Write unquantized tensor as raw blob (BF16 or float)
                    raw_bytes = get_tensor_bytes(tensor)
                    write_section(tname, SECTION_TYPE_RAW_BLOB, raw_bytes)

            del tensors_in_shard

        # Add tokenizer files if present
        tokenizer_path = ckpt_dir / "tokenizer.json"
        if tokenizer_path.exists():
            print("Packing tokenizer.json ...")
            with open(tokenizer_path, "rb") as f:
                tok_bytes = f.read()
            write_section("tokenizer.data", SECTION_TYPE_TOKENIZER_DATA, tok_bytes)

        chat_template_path = ckpt_dir / "chat_template.jinja"
        if chat_template_path.exists():
            print("Packing chat_template.jinja ...")
            with open(chat_template_path, "rb") as f:
                template_bytes = f.read()
            write_section("chat_template.jinja", SECTION_TYPE_CHAT_TEMPLATE, template_bytes)

        # Write Section Table (aligned to 64 bytes)
        pad = (SECTION_ALIGNMENT - (current_offset % SECTION_ALIGNMENT)) % SECTION_ALIGNMENT
        if pad > 0:
            out_f.write(b"\0" * pad)
            current_offset += pad

        table_offset = current_offset
        table_size = len(section_entries) * 128

        print(f"[Info] Writing Section Table: {len(section_entries)} entries ({table_size} bytes)...")
        for entry in section_entries:
            name, type_tag, flags, offset, size, checksum = entry
            name_bytes = name.encode("utf-8")[:MAX_SECTION_NAME_LEN - 1].ljust(MAX_SECTION_NAME_LEN, b"\0")
            packed_entry = struct.pack("<96sIIQQQ", name_bytes, type_tag, flags, offset, size, checksum)
            out_f.write(packed_entry)

        current_offset += table_size
        total_file_size = current_offset + 16 # Including 16-byte footer

        # Seek back and write finalized FileHeader
        out_f.seek(0)
        header_bytes = struct.pack(
            "<8sIIIIQQQQQ",
            MAGIC_BYTES,
            CURRENT_FORMAT_VERSION,
            64,  # header_size
            0,   # flags
            len(section_entries),
            meta_offset,
            meta_size,
            table_offset,
            table_size,
            total_file_size
        )
        assert len(header_bytes) == 64, f"Header size mismatch: {len(header_bytes)}"
        out_f.write(header_bytes)
        out_f.flush()

    print("[Info] Computing trailing file CRC-64 checksum...")
    t_crc0 = time.time()
    file_checksum = crc_engine.calculate_file(str(out_path), current_offset)
    t_crc1 = time.time()
    print(f"       Computed CRC-64: 0x{file_checksum:016x} in {t_crc1 - t_crc0:.2f}s")

    # Append 16-byte FileFooter
    with open(out_path, "ab") as out_f:
        footer_bytes = struct.pack("<Q8s", file_checksum, FOOTER_MAGIC_BYTES)
        out_f.write(footer_bytes)

    t_end = time.time()
    total_time = t_end - t_start
    final_size_gb = os.path.getsize(out_path) / (1024**3)

    # Save parity report
    with open(args.parity_report, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Qwen/Qwen3.8-27B",
            "quant_scheme": f"INT4-G{args.group_size}-SYM",
            "total_tensors_in_source": total_tensors_processed,
            "linear_tensors_quantized": total_linear_quantized,
            "artifact_total_sections": len(section_entries),
            "artifact_size_bytes": os.path.getsize(out_path),
            "artifact_size_gb": round(final_size_gb, 3),
            "conversion_time_seconds": round(total_time, 2),
            "parity_samples_verified": len(parity_results),
            "all_parity_checks_passed": all(r["parity_passed"] for r in parity_results),
            "samples": parity_results
        }, f, indent=2)

    print("\n==========================================================")
    print(" CONVERSION & PARITY VERIFICATION SUMMARY")
    print(f" Artifact Output:       {out_path}")
    print(f" Final File Size:       {final_size_gb:.2f} GB ({os.path.getsize(out_path)} bytes)")
    print(f" Total Sections:        {len(section_entries)}")
    print(f" Linear INT4 Quantized: {total_linear_quantized}")
    print(f" Parity Checks Tested:  {len(parity_results)} tensors")
    all_passed = all(r["parity_passed"] for r in parity_results)
    print(f" Parity Status:         {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    print(f" Total Conversion Time: {total_time:.1f} seconds")
    print(f" Parity Report Saved:   {args.parity_report}")
    print("==========================================================")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

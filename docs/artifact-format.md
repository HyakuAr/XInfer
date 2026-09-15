# `.xinfer` Artifact Container Format Specification

Version: 1.0  
Status: Milestone 1 (M1) Complete

---

## 1. Overview & Design Principles

The `.xinfer` format is a single, self-contained, binary artifact format designed for high-performance single-model inference on Intel Arc hardware (analogous to Neroued/ninfer's `.ninfer` format).

### Core Design Rules:
1. **Self-Contained:** Contains model weights, scales, zero-points, tokenizer data, chat template, and metadata in a single file.
2. **GPU/DMA Ready (64-byte Alignment):** All section data payloads begin on 64-byte aligned file offsets. This enables direct zero-copy memory mapping (`mmap`) and cache line-aligned DMA transfers via Level Zero / USM device allocations.
3. **Fail-Fast Integrity:** Includes per-section CRC-64 checksums and an overall trailing CRC-64 checksum with dedicated magic numbers at both the head and tail of the file. Truncated or corrupted files are detected immediately upon header/footer inspection.
4. **Zero Unnecessary Dependencies:** The container framing and reader/writer are implemented purely in standard C++20 with zero external library requirements.

---

## 2. Binary Layout

```
+-------------------------------------------------------------+
| FileHeader (64 bytes)                                       |
|   - Magic: "XINFER\0\0" (8 bytes)                           |
|   - Format Version: uint32 (1)                              |
|   - Header Size: uint32 (64)                                |
|   - Flags: uint32                                           |
|   - Section Count: uint32                                   |
|   - Metadata Offset & Size: uint64, uint64                  |
|   - Section Table Offset & Size: uint64, uint64             |
|   - Total File Size: uint64                                 |
|   - Reserved: 8 bytes                                       |
+-------------------------------------------------------------+
| Metadata JSON Block (Variable length, UTF-8 string)         |
|   - Model identity, quant configuration, tokenizer info     |
+-------------------------------------------------------------+
| [Padding to 64-byte boundary]                               |
+-------------------------------------------------------------+
| Section 0 Payload (64-byte aligned)                         |
| Section 1 Payload (64-byte aligned)                         |
| ...                                                         |
| Section N-1 Payload (64-byte aligned)                       |
+-------------------------------------------------------------+
| [Padding to 64-byte boundary]                               |
+-------------------------------------------------------------+
| Section Table (section_count * 96 bytes)                    |
|   - SectionEntry 0                                          |
|   - SectionEntry 1                                          |
|   - ...                                                     |
|   - SectionEntry N-1                                        |
+-------------------------------------------------------------+
| FileFooter (16 bytes)                                       |
|   - File CRC-64 Checksum: uint64 (bytes 0 to footer - 1)    |
|   - Footer Magic: "XINFFOOT" (8 bytes)                      |
+-------------------------------------------------------------+
```

---

## 3. Data Structures

All multi-byte numeric values are stored in little-endian byte order. Structs are packed with 1-byte alignment (`#pragma pack(push, 1)`).

### 3.1 FileHeader (64 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| `0x00` | `uint8_t[8]` | `magic` | Identifier bytes: `{'X', 'I', 'N', 'F', 'E', 'R', 0, 0}` |
| `0x08` | `uint32_t` | `format_version` | Container format version (currently `1`) |
| `0x0C` | `uint32_t` | `header_size` | Size of `FileHeader` in bytes (must be `64`) |
| `0x10` | `uint32_t` | `flags` | Reserved bitflags (currently `0`) |
| `0x14` | `uint32_t` | `section_count` | Number of entries in the section table |
| `0x18` | `uint64_t` | `metadata_offset` | Absolute file offset to metadata JSON block |
| `0x20` | `uint64_t` | `metadata_size` | Byte length of metadata JSON block |
| `0x28` | `uint64_t` | `section_table_offset`| Absolute file offset to Section Table |
| `0x30` | `uint64_t` | `section_table_size` | Byte length of Section Table (`section_count * 128`) |
| `0x38` | `uint64_t` | `total_file_size` | Total file size in bytes, including `FileFooter` |

### 3.2 SectionEntry (128 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| `0x00` | `char[96]` | `name` | Null-terminated section identifier (e.g. `model.layers.0.mlp.gate_up_proj.weight`) |
| `0x60` | `uint32_t` | `type_tag` | Tag identifying payload type (see Section Types) |
| `0x64` | `uint32_t` | `flags` | Alignment or compression flags |
| `0x68` | `uint64_t` | `offset` | Absolute file offset to payload (guaranteed 64-byte aligned) |
| `0x70` | `uint64_t` | `size` | Exact byte length of payload |
| `0x78` | `uint64_t` | `checksum` | CRC-64/ECMA-182 checksum of payload bytes |

### 3.3 FileFooter (16 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| `0x00` | `uint64_t` | `file_checksum` | CRC-64/ECMA-182 of all bytes from `0` to `total_file_size - 16` |
| `0x08` | `uint8_t[8]` | `footer_magic` | Footer magic bytes: `{'X', 'I', 'N', 'F', 'F', 'O', 'O', 'T'}` |

---

## 4. Section Type Tags

| Tag Value | Name | Description |
|---|---|---|
| `0x0000` | `RawBlob` | Arbitrary unformatted binary data |
| `0x0001` | `MetadataJson` | UTF-8 JSON document |
| `0x0002` | `TokenizerData` | Tokenizer vocabulary, merges, or tiktoken model file |
| `0x0003` | `TensorWeights` | Quantized linear/embedding weights (e.g. INT4 packed) |
| `0x0004` | `TensorScales` | Dequantization scales/biases/zero-points (FP16/BF16) |
| `0x0005` | `ChatTemplate` | Jinja2 or formatted chat template string |
| `0x00FF` | `Custom` | Custom/user-defined payload |

---

## 5. Checksum Algorithm

The container uses standard **CRC-64/ECMA-182**:
- Reflected polynomial: `0xC96C5795D7870F42ULL`
- Initial value: `0x0000000000000000ULL` (XOR-in `0xFFFFFFFFFFFFFFFFULL`)
- Final XOR: `0xFFFFFFFFFFFFFFFFULL`

Verification sequence:
1. Quick validation: Read `FileFooter`, verify `footer_magic == "XINFFOOT"`, verify `header.total_file_size == actual_file_size`.
2. Full validation: Compute CRC-64 over file up to footer offset and compare with `footer.file_checksum`.
3. Per-section validation: Compute CRC-64 of section payload when loaded and compare with `SectionEntry.checksum`.

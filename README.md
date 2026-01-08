# Task 3: Custom VLM Design for Industrial Quality Inspection

## Overview

This document presents a comprehensive design for a Vision-Language Model (VLM) solution for PCB (Printed Circuit Board) defect inspection in semiconductor manufacturing.

## Problem Statement

A semiconductor manufacturer needs an offline AI system for PCB inspection where:

- Inspectors ask natural language questions about defects
- System returns structured responses with locations and confidence scores
- Inference must complete in <2 seconds
- Available: 50,000 PCB images with defect bounding boxes (no QA pairs)
- Challenge: Generic VLMs hallucinate on domain-specific content

## Design Document

See [VLM_DESIGN_DOCUMENT.md](VLM_DESIGN_DOCUMENT.md) for the complete technical design addressing:

| Section | Topic                                                                 |
| ------- | --------------------------------------------------------------------- |
| **(A)** | Model Selection - Qwen-VL recommendation with justification           |
| **(B)** | Design Strategy - Architecture modifications for PCB inspection       |
| **(C)** | Optimization - Techniques for <2s inference (INT4 quantization, etc.) |
| **(D)** | Hallucination Mitigation - Constrained decoding, grounding loss, RAG  |
| **(E)** | Training Plan - Multi-stage approach with synthetic QA generation     |
| **(F)** | Validation - Counting accuracy, localization, hallucination metrics   |

## Key Recommendations

| Component              | Choice         |
| ---------------------- | -------------- |
| **Base Model**         | Qwen-VL 7B     |
| **Quantization**       | INT4 AWQ       |
| **Fine-tuning**        | LoRA/QLoRA     |
| **Inference Time**     | <2s on CPU     |
| **Hallucination Rate** | <0.1 per image |

## Architecture Diagram

```
PCB Image ──▶ Vision Encoder (ViT-G) ──▶ Cross-Modal Fusion
                                              │
Question ──▶ Text Tokenizer ─────────────────┘
                                              │
                                              ▼
                                    Language Decoder (Qwen-7B)
                                              │
                                              ▼
                                    Structured Output (JSON)
                                              │
                                              ▼
{"defects": [{"type": "solder_bridge", "location": [x,y,w,h], "confidence": 0.94}]}
```

## Files

- `VLM_DESIGN_DOCUMENT.md` - Complete technical design document
- `README.md` - This file

## License

MIT License

# Custom VLM Design for Industrial PCB Quality Inspection

## Executive Summary

This document presents a comprehensive design for a custom Vision-Language Model (VLM) solution tailored for offline PCB (Printed Circuit Board) defect inspection. The system enables inspectors to ask natural language questions about defects and receive structured responses with precise locations and confidence scores, all within a 2-second inference constraint.

---

## Problem Statement

**Scenario**: A semiconductor manufacturer requires an offline AI system for PCB inspection where:

- Inspectors ask natural language questions about defects
- System returns structured responses with locations and confidence scores
- Inference must complete in <2 seconds
- Available: 50,000 PCB images with defect bounding boxes (no QA pairs)
- Challenge: Generic VLMs hallucinate on domain-specific content

---

## A. Model Selection

### Recommended Architecture: **Qwen-VL (7B) with Custom Modifications**

| Model       | Size     | Inference Speed | Fine-tuning Flexibility | Licensing   | Localization  |
| ----------- | -------- | --------------- | ----------------------- | ----------- | ------------- |
| LLaVA-1.5   | 7B/13B   | Medium          | Good                    | Apache 2.0  | Moderate      |
| BLIP-2      | 3B-12B   | Fast            | Limited                 | BSD         | Poor          |
| **Qwen-VL** | 7B       | Fast            | Excellent               | Apache 2.0  | **Excellent** |
| Custom      | Variable | Optimized       | Full control            | Proprietary | Custom        |

### Why Qwen-VL?

1. **Native Bounding Box Support**: Qwen-VL has built-in grounding capabilities with `<box>` tokens, enabling precise coordinate output without architectural changes.

2. **Inference Speed**: 7B parameters allow quantization to INT4, achieving <2s inference on modern CPUs.

3. **Fine-tuning Flexibility**: Full LoRA/QLoRA support for domain adaptation with limited compute.

4. **Apache 2.0 License**: Suitable for commercial/industrial deployment.

5. **Multilingual**: Supports English and Chinese for global manufacturing facilities.

### Architectural Modifications for Precise Localization

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODIFIED QWEN-VL ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ PCB Image    │────▶│ Vision       │────▶│ Position     │    │
│  │ (1024×1024)  │     │ Encoder      │     │ Encoding     │    │
│  └──────────────┘     │ (ViT-G/14)   │     │ (2D Coords)  │    │
│                       └──────────────┘     └──────────────┘    │
│                              │                    │             │
│                              ▼                    ▼             │
│                       ┌──────────────────────────────┐         │
│                       │   Cross-Modal Fusion         │         │
│                       │   + Spatial Attention        │         │
│                       │   (Custom PCB-Aware Layer)   │         │
│                       └──────────────────────────────┘         │
│                              │                                  │
│  ┌──────────────┐           ▼                                  │
│  │ Question     │────▶┌──────────────┐                         │
│  │ "Count       │     │ Language     │                         │
│  │  solder      │     │ Decoder      │                         │
│  │  defects"    │     │ (Qwen-7B)    │                         │
│  └──────────────┘     └──────────────┘                         │
│                              │                                  │
│                              ▼                                  │
│                       ┌──────────────┐                         │
│                       │ Structured   │                         │
│                       │ Output Head  │                         │
│                       │ (JSON + Box) │                         │
│                       └──────────────┘                         │
│                              │                                  │
│                              ▼                                  │
│  Output: {"defects": [{"type": "solder_bridge",                │
│           "location": [x, y, w, h], "confidence": 0.94}]}      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modifications:

1. **Spatial Position Encoding**: Add learnable 2D position embeddings tied to image coordinates for accurate localization.

2. **Structured Output Head**: Custom decoder head that outputs JSON-formatted responses with bounding box coordinates.

3. **Defect-Specific Tokens**: Add special tokens for PCB defect types: `<solder_bridge>`, `<missing_component>`, `<scratch>`, etc.

---

## B. Design Strategy

### Vision Encoder Modifications

| Component         | Original     | Modified                   |
| ----------------- | ------------ | -------------------------- |
| Input Resolution  | 448×448      | **1024×1024** (PCB detail) |
| Patch Size        | 14×14        | **7×7** (finer features)   |
| Position Encoding | Learned      | **Sinusoidal + Learned**   |
| Feature Maps      | Single scale | **Multi-scale FPN**        |

**Rationale**: PCB defects are often tiny (solder bridges, hairline cracks). Higher resolution and smaller patches capture these details.

### Language Decoder Modifications

```python
# Custom output format for PCB inspection
class PCBOutputHead(nn.Module):
    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.defect_classifier = nn.Linear(hidden_dim, num_defect_types)
        self.box_regressor = nn.Linear(hidden_dim, 4)  # [x, y, w, h]
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Linear(hidden_dim, max_defects)

    def forward(self, hidden_states):
        return {
            "defect_type": self.defect_classifier(hidden_states),
            "bounding_box": self.box_regressor(hidden_states),
            "confidence": torch.sigmoid(self.confidence_head(hidden_states)),
            "count": self.count_head(hidden_states)
        }
```

### Cross-Modal Fusion Mechanism

**Original**: Simple cross-attention between visual and text tokens.

**Modified**: PCB-Aware Spatial Cross-Attention

```
Visual Features (V) ─────┐
                         ├──▶ Spatial Cross-Attention ──▶ Fused Features
Text Query (Q) ──────────┘          │
                                    │
                    ┌───────────────┘
                    ▼
            Region Proposal Network (RPN)
                    │
                    ▼
            Defect-Specific Attention Pooling
```

**Key Innovation**: The fusion layer learns to attend to defect-relevant regions based on the text query. For "count solder bridges", it focuses on solder joint areas.

---

## C. Optimization for <2s Inference

### Target Hardware Profile

- **CPU**: Intel Xeon / AMD EPYC (server) or i7/Ryzen (edge)
- **RAM**: 32GB minimum
- **Storage**: SSD for model loading
- **No GPU Required** (offline deployment constraint)

### Optimization Techniques

| Technique                 | Speedup | Quality Loss | Implementation    |
| ------------------------- | ------- | ------------ | ----------------- |
| **INT4 Quantization**     | 4-6x    | <1% accuracy | AWQ / GPTQ        |
| **KV-Cache Optimization** | 2x      | None         | PagedAttention    |
| **Model Pruning**         | 1.5x    | 2-3%         | Magnitude pruning |
| **LoRA Merging**          | 1.2x    | None         | Merge adapters    |
| **Flash Attention**       | 2x      | None         | Memory efficient  |
| **ONNX Runtime**          | 1.5x    | None         | CPU optimization  |

### Quantization Strategy

```python
# AWQ Quantization Configuration
quantization_config = {
    "method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "calibration_dataset": "pcb_defect_samples",  # Domain-specific
    "calibration_samples": 512
}
```

### Inference Pipeline Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                 OPTIMIZED INFERENCE PIPELINE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image Input ──▶ Preprocessor (SIMD) ──▶ Vision Encoder    │
│                      [50ms]                  [400ms]        │
│                                                             │
│  Text Query ──▶ Tokenizer ──▶ Embedding Lookup              │
│                   [10ms]         [20ms]                     │
│                                                             │
│  Fused Features ──▶ LLM Decoder (INT4) ──▶ Output Parser   │
│                          [800ms]              [50ms]        │
│                                                             │
│  Total: ~1.3s (with buffer for variability)                │
└─────────────────────────────────────────────────────────────┘
```

### Memory Optimization

- **Model Sharding**: Split model across CPU cores
- **Activation Checkpointing**: Trade compute for memory
- **Dynamic Batching**: Process multiple queries efficiently
- **Lazy Loading**: Load vision encoder only when needed

---

## D. Hallucination Mitigation

### Why Generic VLMs Hallucinate on PCB

1. **Domain Gap**: Pre-trained on natural images, not industrial PCB
2. **Rare Vocabulary**: "Solder bridge", "tombstoning" not in training data
3. **Precise Counting Required**: VLMs struggle with exact counts
4. **Spatial Confusion**: Similar-looking components across PCB

### Mitigation Strategies

#### 1. Constrained Decoding

```python
# Force output to follow structured schema
class ConstrainedDecoder:
    def __init__(self):
        self.valid_defect_types = [
            "solder_bridge", "missing_component", "scratch",
            "misalignment", "cold_joint", "tombstone", "none"
        ]

    def decode(self, logits, schema):
        # Only allow tokens that match expected JSON structure
        # Prevents free-form hallucinated text
        return constrained_beam_search(logits, schema)
```

#### 2. Grounding Loss Function

```python
def grounding_loss(pred_boxes, pred_confidence, gt_boxes):
    """
    Penalize predictions that don't match ground truth locations.
    Forces model to only report defects it can precisely locate.
    """
    # IoU-based matching
    iou = box_iou(pred_boxes, gt_boxes)
    matched = iou > 0.5

    # High confidence on unmatched boxes = hallucination
    hallucination_penalty = pred_confidence[~matched].mean()

    # Low confidence on matched boxes = missed detection
    detection_loss = (1 - pred_confidence[matched]).mean()

    return detection_loss + 2.0 * hallucination_penalty  # Weight hallucination higher
```

#### 3. Retrieval-Augmented Generation (RAG)

```
Query: "Are there any solder bridges?"
          │
          ▼
┌─────────────────────────────────────────────┐
│           RAG Pipeline                       │
├─────────────────────────────────────────────┤
│ 1. Run defect detector on image             │
│ 2. Retrieve similar defects from database   │
│ 3. Ground response in detected evidence     │
│ 4. Only report defects with visual proof    │
└─────────────────────────────────────────────┘
          │
          ▼
Response: "Yes, 2 solder bridges detected at
          [(234, 456), (789, 123)] with
          confidence [0.92, 0.87]"
```

#### 4. Uncertainty Quantification

```python
class UncertaintyAwareVLM:
    def forward(self, image, query, num_samples=5):
        # Monte Carlo dropout for uncertainty
        outputs = []
        for _ in range(num_samples):
            outputs.append(self.model(image, query, dropout=True))

        # High variance = uncertain = don't report
        mean_pred = torch.mean(outputs, dim=0)
        variance = torch.var(outputs, dim=0)

        # Filter uncertain predictions
        confident_mask = variance < self.uncertainty_threshold
        return mean_pred[confident_mask]
```

#### 5. Negative Training Examples

Include training examples with:

- Images with NO defects → Model must answer "No defects found"
- Tricky questions about non-existent defects → Model must refuse to hallucinate

---

## E. Training Plan

### Multi-Stage Training Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Domain Adaptation (Vision Encoder)                   │
│  ──────────────────────────────────────────                    │
│  • Freeze LLM, train vision encoder on PCB images              │
│  • Contrastive learning with PCB image pairs                   │
│  • Duration: 5 epochs, ~8 hours on 4×A100                      │
│                                                                 │
│  Stage 2: Defect Detection Pre-training                        │
│  ──────────────────────────────────────────                    │
│  • Train on 50K images with bounding box supervision           │
│  • Detection head: classify + localize defects                 │
│  • Duration: 10 epochs, ~16 hours                              │
│                                                                 │
│  Stage 3: QA Pair Generation (Synthetic)                       │
│  ──────────────────────────────────────────                    │
│  • Generate 500K QA pairs from detection annotations           │
│  • Template-based + LLM paraphrasing                           │
│  • Duration: Data generation ~4 hours                          │
│                                                                 │
│  Stage 4: VLM Fine-tuning (LoRA)                               │
│  ──────────────────────────────────────────                    │
│  • LoRA fine-tune LLM on synthetic QA pairs                    │
│  • Grounding loss + answer accuracy loss                       │
│  • Duration: 3 epochs, ~12 hours                               │
│                                                                 │
│  Stage 5: Hallucination Reduction                              │
│  ──────────────────────────────────────────                    │
│  • DPO/RLHF with human feedback on edge cases                  │
│  • Hard negative mining for failure cases                      │
│  • Duration: 2 epochs, ~8 hours                                │
└─────────────────────────────────────────────────────────────────┘
```

### QA Pair Generation Strategy

**Template Categories**:

```python
qa_templates = {
    "counting": [
        ("How many {defect_type} defects are in this image?", "{count}"),
        ("Count the {defect_type} issues.", "There are {count} {defect_type} defects."),
    ],
    "localization": [
        ("Where is the {defect_type}?", "At coordinates ({x}, {y})."),
        ("Locate all defects.", "[{defect_list}]"),
    ],
    "classification": [
        ("What type of defect is at ({x}, {y})?", "{defect_type}"),
        ("Describe the defects in this PCB.", "{defect_description}"),
    ],
    "binary": [
        ("Is this PCB defective?", "{yes_no}"),
        ("Are there any {defect_type} defects?", "{yes_no}"),
    ],
    "severity": [
        ("Rate the severity of defects.", "{severity_level}"),
        ("Which defect is most critical?", "{critical_defect}"),
    ]
}
```

**Paraphrasing with LLM**:

```python
def generate_qa_pairs(image_annotation):
    base_qa = template_based_generation(image_annotation)

    # Use GPT-4/Claude to paraphrase for diversity
    paraphrased = []
    for q, a in base_qa:
        new_q = llm_paraphrase(q, style="inspector_query")
        paraphrased.append((new_q, a))

    return base_qa + paraphrased
```

### Data Augmentation

| Augmentation            | Purpose                 | Probability |
| ----------------------- | ----------------------- | ----------- |
| **Rotation (±15°)**     | Handle PCB orientation  | 30%         |
| **Brightness/Contrast** | Lighting variation      | 40%         |
| **Gaussian Noise**      | Sensor noise simulation | 20%         |
| **Crop & Resize**       | Scale invariance        | 30%         |
| **Color Jitter**        | Camera variation        | 25%         |
| **Synthetic Defects**   | Defect augmentation     | 15%         |

### Evaluation Metrics

| Metric                 | Description                 | Target |
| ---------------------- | --------------------------- | ------ |
| **Counting Accuracy**  | Exact match on defect count | >90%   |
| **mAP@0.5**            | Detection precision         | >85%   |
| **Answer Accuracy**    | Correct QA responses        | >92%   |
| **Hallucination Rate** | False positives per image   | <0.1   |
| **Inference Time**     | End-to-end latency          | <2s    |
| **Localization Error** | Mean pixel error            | <10px  |

---

## F. Validation Framework

### 1. Counting Accuracy Validation

```python
def validate_counting(model, test_set):
    exact_match = 0
    off_by_one = 0

    for image, gt_count in test_set:
        pred_count = model.count_defects(image)

        if pred_count == gt_count:
            exact_match += 1
        elif abs(pred_count - gt_count) == 1:
            off_by_one += 1

    return {
        "exact_accuracy": exact_match / len(test_set),
        "off_by_one_accuracy": (exact_match + off_by_one) / len(test_set)
    }
```

**Validation Protocol**:

- Hold-out test set: 5,000 images (10% of dataset)
- Cross-validation: 5-fold for robustness
- Edge cases: Images with 0, 1, 5, 10+ defects

### 2. Localization Precision Validation

```python
def validate_localization(model, test_set, iou_threshold=0.5):
    all_predictions = []
    all_ground_truth = []

    for image, gt_boxes in test_set:
        pred_boxes = model.detect_defects(image)
        all_predictions.append(pred_boxes)
        all_ground_truth.append(gt_boxes)

    # Calculate mAP
    mAP = calculate_map(all_predictions, all_ground_truth, iou_threshold)

    # Calculate center point error
    center_errors = []
    for pred, gt in zip(all_predictions, all_ground_truth):
        matched_pred, matched_gt = match_boxes(pred, gt)
        for p, g in zip(matched_pred, matched_gt):
            error = np.sqrt((p.cx - g.cx)**2 + (p.cy - g.cy)**2)
            center_errors.append(error)

    return {
        "mAP@0.5": mAP,
        "mean_center_error_px": np.mean(center_errors),
        "median_center_error_px": np.median(center_errors)
    }
```

### 3. Hallucination Rate Validation

```python
def validate_hallucination(model, clean_test_set, adversarial_set):
    """
    Test on:
    1. Clean images with no defects
    2. Adversarial questions about non-existent defects
    """

    # Test 1: Clean images
    false_positives = 0
    for image in clean_test_set:
        response = model.query(image, "What defects are present?")
        if response != "No defects found":
            false_positives += 1

    # Test 2: Adversarial questions
    hallucinations = 0
    for image, fake_defect in adversarial_set:
        response = model.query(image, f"Where is the {fake_defect}?")
        if "not found" not in response.lower():
            hallucinations += 1

    return {
        "false_positive_rate": false_positives / len(clean_test_set),
        "adversarial_hallucination_rate": hallucinations / len(adversarial_set),
        "overall_hallucination_rate": (false_positives + hallucinations) /
                                      (len(clean_test_set) + len(adversarial_set))
    }
```

### 4. End-to-End Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Test Set: 5,000 images                                         │
│  ├── 3,000 with defects (varied count)                         │
│  ├── 1,500 clean (no defects)                                  │
│  └── 500 adversarial (tricky questions)                        │
│                                                                 │
│  Metrics Tracked:                                               │
│  ├── Counting Accuracy: >90% exact, >95% off-by-one           │
│  ├── mAP@0.5: >85%                                             │
│  ├── Center Error: <10 pixels                                  │
│  ├── Hallucination Rate: <0.1 per image                        │
│  ├── Inference Time: <2s (P95)                                 │
│  └── Answer Accuracy: >92%                                     │
│                                                                 │
│  Continuous Validation:                                         │
│  ├── A/B testing with production queries                       │
│  ├── Human inspector feedback loop                             │
│  └── Edge case collection for retraining                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Component                    | Recommendation                              |
| ---------------------------- | ------------------------------------------- |
| **Base Model**               | Qwen-VL 7B                                  |
| **Vision Encoder**           | ViT-G/14 @ 1024×1024, 7×7 patches           |
| **Language Decoder**         | Qwen-7B with structured output head         |
| **Quantization**             | INT4 AWQ                                    |
| **Optimization**             | LoRA fine-tuning + Flash Attention          |
| **Hallucination Mitigation** | Constrained decoding + RAG + Grounding loss |
| **Inference Target**         | <2s on CPU (Intel Xeon)                     |
| **Training Data**            | 50K images → 500K synthetic QA pairs        |

This architecture balances accuracy, speed, and reliability for industrial deployment while maintaining strict hallucination controls critical for quality inspection applications.

---

## References

1. Bai et al., "Qwen-VL: A Versatile Vision-Language Model", 2023
2. Liu et al., "LLaVA: Visual Instruction Tuning", NeurIPS 2023
3. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training", ICML 2023
4. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023
5. Lin et al., "AWQ: Activation-aware Weight Quantization", 2023

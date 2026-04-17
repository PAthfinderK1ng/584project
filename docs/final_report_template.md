# Final Report Template

## Title

Evaluating LoRA Adapter Merging for Multi-Task Reasoning and Code Generation in Large Language Models

## 1. Introduction

- Briefly motivate PEFT and LoRA.
- Explain why multi-task adapter serving is operationally expensive.
- State the central question: can adapter merging preserve single-task performance without retraining?

## 2. Research Questions

1. Can math and code LoRA adapters be merged effectively without retraining the base model?
2. How do Linear, TIES, and DARE compare on task performance and cross-task interference?
3. What does the observed performance suggest about complementary vs conflicting task vectors?

## 3. Experimental Setup

### 3.1 Base model

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: 4-bit NF4
- LoRA rank / alpha / target modules
- Hardware used

### 3.2 Datasets

- Math training: GSM8K train
- Math evaluation: GSM8K test, MATH-500
- Code training: MBPP sanitized train
- Code evaluation: HumanEval

### 3.3 Conditions

- Base model
- Math adapter
- Code adapter
- Merged Linear
- Merged TIES
- Merged DARE

## 4. Implementation

- Describe the training pipeline.
- Describe adapter merging in adapter parameter space.
- Describe answer extraction, symbolic normalization, and HumanEval execution.

## 5. Results

### 5.1 Main performance table

Insert `results/analysis/summary_table.csv`.

### 5.2 Delta-performance table

Insert `results/analysis/delta_table.csv`.

### 5.3 Figures

- `results/analysis/math_performance.png`
- `results/analysis/code_performance.png`

## 6. Discussion

- Which merge method preserved math best?
- Which merge method preserved code best?
- Was there asymmetric interference?
- Was the degradation acceptable for practical deployment?

## 7. Efficiency Analysis

- Trainable parameter count
- Peak memory
- Training time
- Inference-time deployment tradeoff: multiple adapters vs one merged adapter

## 8. Limitations

- Limited run budget
- Single base model
- Small number of task domains
- Potential sensitivity to prompt format and evaluation heuristics

## 9. Conclusion

- Summarize which merge method worked best.
- State whether merging is a viable deployment strategy under your results.
- Suggest next steps such as broader task coverage, weight search, or multi-adapter composition.

## Appendix

- Training hyperparameters
- Example generations
- Error analysis case studies


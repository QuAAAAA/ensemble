# NYCU Speech Lab at SemEval-2026 Task 3: Heterogeneous Model Ensemble with Adaptive Weighted Voting for Dimensional Aspect Sentiment Quadruplet Extraction

This repository contains the codebase accompanying the paper *NYCU speech lab at SemEval-2026 Task 3: Ensemble Is All You Need*. We introduce a robust ensemble method for Dimensional Aspect-Based Sentiment Analysis (DimABSA), combining predictions from diverse Large Language Models (LLMs) to enhance performance.

Our method **Weighted Ensemble with Voting** achieves superior performance by aggregating outputs from multiple state-of-the-art models including Qwen, Gemma, Llama, and RoBERTa.

## 🏆 Leaderboard

**Official evaluation results and rankings for SemEval-2026 Task 3 (Codabench Test Set). Our system secured the 1st rank in both domains.**

| Domain (Rank 1st) |    CF1     | CPREC. | CREC.  |   CTP   |   TP   |  FP   |   FN   |
| :---------------: | :--------: | :----: | :----: | :-----: | :----: | :---: | :----: |
|  ZHO-Restaurant   | **0.5521** | 0.5951 | 0.5148 | 1472.98 |  1561  |  914  |  1300  |
|    ZHO-Laptop     | **0.4824** | 0.5816 | 0.4121 | 793.31  |  826   |  538  |  1099  |
|    **AVERAGE**    | **0.5172** | 0.5884 | 0.4635 | 1133.14 | 1193.5 | 726.0 | 1199.5 |

## Models

We utilize predictions from a diverse set of models to maximize coverage and accuracy:

*   **Qwen Models**: Qwen 32B (various checkpoints), Qwen 14B.
*   **Gemma Models**: Gemma 3 (3e, 4e, 5e).
*   **Llama Models**: Llama (3e, best loss).
*   **RoBERTa**: Fine-tuned RoBERTa model.
*   **Closed-Source Models**: GPT, Gemini.

Each model contributes to the final decision based on a normalized weight derived from its validation performance.

---

## Repository Structure

*   `ensemble.py` – The main script that performs the weighted ensemble, voting, and result generation.
*   `test/` – Directory containing the input JSONL files with predictions from individual models.
*   `output/` – Directory where the final ensemble results (`pred_zho_restaurant.jsonl` and `.zip`) are stored.

### Setup Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/QuAAAAA/ensemble.git
cd ensemble
```

#### 2. Prepare Input Files

Ensure that the model prediction files are placed in the `test/` directory. The default filenames include:
*   `qwen32B-3e.jsonl`, `qwen32B_best_loss.jsonl`, `qwen32B_best_cF1.jsonl`
*   `qwen14B-3e.jsonl`, `qwen14B_best_loss.jsonl`
*   `robertwwm.jsonl`
*   `gemma-3e.jsonl`, `gemma-4e.jsonl`, `gemma-5e.jsonl`
*   `llama-3e.jsonl`, `llama_best_loss.jsonl`
*   `gpt.jsonl`, `gemini.jsonl`

#### 3. Run the Ensemble Script

Execute the main python script to generate the ensemble results:

```bash
python ensemble.py
```

The script will:
1.  Load all available prediction files.
2.  Apply the weighted voting and VA fusion logic.
3.  Generate the final output in `output/pred_zho_restaurant.jsonl` and compress it into a zip file.

---

## Citation Information

If you use this codebase in your work, please cite:

```bibtex
@misc{nycuspeechlab2026semeval,
  title={NYCU speech lab at SemEval-2026 Task 3: Ensemble Is All You Need}, 
  author={Hao-Chun Hsieh and Cheng-En Wu and Yuan-Fu Liao},
  year={2026},
  howpublished={SemEval-2026 Task 3 Submission},
  institution={Institute of Artificial Intelligence Innovation, National Yang Ming Chiao Tung University}
}
```

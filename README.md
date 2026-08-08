# Multimodal Consistency Reasoning Framework for Fake News Detection

A multimodal fake news detection framework that goes beyond binary classification by reasoning about **consistency between text, image, and metadata**. Built and evaluated on the [Fakeddit dataset](https://github.com/entitize/Fakeddit), the framework combines a RoBERTa + ResNet-50 classifier, a CLIP-based text-image alignment model, OCR-based text consistency checking, an XGBoost metadata model, and a learned fusion layer — all tied together by a novel **Consistency Score** that flags samples where modalities disagree.

## 🔥 Key Features

- **6-way classification** of news posts: `TRUE`, `SATIRE`, `FALSE_CONNECTION`, `IMPOSTER`, `MANIPULATED`, `MISLEADING`
- **RoBERTa-base + ResNet-50** multimodal classifier with weighted cross-entropy for class imbalance
- **CLIP ViT-B/32** for semantic text-image alignment scoring
- **OCR-based consistency check** (EasyOCR + Sentence-Transformers) comparing embedded image text against the headline
- **XGBoost** model on image metadata features (color stats, edge stats, OCR signal, etc.)
- **Learned MLP fusion layer** combining CLIP and metadata model outputs (replacing a fixed-alpha blend)
- **Consistency Score** — a novel score combining text-image alignment, OCR similarity, and metadata confidence to flag inconsistent / likely-fake samples
- Cosine learning-rate scheduling with early stopping, gradient clipping, and stronger image augmentation
- Full evaluation suite: classification reports, confusion matrices, threshold analysis, and per-component (branch-wise) performance comparison
- Publication-ready figures for paper writeups

## 📊 Pipeline Overview

| Component | Model / Method | Role |
|---|---|---|
| Text-Image Alignment | CLIP ViT-B/32 | Semantic cross-modal agreement |
| OCR-Text Consistency | EasyOCR + MiniLM | Embedded text vs. headline match |
| Metadata Reasoning | XGBoost | Visual feature-based signal |
| Cross-Modal Fusion | Learned MLP | Adaptive combination of modalities |
| Consistency Verification | Threshold scoring | Flags inconsistent samples |

## 🗂️ Dataset

This project uses the **Fakeddit** multimodal dataset (Reddit posts with text, image, and 6-way fine-grained labels), downloaded automatically via [`kagglehub`](https://github.com/Kaggle/kagglehub) (`vanshikavmittal/fakeddit-dataset`). No manual download is required — the notebook handles fetching, caching, and image downloading.

## 📁 Repository Structure

```
.
├── notebooks/
│   └── Fake_News_Paper_Code.ipynb   # Full end-to-end pipeline (data → training → fusion → consistency reasoning)
├── outputs/                          # Generated results (CSVs, figures) — populated after running
├── checkpoints/                      # Saved model weights — populated after running
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

The notebook was developed for a Google Colab GPU runtime. It works locally as well, provided you have a CUDA-capable GPU (recommended) or are prepared to run on CPU (slower).

## 🚀 Usage

1. Open `notebooks/Fake_News_Paper_Code.ipynb` in Jupyter or Google Colab.
2. Run the cells sequentially — the notebook is organized into clearly labeled steps (environment setup, data download, preprocessing, model training, fusion, consistency scoring, and evaluation).
3. A Kaggle account/API key is required for `kagglehub` to download the Fakeddit dataset.
4. Outputs (model checkpoints, result CSVs, and figures) are saved automatically to the `outputs/` and `checkpoints/` directories.

## 📈 Results

The notebook produces:
- Per-model accuracy and macro-F1 comparison (RoBERTa+ResNet50, CLIP, XGBoost, Learned Fusion)
- Confusion matrices for each model
- Consistency score distributions and threshold-vs-accuracy analysis
- A component-wise performance breakdown isolating the individual contribution of each evidence branch (semantic, OCR, metadata) prior to fusion

## 📝 Notes

- Random seeds are fixed (`SEED = 42`) throughout for reproducibility.
- Class imbalance is handled via balanced class weighting in the loss function.
- This repository accompanies a research paper on multimodal consistency reasoning for fake news detection; figure-generation cells are included for publication-quality plots.
- All reported results are obtained on a held-out validation split used during training; no independently held-out test partition is currently reserved (see the paper's Limitations section).

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- [Fakeddit dataset](https://github.com/entitize/Fakeddit) authors
- Hugging Face `transformers` and `sentence-transformers`
- OpenAI CLIP
- EasyOCR

# Cat & Dog Image Classification Pipeline 🐱🐶

A modular computer vision pipeline built with **PyTorch** and **MobileNetV2**, implementing Clean Architecture principles and Transfer Learning.

---

## Highlights & Results
- **Validation Accuracy**: `95.24%` on the evaluation set (10 Epochs).
- **Architecture**: MobileNetV2 (frozen feature extractor) + Custom Classification Head.
- **Tech Stack**: PyTorch, Python, `uv` package manager, Pathlib,Git,DVC.

---

## 📁 Project Structure
```text
.
├── notebooks/
│   └── eda.ipynb       # Exploratory Data Analysis & data insights
├── src/
│   ├── dataset.py      # Custom Dataset handling & DataLoaders
│   ├── model.py        # MobileNetV2 architecture & head customization
│   ├── split.py        # Dataset splitting script (Train/Val)
│   ├── train.py        # Training script
│   └── eval.py         # Evaluation loop & metrics computation
├── pyproject.toml      # Project dependencies & configurations
└── README.md

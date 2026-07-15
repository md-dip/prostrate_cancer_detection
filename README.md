# Prostate Cancer Detection  Web Interface

Flask web app for the HybridProstateCancerNet (EfficientNet-B4 + GAT + GCN, SE-gated 832-dim fusion). Drag and drop an H&E histopathology image → get Benign/Malignant prediction with TTA-averaged confidence, plus Sobel / Canny / morphology edge maps.

## Project layout

```
prostate_app/
├── app.py             
├── model.py             
├── inference.py       
├── templates/
│   └── index.html       
├── checkpoints/
│   └── hybrid_best.pth  
├── requirements.txt
├── Procfile            
├── railway.toml
└── .gitignore
```

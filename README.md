# Prostate Cancer Detection — Web Interface

Flask web app for the HybridProstateCancerNet (EfficientNet-B4 + GAT + GCN, SE-gated 832-dim fusion). Drag and drop an H&E histopathology image → get Benign/Malignant prediction with TTA-averaged confidence, plus Sobel / Canny / morphology edge maps.

## Project layout

```
prostate_app/
├── app.py               # Flask server
├── model.py             # HybridProstateCancerNet (matches notebook)
├── inference.py         # SLIC graph + TTA×5 + edge maps
├── templates/
│   └── index.html       # drag-drop UI
├── checkpoints/
│   └── hybrid_best.pth  # ← PUT YOUR TRAINED WEIGHTS HERE
├── requirements.txt
├── Procfile             # for Railway
├── railway.toml
└── .gitignore
```

## 1. Put your checkpoint in place

Drop your best trained model into `checkpoints/hybrid_best.pth`.
Either a raw state_dict or `{'model_state_dict': ...}` works — the loader handles both.

Change the path with the env var if needed:
```bash
export CHECKPOINT_PATH=checkpoints/my_other.pth
```

## 2. Run locally

```bash
cd prostate_app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install torch-geometric extras to match torch 2.3.1 (CPU):
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cpu.html

python app.py
```

Open http://localhost:5000

For GPU locally, install the CUDA build of torch first (uninstall the CPU one), then reinstall the matching PyG wheels.

## 3. Smoke-test

```bash
curl http://localhost:5000/health
# {"status":"ok","device":"cpu","checkpoint_loaded":true}

curl -F "image=@sample.jpg" http://localhost:5000/predict | jq
```

Expected JSON:
```json
{
  "prediction": {
    "label": "Malignant",
    "confidence": 98.7,
    "prob_benign": 1.3,
    "prob_malignant": 98.7,
    "tta_individual": [[...],[...],...]
  },
  "images": { "original": "...", "sobel": "...", "canny": "...", "morph": "..." }
}
```

## 4. Deploy to Railway

```bash
# 1. Create a GitHub repo with this folder, push it.
# 2. Add checkpoints/hybrid_best.pth to the repo (Git LFS recommended — file is ~70 MB).
# 3. On railway.app: New Project → Deploy from GitHub repo → pick this repo.
# 4. Railway auto-detects Procfile + railway.toml. First deploy ~5-8 min (torch is heavy).
# 5. Go to Settings → Networking → Generate Domain.
```

### Railway gotchas

- **Memory**: torch + torch-geometric + EfficientNet-B4 + SLIC needs ≥ 2 GB RAM. The free Hobby plan should be fine but watch for OOM kills on cold start.
- **Cold-start timeout**: first request after idle loads the model (~10-20 s on CPU). `Procfile` sets `--timeout 180` to allow this.
- **PyG extras**: if Railway build fails on `torch-scatter`, set the Nixpacks env var `NIXPACKS_PIP_INSTALL_FLAGS=--no-build-isolation` in Railway Settings.
- **Image size**: keep uploads small. Cap is set to 16 MB in `app.py`.

## Notes on what runs internally

- **TTA**: same 5 transforms used during test in your notebook (identity / hflip / center-crop / 90° rotate / vflip). Final probability is the mean.
- **SLIC graph**: rebuilt per upload with identical hyperparameters (`n_segments=75, k=5`, 8-dim node features) so the GCN branch sees the same distribution it was trained on.
- **GAT branch caveat**: at batch=1 the fully-connected batch graph has no edges (only self-loops added by GATConv). The app duplicates the input to batch=2 so the GAT branch receives a real edge during inference (and so BatchNorm in the head doesn't error). This matches how a 2-image minibatch would behave at test time.
- **Edge maps**: Sobel (Gx² + Gy²), Canny (80/180 thresholds), morphology = dilate(Canny, 2×2) → close(3×3). Visual only — not a model input.

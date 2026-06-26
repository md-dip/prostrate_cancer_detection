"""
Flask app for HybridProstateCancerNet inference.

Endpoints:
  GET  /            → drag-drop UI
  POST /predict     → multipart form-data with 'image' file → JSON result

Checkpoint loading:
  1. If a local file exists at CHECKPOINT_PATH (default: checkpoints/hybrid_best.pth),
     use it directly.
  2. Otherwise, if HF_REPO_ID is set, download from Hugging Face Hub.
     Set:
       HF_REPO_ID  = "yourname/prostate-hybrid-cnn-gnn"
       HF_FILENAME = "hybrid_best.pth"        (optional, this is the default)
       HUGGINGFACE_HUB_TOKEN = "hf_xxx..."    (only needed for private repos)

  Handles checkpoints saved as:
    - {'model':            state_dict, ...}   ← your notebook format
    - {'model_state_dict': state_dict, ...}
    - {'state_dict':       state_dict, ...}
    - raw state_dict
  Also strips 'module.' prefix from DataParallel-saved checkpoints.
"""
import os
import io
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch

from model import HybridProstateCancerNet
from inference import predict, compute_edge_maps, check_he_image, DEVICE

# ── App setup ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload cap

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/hybrid_best.pth')
HF_REPO_ID      = os.environ.get('HF_REPO_ID', '')          # e.g. "yourname/prostate-hybrid-cnn-gnn"
HF_FILENAME     = os.environ.get('HF_FILENAME', 'hybrid_best.pth')

# ── Download checkpoint from Hugging Face if not present locally ────
if not os.path.exists(CHECKPOINT_PATH) and HF_REPO_ID:
    try:
        from huggingface_hub import hf_hub_download
        print(f'[startup] no local checkpoint; downloading from HF: {HF_REPO_ID}/{HF_FILENAME}')
        os.makedirs(os.path.dirname(CHECKPOINT_PATH) or '.', exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir='checkpoints',
            local_dir_use_symlinks=False,
        )
        CHECKPOINT_PATH = downloaded
        print(f'[startup] downloaded to: {downloaded}')
    except Exception as e:
        print(f'[startup] ERROR — Hugging Face download failed: {e}')
        print('[startup] will start without weights — predictions will be random')

# ── Load model once at startup ──────────────────────────────────────
print(f'[startup] device: {DEVICE}')
print(f'[startup] loading checkpoint: {CHECKPOINT_PATH}')

model = HybridProstateCancerNet(num_classes=2, hidden=256, gat_heads=4, dropout=0.4)

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Pull out the actual state_dict from whatever wrapper format was used
    if isinstance(ckpt, dict):
        state = None
        for k in ('model', 'model_state_dict', 'state_dict'):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                print(f'[startup] found weights under key: "{k}"')
                if 'epoch'    in ckpt: print(f'[startup]   saved at epoch: {ckpt["epoch"]}')
                if 'best_auc' in ckpt: print(f'[startup]   best val AUC:  {ckpt["best_auc"]:.4f}')
                if 'best_acc' in ckpt: print(f'[startup]   best val acc:  {ckpt["best_acc"]:.4f}')
                break
        if state is None:
            state = ckpt
            print('[startup] using checkpoint dict as raw state_dict')
    else:
        state = ckpt

    # Strip 'module.' prefix if the model was saved via DataParallel
    new_state = {}
    stripped = 0
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
            stripped += 1
        else:
            new_state[k] = v
    if stripped:
        print(f'[startup] stripped "module." prefix from {stripped} keys')
    state = new_state

    # strict=True so any mismatch fails loudly instead of silently random-init'ing
    try:
        model.load_state_dict(state, strict=True)
        print('[startup] checkpoint loaded ✓ (strict=True passed)')
    except RuntimeError as e:
        print('[startup] strict load FAILED — falling back to strict=False')
        print(f'[startup] error was: {str(e)[:500]}')
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[startup] missing keys    ({len(missing)}): {missing[:10]}')
        print(f'[startup] unexpected keys ({len(unexpected)}): {unexpected[:10]}')
        print('[startup] ⚠️  PARTIAL LOAD — predictions may be unreliable. '
              'Paste this output to debug.')
else:
    print(f'[startup] WARNING — no checkpoint at {CHECKPOINT_PATH}. '
          f'Predictions will be RANDOM. Set HF_REPO_ID or put a .pth in checkpoints/.')

model.to(DEVICE)
model.eval()


# ── Routes ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'checkpoint_loaded': os.path.exists(CHECKPOINT_PATH),
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image file in request (field name must be "image")'}), 400

        f = request.files['image']
        if not f.filename:
            return jsonify({'error': 'empty filename'}), 400

        img = Image.open(io.BytesIO(f.read())).convert('RGB')

        # OOD filter — reject non-H&E uploads before running the model
        is_he, reason, he_stats = check_he_image(img)
        if not is_he:
            return jsonify({
                'error': 'invalid_image',
                'message': reason,
                'stats': he_stats,
            }), 400

        # Run inference + edge maps
        result = predict(model, img)
        edges = compute_edge_maps(img)

        return jsonify({
            'prediction': {
                'label': result['label'],
                'confidence': round(result['confidence'] * 100, 2),
                'prob_benign':    round(result['prob_benign']    * 100, 2),
                'prob_malignant': round(result['prob_malignant'] * 100, 2),
                'tta_individual': result['tta_individual'],
            },
            'images': edges,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
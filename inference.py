"""
Inference pipeline:
  • build SLIC graph (same 8-dim node features as training)
  • TTA x5 (same 5 transforms as the notebook)
  • Sobel edge map (jet colormap on dark background — matches notebook)
  • Canny + 4 morphology ops (Erosion, Dilation, Opening, Closing) — 3x3 ellipse kernel

All preprocessing matches the trained val_transform exactly:
    Resize 224 → ToTensor → Normalize(ImageNet mean/std)
"""
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
from skimage.segmentation import slic
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, Batch

IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── TTA transforms (same 5 used in notebook) ─────────────────────────
_tta_transforms = [
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
               T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    T.Compose([T.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
               T.CenterCrop(IMG_SIZE), T.ToTensor(),
               T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),
               T.RandomRotation(degrees=(90, 90)), T.ToTensor(),
               T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),
               T.RandomVerticalFlip(p=1.0), T.ToTensor(),
               T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
]


# ── SLIC graph (identical to training cache builder) ─────────────────
def build_slic_graph(img_pil, n_segments=75, k=5, img_size=IMG_SIZE):
    img_resized = img_pil.resize((img_size, img_size))
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    gray = img_arr.mean(axis=2)

    segments = slic(img_arr, n_segments=n_segments, compactness=10,
                    sigma=1, start_label=0, channel_axis=2)
    labels = np.unique(segments)
    n_nodes = len(labels)

    feats = np.zeros((n_nodes, 8), dtype=np.float32)
    centroids = np.zeros((n_nodes, 2), dtype=np.float32)
    H, W = gray.shape

    for i, lbl in enumerate(labels):
        mask = segments == lbl
        ys, xs = np.where(mask)
        region_rgb = img_arr[mask]
        region_gray = gray[mask]
        feats[i, 0] = region_rgb[:, 0].mean()
        feats[i, 1] = region_rgb[:, 1].mean()
        feats[i, 2] = region_rgb[:, 2].mean()
        feats[i, 3] = region_gray.std()
        feats[i, 4] = region_gray.mean()
        feats[i, 5] = ys.mean() / H
        feats[i, 6] = xs.mean() / W
        feats[i, 7] = mask.sum() / (H * W)
        centroids[i, 0] = ys.mean()
        centroids[i, 1] = xs.mean()

    if n_nodes > 1:
        k_eff = min(k, n_nodes - 1)
        A = kneighbors_graph(centroids, n_neighbors=k_eff, mode='connectivity',
                             include_self=False).tocoo()
        edge_index = np.vstack([A.row, A.col]).astype(np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return Data(
        x=torch.from_numpy(feats),
        edge_index=torch.from_numpy(edge_index)
    )


# ── helpers ──────────────────────────────────────────────────────────
def _np_to_b64(arr, is_gray=False):
    """Encode a numpy uint8 array (HxW or HxWx3) as base64 PNG."""
    if is_gray:
        pil = Image.fromarray(arr).convert('L')
    else:
        pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')
# ── H&E sanity check (out-of-distribution filter) ────────────────────
def check_he_image(img_pil):
    """
    Heuristic check: does this image plausibly look like H&E histopathology?
    Rejects non-pathology uploads (photos, X-rays, MRIs, screenshots).

    Returns (is_he: bool, reason: str, stats: dict).
    """
    img = np.array(img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # H&E stains live in two hue regions (OpenCV hue range is 0–180):
    #   Eosin pink-red:        hue 160–180 and 0–15 (red wraps around)
    #   Hematoxylin purple:    hue 120–160
    pink_mask = ((h >= 120) | (h <= 15)) & (s >= 25) & (v >= 50) & (v <= 240)

    # Slide background is bright + low-saturation
    white_mask = (s < 25) & (v > 220)

    tissue_mask = ~white_mask

    pink_frac        = float(pink_mask.mean())
    white_frac       = float(white_mask.mean())
    tissue_frac      = float(tissue_mask.mean())
    tissue_pink_frac = float(pink_mask.sum() / max(tissue_mask.sum(), 1))

    stats = {
        'pink_purple_frac':    round(pink_frac, 3),
        'background_frac':     round(white_frac, 3),
        'tissue_frac':         round(tissue_frac, 3),
        'tissue_pink_frac':    round(tissue_pink_frac, 3),
    }

    # Decision rules (tuned for DiagSet — adjust if you get false rejections)
    if pink_frac < 0.05:
        return False, (
            "This image does not look like an H&E-stained histopathology slide "
            "(too little pink/purple staining detected). "
            "Please upload a prostate H&E image from a dataset like DiagSet."
        ), stats

    if tissue_frac > 0.10 and tissue_pink_frac < 0.25:
        return False, (
            "This image has tissue-like regions, but the color profile "
            "does not match H&E staining. Please upload a prostate H&E slide."
        ), stats

    return True, "OK", stats

# ── Edge maps + morphology (matches notebook visuals) ────────────────
def compute_edge_maps(img_pil):
    """
    Returns dict of base64 PNGs:
      original, sobel (colored), canny,
      morph_erosion, morph_dilation, morph_opening, morph_closing
    """
    img = np.array(img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel — colored with JET colormap (matches notebook)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    sobel_norm = (sobel_mag / (sobel_mag.max() + 1e-8) * 255).astype(np.uint8)
    sobel_colored = cv2.applyColorMap(sobel_norm, cv2.COLORMAP_JET)
    sobel_colored = cv2.cvtColor(sobel_colored, cv2.COLOR_BGR2RGB)

    # Canny edges
    # Canny edges
    canny = cv2.Canny(gray, 80, 180)

    return {
        'original': _np_to_b64(img),
        'sobel':    _np_to_b64(sobel_colored),
        'canny':    _np_to_b64(canny, is_gray=True),
    }


# ── Inference (TTA x5, model in eval mode) ───────────────────────────
@torch.no_grad()
def predict(model, img_pil):
    model.eval()
    img_pil = img_pil.convert('RGB')

    graph = build_slic_graph(img_pil)

    tta_probs = []
    for tfm in _tta_transforms:
        # Duplicate to batch=2 so the GAT branch has at least one real edge
        img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
        img_t = torch.cat([img_t, img_t], dim=0)
        gb2 = Batch.from_data_list([graph, graph]).to(DEVICE)

        logits = model(img_t, gb2)
        prob = F.softmax(logits, dim=1)[0]
        tta_probs.append(prob.cpu().numpy())

    tta_arr = np.stack(tta_probs)
    mean_prob = tta_arr.mean(axis=0)
    pred_class = int(np.argmax(mean_prob))
    label = ['Benign', 'Malignant'][pred_class]

    return {
        'label': label,
        'pred_class': pred_class,
        'prob_benign':    float(mean_prob[0]),
        'prob_malignant': float(mean_prob[1]),
        'confidence': float(mean_prob.max()),
        'tta_individual': tta_arr.tolist(),
    }
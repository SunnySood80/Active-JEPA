from cv2 import kmeans
import gc, math, numpy as np, atexit, csv, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - CRITICAL for DDP to prevent hanging
import matplotlib.pyplot as plt

from MaskJEPA import MaskJEPA2D
from model import SegmentationHead
from Dataloader import batch_size_pretrain, jepa_dataset, jepa_collate
from Dataset import ADE20KDataset, ade_collate
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from utils import lr_lambda
import os
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




pretrain_loader = DataLoader(
    jepa_dataset,
    batch_size=batch_size_pretrain,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=jepa_collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pretrained JEPA model...")



class JEPASegmentationModel(nn.Module):
    """
    JEPA backbone + pixel decoder -> Enhanced SegmentationHead.
    """
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model.context_encoder
        self.pixel_decoder = backbone_model.pixel_decoder
        self.embed_dim = backbone_model.embed_dim
        self.ds16 = backbone_model.ds16
        self.ds32 = backbone_model.ds32

    def forward(self, x):
        B, C, H, W = x.shape
        tokens, (enc_h, enc_w) = self.backbone(x)                 # [B, P, D]
        feat = tokens.transpose(1,2).reshape(B, self.embed_dim, enc_h, enc_w)

        # pyramid as in pretrain
        C3  = F.interpolate(feat, size=(H//8,  W//8),  mode='bilinear', align_corners=False)
        x16 = self.ds16(C3)
        x32 = self.ds32(x16)
        C4  = F.interpolate(x16, size=(H//16, W//16), mode='bilinear', align_corners=False)
        C5  = F.interpolate(x32, size=(H//32, W//32), mode='bilinear', align_corners=False)

        Fi1, F_last = self.pixel_decoder([C3, C4, C5], (H, W))    # Fi1 ~ s/8, F_last ~ s/4
        return F_last # [B, K, H, W]



jepa_model = MaskJEPA2D(
    in_chans=3, tau=0.996, fi1_mask_ratio=0.5,
    num_queries=50, num_cross_attn=5, num_self_attn=1, patch_size=8
).to(device)

weights_path = "./quick_test/jepa_rl_training_output_100_latent_denoise_scatter_.4.4.2/mask_jepa_rl_pretrained_weights.pt"
if not os.path.exists(weights_path):
    print(f"ERROR: Pretrained JEPA weights not found at {weights_path}")
    print("Please run train.py first to pretrain the JEPA model!")
    exit(1)

# Load pretrained weights
ckpt = torch.load(weights_path, map_location=device)
jepa_model.context_encoder.load_state_dict(ckpt['backbone_state_dict'])
jepa_model.pixel_decoder.load_state_dict(ckpt['pixel_decoder_state_dict'])

model = JEPASegmentationModel(jepa_model).to(device)
model.eval()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sample_idx = 18
sample_image = pretrain_loader.dataset[sample_idx]['image']  # (3, H, W)


print("Running probe on JEPA features with kmeans...")
with torch.no_grad():
    features = model(sample_image.unsqueeze(0).to(device))  # (B, C, H, W)
    B, C, H, W = features.shape
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()



# Fit kmeans on this image's features
print(f"Fitting k-means on {features_flat.shape[0]} patches...")

# k_values = range(2, 21)
# inertias = []

# for k in k_values:
#     kmeans_test = KMeans(n_clusters=k, random_state=0, n_init=10)
#     kmeans_test.fit(features_flat)
#     inertias.append(kmeans_test.inertia_)

kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
kmeans.fit(features_flat)

# # Plot elbow curve
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, inertias, 'bo-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.savefig('elbow.png')
# print("Saved elbow.png - look for the 'elbow' point")

# Get cluster assignments
cluster_map = kmeans.predict(features_flat).reshape(H, W)
print(f"Extracted features shape: {features_flat.shape}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(sample_image.permute(1, 2, 0))
axes[0].set_title("Original Image")
axes[1].imshow(cluster_map, cmap='viridis')
axes[1].set_title("Cluster Assignments")
plt.colorbar(axes[1].images[0], ax=axes[1])
plt.savefig('cluster_map_3_rl.png')
print("Saved cluster_map.png")
"""
Panoramic Place Recognition using CNN global descriptors.

Extracts compact image-level descriptors from equirectangular panoramas
using a pre-trained ResNet, then performs image retrieval via cosine similarity.

Approach:
  - ResNet-18 (pretrained on ImageNet) as feature extractor
  - Input: equirectangular image (H×W×3)
  - Output: 512-d L2-normalized global descriptor
  - Retrieval: cosine similarity → nearest neighbor search

This is the "off-the-shelf" baseline from the NetVLAD paper (Arandjelovic 2016),
which showed that standard CNN features work surprisingly well for place recognition.

For panoramic images specifically, the CNN benefits from the full 360° context:
rotation of the camera simply shifts the equirectangular image horizontally,
so the global descriptor is inherently rotation-robust.
"""

import numpy as np

try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PlaceRecognizer:
    """Panoramic place recognition via CNN global descriptors."""

    def __init__(self, device="cpu", backbone="resnet18"):
        """
        Args:
            device: "cpu" or "cuda".
            backbone: CNN architecture ("resnet18" or "resnet50").
        """
        if not HAS_TORCH:
            raise ImportError("torch and torchvision required for PlaceRecognizer")

        self.device = device

        # Load pre-trained ResNet, remove classifier head
        if backbone == "resnet18":
            self.model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )
            self.feat_dim = 512
        elif backbone == "resnet50":
            self.model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove the final fc layer, keep avgpool
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()

        # ImageNet normalization
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.database = None       # (N, feat_dim) normalized descriptors
        self.database_meta = None  # list of metadata dicts

    def _preprocess(self, image):
        """
        Preprocess a numpy image for ResNet.

        Args:
            image: (H, W, 3) uint8 RGB numpy array.

        Returns:
            (1, 3, 224, 224) normalized tensor.
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        # Convert to PIL and resize to 224x224
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        # Resize keeping aspect ratio, then center crop
        pil_img = pil_img.resize((256, 256), Image.LANCZOS)
        pil_img = TF.center_crop(pil_img, 224)

        # To tensor [0, 1]
        tensor = TF.to_tensor(pil_img).to(self.device)
        tensor = self.normalize(tensor)
        return tensor.unsqueeze(0)

    def extract_descriptor(self, image):
        """
        Extract a global descriptor from a panoramic image.

        Args:
            image: (H, W, 3) uint8 RGB equirectangular image.

        Returns:
            (feat_dim,) float32 L2-normalized descriptor.
        """
        tensor = self._preprocess(image)

        with torch.no_grad():
            feat = self.model(tensor)  # (1, feat_dim)

        feat = feat.cpu().numpy().ravel().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(feat) + 1e-10
        return feat / norm

    def build_database(self, images, metadata=None):
        """
        Build a searchable database of image descriptors.

        Args:
            images: list of (H, W, 3) uint8 images.
            metadata: optional list of metadata dicts (one per image).
        """
        n = len(images)
        self.database = np.zeros((n, self.feat_dim), dtype=np.float32)

        for i, img in enumerate(images):
            self.database[i] = self.extract_descriptor(img)
            if (i + 1) % 20 == 0:
                print(f"\r  Building database: {i+1}/{n}", end="", flush=True)
        if n >= 20:
            print()

        self.database_meta = metadata if metadata else [{"id": i} for i in range(n)]
        print(f"Database built: {n} images, {self.feat_dim}-dim descriptors")

    def query(self, image, top_k=5):
        """
        Retrieve the most similar database images.

        Args:
            image: (H, W, 3) uint8 query image.
            top_k: number of results to return.

        Returns:
            results: list of (index, similarity, metadata) tuples, sorted by similarity.
        """
        if self.database is None:
            raise ValueError("Database not built. Call build_database() first.")

        q_desc = self.extract_descriptor(image)
        similarities = self.database @ q_desc  # cosine similarity (both normalized)

        top_idx = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "metadata": self.database_meta[idx] if self.database_meta else None,
            })

        return results


def evaluate_place_recognition(images, positions, room_labels=None,
                                leave_one_out=True, device="cpu"):
    """
    Evaluate place recognition performance.

    Args:
        images: list of images.
        positions: (N, 3) positions.
        room_labels: optional list of room labels.
        leave_one_out: if True, each image queries all others.
        device: compute device.

    Returns:
        results: dict with retrieval metrics.
    """
    rec = PlaceRecognizer(device=device)
    rec.build_database(images)

    n = len(images)
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    all_similarities = []
    queries = []

    for i in range(n):
        query_img = images[i]
        results = rec.query(query_img, top_k=min(6, n))

        # Exclude self-retrieval
        filtered = [r for r in results if r["index"] != i]
        top1 = filtered[0]["index"] if filtered else -1

        # Check if the top match is from the same room (if labels available)
        same_room = False
        if room_labels:
            same_room = room_labels[top1] == room_labels[i] if top1 >= 0 else False

        top1_correct += 1 if top1 == i else 0  # self-retrieval doesn't count
        if top1 == i:
            # If self was retrieved, check the 2nd best
            actual_top1 = filtered[0]["index"] if len(filtered) > 0 else -1

        queries.append({
            "query_idx": i,
            "top1_idx": filtered[0]["index"] if filtered else -1,
            "top1_similarity": filtered[0]["similarity"] if filtered else 0,
            "top3_indices": [r["index"] for r in filtered[:3]],
            "top5_indices": [r["index"] for r in filtered[:5]],
            "same_room_top1": same_room,
        })

    # Compute metrics
    # For place recognition, "correct" = top-1 is a nearby position
    # (within some distance threshold)
    position_errors = []
    for q in queries:
        if q["top1_idx"] >= 0:
            err = np.linalg.norm(positions[q["query_idx"]] - positions[q["top1_idx"]])
            position_errors.append(err)

    metrics = {
        "n_queries": n,
        "mean_top1_distance": float(np.mean(position_errors)) if position_errors else -1,
        "median_top1_distance": float(np.median(position_errors)) if position_errors else -1,
        "top1_within_1m": sum(1 for e in position_errors if e < 1.0),
        "top1_within_3m": sum(1 for e in position_errors if e < 3.0),
    }

    return metrics, queries, rec

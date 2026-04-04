from abc import ABC, abstractmethod

import numpy as np


class TeamClassifier(ABC):
    """Assigns team labels to player crop images."""

    @abstractmethod
    def classify(self, crops: list[np.ndarray]) -> list[str]:
        """Given a list of BGR player crop images, return team labels ('A', 'B', 'referee')."""
        ...


class FakeTeamClassifier(TeamClassifier):
    """Returns a fixed team label for all crops — used in tests."""

    def __init__(self, label: str = "A") -> None:
        self._label = label

    def classify(self, crops: list[np.ndarray]) -> list[str]:
        return [self._label] * len(crops)


class CLIPTeamClassifier(TeamClassifier):
    """
    K-means team assignment using CLIP visual embeddings.

    Usage:
        clf = CLIPTeamClassifier()
        clf.fit(all_crops_from_shot)   # cluster embeddings into k=3 groups
        labels = clf.classify(crops)   # predict team for new crops
    """

    def __init__(self, n_clusters: int = 3) -> None:
        self._n_clusters = n_clusters
        self._kmeans = None
        # Cluster-ID → team name; 0 and 1 are teams, 2 is referee by default.
        # Caller can override by inspecting cluster centroids if needed.
        self._id_to_name: dict[int, str] = {0: "A", 1: "B", 2: "referee"}
        self._processor = None
        self._clip_model = None

    def _load_clip(self) -> None:
        if self._processor is not None:
            return
        from transformers import CLIPProcessor, CLIPModel  # lazy import
        import torch
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model.eval()

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        import torch
        self._load_clip()
        pil_images = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR → RGB
        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._clip_model.get_image_features(**inputs)
        return feats.numpy()

    def fit(self, crops: list[np.ndarray]) -> None:
        """Cluster a representative batch of crops to fix team identities for this shot."""
        from sklearn.cluster import KMeans

        feats = self._embed(crops)
        km = KMeans(n_clusters=self._n_clusters, n_init=10, random_state=0)
        km.fit(feats)
        self._kmeans = km

    def classify(self, crops: list[np.ndarray]) -> list[str]:
        if self._kmeans is None:
            raise RuntimeError("Call fit() before classify()")
        feats = self._embed(crops)
        cluster_ids = self._kmeans.predict(feats)
        return [self._id_to_name.get(int(cid), "unknown") for cid in cluster_ids]

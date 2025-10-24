"""Face embedding and clustering helpers for character datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingConfig:
    """Configure embedding backends."""

    backend: str = "clip"
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"


def embed_faces(
    image_paths: Iterable[Path],
    config: EmbeddingConfig,
    dry_run: bool = False,
) -> Dict[Path, List[float]]:
    """Return embeddings for each image path."""

    paths = list(image_paths)

    if dry_run:
        logger.info("Dry-run face embedding for %d images", len(paths))
        return {path: [0.0, 0.1, -0.1] for path in paths}

    if config.backend == "clip":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError:
            logger.warning("SentenceTransformer not available; returning placeholder embeddings.")
            return {path: [0.0, 0.1, -0.1] for path in paths}

        model = SentenceTransformer(config.model_name, device=config.device)
        vectors = model.encode([str(path) for path in paths], convert_to_numpy=True)  # type: ignore[arg-type]
        return {path: vector.tolist() for path, vector in zip(paths, vectors, strict=False)}

    logger.warning("Unsupported embedding backend %s", config.backend)
    return {path: [0.0] for path in paths}


def cluster_embeddings(embeddings: Dict[Path, List[float]], clusters: int = 4) -> Dict[int, List[Path]]:
    """Cluster embeddings to suggest character groups."""

    try:
        from sklearn.cluster import KMeans  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        logger.warning("scikit-learn not installed; returning single cluster.")
        return {0: list(embeddings.keys())}

    vectors = np.array(list(embeddings.values()))
    estimator = KMeans(n_clusters=min(clusters, len(vectors)), n_init="auto")
    labels = estimator.fit_predict(vectors)
    mapping: Dict[int, List[Path]] = {}
    for label, path in zip(labels, embeddings.keys(), strict=False):
        mapping.setdefault(int(label), []).append(path)
    return mapping

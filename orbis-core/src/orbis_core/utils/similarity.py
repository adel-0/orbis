"""
Utility functions for similarity score calculations and normalization.
Provides consistent similarity score handling across applications.
"""

import math


def normalize_cosine_similarity(distance: float, clamp: bool = True) -> float:
    """
    Convert cosine distance to similarity score.

    Cosine distance = 1 - cosine similarity
    So similarity = 1 - distance

    Args:
        distance: Cosine distance value
        clamp: Whether to clamp result to [0, 1] range

    Returns:
        Normalized similarity score
    """
    similarity = 1.0 - distance

    if clamp:
        similarity = max(0.0, min(1.0, similarity))

    return similarity


def normalize_euclidean_distance(distance: float, max_distance: float = None) -> float:
    """
    Convert Euclidean distance to similarity score.

    Uses inverse distance formula: similarity = 1 / (1 + distance)
    Optionally normalizes by maximum distance.

    Args:
        distance: Euclidean distance value
        max_distance: Maximum possible distance for normalization

    Returns:
        Normalized similarity score between 0 and 1
    """
    if max_distance is not None:
        # Normalize distance to [0, 1] range first
        normalized_distance = min(distance / max_distance, 1.0)
        return 1.0 - normalized_distance
    else:
        # Use inverse distance formula
        return 1.0 / (1.0 + distance)


def normalize_dot_product_similarity(dot_product: float,
                                   norm_a: float = None,
                                   norm_b: float = None) -> float:
    """
    Normalize dot product to similarity score.

    If norms are provided, computes cosine similarity.
    Otherwise assumes vectors are normalized.

    Args:
        dot_product: Dot product of two vectors
        norm_a: L2 norm of first vector (optional)
        norm_b: L2 norm of second vector (optional)

    Returns:
        Normalized similarity score
    """
    if norm_a is not None and norm_b is not None:
        # Compute cosine similarity
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = dot_product / (norm_a * norm_b)
    else:
        # Assume vectors are already normalized
        similarity = dot_product

    # Clamp to [0, 1] range (cosine similarity can be [-1, 1])
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def compute_vector_similarity(vector_a: list[float],
                             vector_b: list[float],
                             method: str = "cosine") -> float:
    """
    Compute similarity between two vectors using specified method.

    Args:
        vector_a: First vector
        vector_b: Second vector
        method: Similarity method ("cosine", "euclidean", "dot")

    Returns:
        Similarity score between 0 and 1

    Raises:
        ValueError: If vectors have different lengths or invalid method
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length")

    if not vector_a or not vector_b:
        return 0.0

    if method == "cosine":
        return _cosine_similarity(vector_a, vector_b)
    elif method == "euclidean":
        return _euclidean_similarity(vector_a, vector_b)
    elif method == "dot":
        return _dot_product_similarity(vector_a, vector_b)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return max(0.0, min(1.0, similarity))


def _euclidean_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute similarity from Euclidean distance."""
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b, strict=False)))
    return normalize_euclidean_distance(distance)


def _dot_product_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute normalized dot product similarity."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    return normalize_dot_product_similarity(dot_product, norm_a, norm_b)


def batch_normalize_similarities(similarities: list[float],
                                method: str = "min_max") -> list[float]:
    """
    Normalize a batch of similarity scores.

    Args:
        similarities: List of similarity scores
        method: Normalization method ("min_max", "z_score", "robust")

    Returns:
        List of normalized similarity scores
    """
    if not similarities:
        return []

    if method == "min_max":
        min_val = min(similarities)
        max_val = max(similarities)
        if max_val == min_val:
            return [0.5] * len(similarities)  # All equal, return neutral score
        return [(s - min_val) / (max_val - min_val) for s in similarities]

    elif method == "z_score":
        mean_val = sum(similarities) / len(similarities)
        variance = sum((s - mean_val) ** 2 for s in similarities) / len(similarities)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        # Normalize and clamp to [0, 1]
        normalized = [(s - mean_val) / std_dev for s in similarities]
        min_norm = min(normalized)
        max_norm = max(normalized)
        if max_norm == min_norm:
            return [0.5] * len(normalized)
        return [(n - min_norm) / (max_norm - min_norm) for n in normalized]

    elif method == "robust":
        sorted_sims = sorted(similarities)
        n = len(sorted_sims)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_sims[q1_idx]
        q3 = sorted_sims[q3_idx]
        iqr = q3 - q1 if q3 != q1 else 1.0

        # Normalize using IQR and clamp
        normalized = [max(0.0, min(1.0, (s - q1) / iqr)) for s in similarities]
        return normalized

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def confidence_weighted_similarity(similarity: float,
                                  confidence: float,
                                  confidence_threshold: float = 0.5) -> float:
    """
    Apply confidence weighting to similarity score.

    Args:
        similarity: Base similarity score
        confidence: Confidence score [0, 1]
        confidence_threshold: Minimum confidence for full weight

    Returns:
        Confidence-weighted similarity score
    """
    if confidence < confidence_threshold:
        # Reduce similarity for low confidence
        weight = confidence / confidence_threshold
        return similarity * weight
    else:
        # Boost similarity for high confidence
        boost = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
        return min(1.0, similarity + (1.0 - similarity) * boost * 0.1)

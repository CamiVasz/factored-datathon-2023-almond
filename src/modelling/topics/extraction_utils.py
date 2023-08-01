import pandas as pd
import numpy as np
import scipy.sparse as sp


def group_reviews_per_topic(reviews: pd.DataFrame) -> pd.DataFrame:
    return (
        reviews
        .groupby('topic', as_index=False)
        .agg({'text': ' '.join})
    )


def mark_empty_docs(text: str) -> str:
    if not text:
        return "emptydoc"
    return text


def check_reviews_schema(reviews: pd.DataFrame):
    pass


def top_n_idx_sparse(matrix: sp.csr_matrix, n: int) -> np.ndarray:
    indices = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[
            le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]
        ]
        values = [
            values[index] if len(values) >= index + 1 else None 
            for index in range(n)
        ]
        indices.append(values)
    return np.array(indices)


def top_n_values_sparse(
        matrix: sp.csr_matrix, indices: np.ndarray) -> np.ndarray:
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([
            matrix[row, value] if value is not None else 0 for value in values
        ])
        top_values.append(scores)
    return np.array(top_values)

import typing as tp
import pandas as pd
import numpy as np
import faiss
from numpy import typing as ntp
import tensorflow_hub as tfhub
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


Embedder = tp.Callable[[list[str]], ntp.ArrayLike]


def load_model(model_name: str):
    if "universal-sentence-encoder" in model_name:
        model = tfhub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        def inner_forward_fn(input_texts: list[str]):
            return model(input_texts)

    else:
        model = SentenceTransformer(model_name)

        def inner_forward_fn(input_texts: list[str]):
            return model.encode(input_texts, convert_to_tensor=True)

    return inner_forward_fn


def get_matching_reviews_ids(relevant_products: pd.DataFrame):
    matching_reviews_ids = np.concatenate(
        [np.array(i).reshape(-1) for i in relevant_products.reviewID.tolist()]
    )

    return matching_reviews_ids


def query_relevant_documents(
    product_model: Embedder,
    indexer: faiss.Index,
    products: pd.DataFrame,
    query_text: str,
) -> pd.DataFrame:
    embedded_query = product_model([query_text])
    dist, idx = indexer.search(embedded_query, 64)

    relevant_products = products.iloc[idx[dist < 1]]
    return relevant_products


def get_relevant_reviews(
    relevant_products: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    review_ids = ":".join(relevant_products.reviewID).split(":")
    relevant_reviews = reviews.loc[review_ids].drop_duplicates("reviewText")

    summaries = relevant_reviews.summary

    relevant_reviews = relevant_reviews[~summaries.isna()]
    relevant_reviews = relevant_reviews[~summaries.str.match(r"\w+ Star(s)?")]

    return relevant_reviews


def clusterize_reviews(
    relevant_reviews: pd.DataFrame,
    reviews_embedder: Embedder,
    clusterer,
) -> pd.Series:
    embedded_reviews = reviews_embedder(relevant_reviews.summary.tolist())
    dist_matrix = cosine_distances(embedded_reviews).astype(np.float64)
    clusters = clusterer.fit(dist_matrix)
    return clusters.labels_


def get_key_reviews(
    reviews_with_topics, extracted_topics, top_k_topics: int = 5
) -> list[str]:
    hist_of_topics = reviews_with_topics.topic.value_counts()
    top_k = min(top_k_topics, len(hist_of_topics))
    indices = hist_of_topics.iloc[:top_k].index

    top_rated_reviews = set(
        reviews_with_topics
        .sort_values(['topic', 'overall'], ascending=False)
        .groupby('topic')
        .head(1)
        .set_index('topic')
        .loc[indices]
        .reviewText
        .tolist()
    )
    representative_reviews = {
        extracted_topics[idx].representative_examples[0]
        for idx in indices
    }

    return list(top_rated_reviews | representative_reviews)


def _format_review(x):
    single_line = x.split("\n")[0]
    return f' - {single_line.strip()}'


def key_reviews_to_prompt(reviews):
    return '\n'.join([
        _format_review(i) for i in reviews
    ])

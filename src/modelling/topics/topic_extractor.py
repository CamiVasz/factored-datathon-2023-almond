import dataclasses
import collections
import typing as tp

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.modelling.topics import extraction_utils


TopicID = tp.Union[str, int]


@dataclasses.dataclass
class Topic:
    topic_id: TopicID
    n_grams: list[tuple[str, float]]
    representative_examples: list[str]
    stats: dict[str, float]
    sentiment: tp.Optional[str] = None  # postive | negative
    text_label: tp.Optional[str] = None


@dataclasses.dataclass
class TopicExtractionConfig:
    vectorizer_model: CountVectorizer
    ctfidf_model: TfidfTransformer
    number_of_grams_per_topic: int = 10
    number_of_representative_documents: int = 3
    reduce_topics: tp.Union[int, None] = None
    review_text_key: str = "reviewText"

    def get_vectorizer_model(self):
        return self.vectorizer_model

    def get_extraction_model(self):
        return self.ctfidf_model


class TopicExtractor:
    def __init__(self, config: TopicExtractionConfig):
        self.config = config
        self.vectorizer_model = self.config.get_vectorizer_model()
        self.ctfidf_model = self.config.get_extraction_model()
        
        self.review_text_key = self.config.review_text_key
        self.c_tf_idf = None

    def __call__(
            self, reviews: pd.DataFrame
    ) -> dict[TopicID, Topic]:
        extraction_utils.check_reviews_schema(reviews)
        topic_stats = self.compute_topic_stats(reviews)
        self.extract_topics(reviews)

        representative_examples = self.extract_representative_documents(
            reviews)

        return {
            topic_id: Topic(
                topic_id=topic_id,
                n_grams=self.words_per_topic[topic_id],
                representative_examples=example,
                stats=topic_stats[topic_id]
            ) for topic_id, example in representative_examples.items()
        }

    def extract_topics(self, reviews: pd.DataFrame):
        reviews_per_topic = extraction_utils.group_reviews_per_topic(
            reviews, self.review_text_key)
        self.c_tf_idf, vocab = self.compute_c_tf_idf(reviews_per_topic)
        self.words_per_topic = self.extract_words_per_topic(reviews, vocab)

    def _prepare_c_tf_idf_text(self, raw_text: pd.Series) -> pd.Series:
        clean_text = raw_text.str.replace("\n", " ")
        clean_text = clean_text.str.replace("\t", " ")
        clean_text = clean_text.str.replace(r"[^A-Za-z0-9 ]+", "", regex=True)
        clean_text = clean_text.apply(extraction_utils.mark_empty_docs)

        return clean_text

    def compute_topic_stats(self, reviews):
        return collections.defaultdict(dict)

    def compute_c_tf_idf(
            self,
            reviews_per_topic: pd.DataFrame
            ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Compute C-TF-IDF per topic

        Args:
            reviews_per_topic: A per topic dataframe, it must be the output of
                `extraction_utils.group_reviews_per_topic`
        """
        clean_reviews = self._prepare_c_tf_idf_text(
          reviews_per_topic[self.review_text_key])

        # update in place
        self.vectorizer_model.fit(clean_reviews)
        vectorized_reviews = self.vectorizer_model.transform(clean_reviews)

        vocab = self.vectorizer_model.get_feature_names_out()
        c_tf_idf = self.ctfidf_model.fit_transform(vectorized_reviews)

        return c_tf_idf, vocab

    def extract_words_per_topic(
            self, reviews: pd.DataFrame, vocab: np.ndarray):
        labels = reviews.topic.unique().astype(int)

        indices = extraction_utils.top_n_idx_sparse(
            self.c_tf_idf, self.config.number_of_grams_per_topic
        )
        scores = extraction_utils.top_n_values_sparse(self.c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {
            label: [
                (vocab[word_index], score)
                if word_index is not None and score > 0
                else ("", 0.00001)
                for word_index, score in
                zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }
        topics = {
            label: values[:self.config.number_of_grams_per_topic]
            for label, values in topics.items()
        }

        return topics

    def extract_representative_documents(self, reviews):
        sample_reviews_per_topic = (
            reviews.groupby('topic')
                   .sample(n=500, replace=True)
                   .drop_duplicates(subset=[self.review_text_key])
        )

        repr_docs = []
        repr_docs_indices = []
        repr_docs_mappings = {}
        repr_docs_ids = []
        labels = sorted(list(self.words_per_topic.keys()))

        for index, topic in enumerate(labels):
            # Slice data
            selection = sample_reviews_per_topic.loc[
                sample_reviews_per_topic.topic == topic, :]
            selected_docs = selection[self.review_text_key].values
            selected_full_docs = selection['reviewText'].values
            selected_docs_ids = selection.index.tolist()

            # Calculate similarity
            nr_repr_docs = self.config.number_of_representative_documents
            nr_docs = min(nr_repr_docs, len(selected_docs))
            bow = self.vectorizer_model.transform(selected_docs)
            ctfidf = self.ctfidf_model.transform(bow)
            sim_matrix = cosine_similarity(ctfidf, self.c_tf_idf[index])

            # TODO(shpotes): add diversity

            # extract top n most representative documents
            indices = np.argpartition(
                sim_matrix.reshape(1, -1)[0], -nr_docs)[-nr_docs:]
            docs = [selected_full_docs[index] for index in indices]

            doc_ids = [
                selected_docs_ids[index] 
                for index, doc in enumerate(selected_docs) if doc in docs
            ]
            repr_docs_ids.append(doc_ids)
            repr_docs.extend(docs)
            repr_docs_indices.append([
                repr_docs_indices[-1][-1] + i + 1 if index != 0 else i 
                for i in range(nr_docs)
            ])

        repr_docs_mappings = {
            topic: repr_docs[i[0]:i[-1]+1]
            for topic, i in zip(self.words_per_topic.keys(), repr_docs_indices)
        }

        return repr_docs_mappings

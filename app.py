import os

import google.generativeai as palm
import streamlit as st
import pandas as pd
import faiss
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

from src.modelling.topics.topic_extractor import (
    TopicExtractionConfig, TopicExtractor
)
from src.modelling.topics.class_tf_idf import ClassTfidfTransformer
from src import deploy_utils


def get_prompt(title, reviews):
    return f"""We are doing a marketing research analysis, in particular we are trying to understand what users thing about a particular market in order to generate tips for future sellers.
In particular, we are interesting to analyze the market for "{title}"

This is what amazon customers are saying about similar products:
{reviews}

Can you write some recomendations about how can we disrupt this market? Try to propose the necesary methodology to create a breaking product."""


def get_prompt_without_reviews(title):
    return f"""We are doing a marketing research analysis, in particular we are trying to understand what users thing about a particular market in order to generate tips for future sellers.
In particular, we are interesting to analyze the market for "{title}"

Take into account what customers are saying in the internet about these products. How are their reviews? How is the distribution of the product? What characteristics

Can you write some recomendations about how can we disrupt this market? Try to propose the necesary methodology to create a breaking product."""


no_electronics_message = """
Sorry, we are currently only recommending business that operate around electronics. Would you like to input another search?


This doesn't mean you make a mistake, I search amazon products and try to extract relevant reviews from similar products and we didn't find relevant products for your search.

#### Maybe you are way ahead of the market!


```
.                                           
                                      ___,,,
                                      \_[o o]
     Errare humanum est!              C\  _\/
             /                     _____),_/__
        ________                  /     \/   /
      _|       .|                /      o   /
     | |       .|               /          /
      \|       .|              /          /
       |________|             /_        \/
       __|___|__             _//\        \\
 _____|_________|____       \  \ \        \\
                    _|       ///  \        \\
                   |               \       /
                   |               /      /
                   |              /      /
 ________________  |             /__    /_
              ...|_|.............. /______\.......
```
"""

TEST_MODE = False


def setup_palm():
    palm.configure(api_key=os.environ.get('PALM_TOKEN'))


@st.cache_data
def load_data():
    reviews = pd.read_csv("data/filtered_reviews.csv").set_index("reviewID")
    products = pd.read_csv("data/products.csv")

    return reviews, products


def load_uncached_models():
    topic_extraction_config = TopicExtractionConfig(
        vectorizer_model=CountVectorizer(
            ngram_range=(1, 3), stop_words="english"),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        number_of_representative_documents=5,
        review_text_key="summary",
    )

    topic_extractor = TopicExtractor(topic_extraction_config)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, min_samples=5, metric="precomputed")

    return topic_extractor, clusterer


@st.cache_resource
def load_models():
    product_model = deploy_utils.load_model("all-MiniLM-L6-v2")
    reviews_model = deploy_utils.load_model(
        "https://tfhub.dev/google/universal-sentence-encoder/4"
    )
    product_indexer = faiss.read_index("vectordb/populated.index")

    return reviews_model, product_model, product_indexer


def render_cta_link(url, label, font_awesome_icon):
    st.markdown(
        '<link rel="stylesheet" href="<https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css>">',
        unsafe_allow_html=True,
    )
    button_code = f"""<a href="{url}" target=_blank><i class="fa {font_awesome_icon}"></i> {label}</a>"""
    return st.markdown(button_code, unsafe_allow_html=True)


def handler_review_query():
    relevant_products = deploy_utils.query_relevant_documents(
        product_model=product_model,
        indexer=product_indexer,
        products=products,
        query_text=st.session_state.user_search_query,
    )

    # TODO: check if there are relevant products
    if len(relevant_products) == 0:
        st.session_state.user_prompt = None
        st.session_state.palm_output = no_electronics_message
        return

    relevant_reviews = deploy_utils.get_relevant_reviews(
        relevant_products, reviews)

    raw_topic_assigment = deploy_utils.clusterize_reviews(
        relevant_reviews, reviews_model, clusterer)

    relevant_reviews["topic"] = raw_topic_assigment
    reviews_with_topics = relevant_reviews[relevant_reviews["topic"] != -1]

    # TODO: check if there are still topics

    extracted_topics = topic_extractor(reviews_with_topics)

    key_reviews = deploy_utils.get_key_reviews(
        reviews_with_topics,
        extracted_topics,
    )

    reviews_prompt = deploy_utils.key_reviews_to_prompt(key_reviews)
    prompt = get_prompt(st.session_state.user_search_query, reviews_prompt)
    st.session_state.user_prompt = prompt


def handler_product_without_reviews():
    st.session_state.user_prompt = get_prompt_without_reviews(
        st.session_state.user_search_query)


def palm_handler():
    response = palm.generate_text(prompt=st.session_state.user_prompt)
    st.session_state.palm_output = response.result


def render_search():
    """
    Render the search form in the sidebar.
    """
    with st.sidebar:
        query = st.text_input(
            label="What kind of product are you trying to sell?",
            placeholder="Your magic idea goes here ✨",
            key="user_search_query",
        )

        if query:
            try:
                handler_review_query()
            except:
                handler_product_without_reviews()

        if TEST_MODE:
            _ = st.text_area(
                label="test env",
                placeholder="prompt here",
                key="user_prompt"
            )

        if "user_prompt" in st.session_state and st.session_state.user_prompt:
            palm_handler()

        st.write("---")
        render_cta_link(
            url="https://github.com/CamiVasz/factored-datathon-2023-almond",
            label="Check the code",
            font_awesome_icon="fa-github",
        )


def render_palm_results():
    # TODO: temporal
    st.write("# ALMond recommendations")
    st.write(st.session_state.palm_output)

# Execution start here!

st.set_page_config(
    page_title="almond - demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_palm()
reviews, products = load_data()
reviews_model, product_model, product_indexer = load_models()
topic_extractor, clusterer = load_uncached_models()

render_search()
if "palm_output" in st.session_state:
    render_palm_results()

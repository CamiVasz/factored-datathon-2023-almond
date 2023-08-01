import typing as tp
import tensorflow_hub as hub
import faiss
import numpy as np
from faiss.contrib.ondisk import merge_ondisk
import pathlib


def batch_list(input_list: list[tp.Any], batch_size: int) -> list[tp.Any]:
    pointers = [*range(0, len(input_list), batch_size), None]
    for i, j in zip(pointers, pointers[1:]):
        yield input_list[i:j]


def train_index(
    model,
    faiss_index_str: str,
    text_batch: list[str],
    trained_index_path: pathlib.Path,
):
    model_output = model(text_batch)
    index = faiss.index_factory(512, faiss_index_str)
    index.train(model_output)
    faiss.write_index(index, trained_index_path)


def add_sharded_embeddings(
    model,
    batched_inputs: list[list[str]],
    trained_index_path: pathlib.Path,
    shard_root_dir: pathlib.Path,
):
    for idx, batch in enumerate(batched_inputs):
        index = faiss.read_index(trained_index_path)
        encoded_tensor = model(batch)
        index.add(encoded_tensor)
        faiss.write_index(index, shard_root_dir / f"shard_{idx}.index")


def merge_shards(
    trained_index_path: pathlib.Path,
    shard_root_dir: pathlib.Path,
    populated_index_path: pathlib.Path,
    merged_index_path: pathlib.Path,
):
    # NOTE: run this on the deploy env!
    index = faiss.read_index(trained_index_path)
    block_fnames = list(shard_root_dir.iterdir())
    merge_ondisk(index, block_fnames, merged_index_path)

    faiss.write_index(index, populated_index_path)


def load_populated_index(populated_index_path: pathlib.Path, nprobe: int = 16):
    populated_index = faiss.read_index(populated_index_path)
    populated_index.nprobe = 16

    return populated_index


def run_query(populated_index, model, query: str, top_k: int = 8) -> np.ndarray:
    encoded_query = model([query])
    _, idx = populated_index.search(encoded_query)

    return idx[0]

from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import re
import numpy as np
from markdownify import markdownify as md
import os
import time
from multiprocessing import Pool
from tqdm import tqdm

from huggingface_hub import Repository
from aismicroorg.utils import parse_arguments, load_config


def get_datafolders_name(*dataset_names):
    """
    Currently only works for 1 part parquets.
    """
    l = []
    for name in dataset_names:
        l.append(f"data/{name}.stackexchange.com/train-00000-of-00001.parquet")
    return l


def binary_comparison(answers):
    """Returns tuples of answers, first always best"""
    pairs = []

    for i in range(len(answers) - 1):
        for j in range(i + 1, len(answers)):
            if answers[i]["pm_score"] > answers[j]["pm_score"]:
                pairs.append((answers[i]["text"], answers[j]["text"]))
            elif answers[i]["pm_score"] < answers[j]["pm_score"]:
                pairs.append((answers[j]["text"], answers[i]["text"]))
    return pairs


def lang_callback(el):
    lang = el["class"][0] if el.has_attr("class") else None

    if not lang is None:
        lang = lang.split("-")[-1]
    return lang


def html2md(text):
    text = md(text, code_language_callback=lang_callback)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    return text.encode("utf-8", "replace").decode()


def preprocess(examples):
    """Cleans HTML and returns paired answers (j is better than k). Note that this returns more examples (one for each pair per question)."""

    MAX_PAIRS_PER_QUESTION = 10
    n_samples = len(examples["qid"])

    # initialize empty lists for new samples
    new_examples = {"question": [], "response_j": [], "response_k": []}
    for key in examples:
        new_examples[key] = []

    for sample_id in range(n_samples):
        # get pairs where first is always the better one
        pairs = binary_comparison(examples["answers"][sample_id])
        n_answers = len(examples["answers"][sample_id])

        # sample if we get more pairs than maximum
        if len(pairs) > MAX_PAIRS_PER_QUESTION:
            indices = np.random.choice(
                list(range(len(pairs))), MAX_PAIRS_PER_QUESTION, replace=False
            )
            pairs = [pairs[i] for i in indices]

        # construct the samples
        for pair in pairs:
            for key in examples:
                if key == "question":
                    new_examples[key].append(html2md(examples[key][sample_id]))
                else:
                    new_examples[key].append(examples[key][sample_id])
            new_examples["response_j"].append(html2md(pair[0]))
            new_examples["response_k"].append(html2md(pair[1]))
    return new_examples


def save_shard(shard_tuple):
    """Save shard"""
    filename, shard = shard_tuple
    # use to_json instead to save as json file
    shard.to_parquet(filename)


def save_manual_shards(
    ds,
    user="TommyBark",
    remote_dataset_repo="stack-exchange-paired_micro",
    subfolder="train",
):
    """Save sharded data
    Args:
        ds (Dataset): dataset to be saved
        user (str): user name
        remote_dataset_repo (str): remote dataset repository
        out_path (str): path to save the shards"""
    # this will create a folder OUT_PATH that is a clone of REMOTE_DATASET_REPO
    # you can save the shards inside it and do git add/commit/push to push data to the hub
    out_path = remote_dataset_repo
    # if out path doesnt already exist
    if not os.path.exists(out_path):
        repo = Repository(
            local_dir=out_path,
            clone_from=user + "/" + remote_dataset_repo,
            repo_type="dataset",
            use_auth_token=True,
            git_user=user,
        )

    # files will be numerous we save them in a folder called data inside out_path
    if not os.path.exists(out_path + "/data"):
        os.mkdir(out_path + "/data")
    os.mkdir(out_path + f"/data/{subfolder}")

    SHARD_SIZE = 1000 << 20
    if ds._indices is not None:
        dataset_nbytes = ds.data.nbytes * len(ds._indices) / len(ds.data)
    else:
        dataset_nbytes = ds.data.nbytes
    num_shards = int(dataset_nbytes / SHARD_SIZE) + 1
    print(f"Number of shards: {num_shards}")

    print("sharding the dataset")
    t_start = time.time()
    shards = (
        ds.shard(num_shards=num_shards, index=i, contiguous=True)
        for i in range(num_shards)
    )
    # use f"{OUT_PATH}/data/train-{index:05d}-of-{num_shards:05d}.json" instead for json files
    filenames = (
        f"{out_path}/data/{subfolder}/train-{index:05d}-of-{num_shards:05d}.parquet"
        for index in range(num_shards)
    )

    with Pool(16) as p:
        list(
            tqdm(
                p.imap_unordered(save_shard, zip(filenames, shards), chunksize=4),
                total=num_shards,
            )
        )
    print(f"Time to save dataset: {time.time()-t_start:.2f}")


def download_and_prepare_dataset(TOPIC_NAMES: list) -> None:
    ds = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        data_files=get_datafolders_name(*TOPIC_NAMES),
    )
    ds = ds.shuffle(seed=42)
    index = list(range(len(ds)))

    ds_splits = DatasetDict(
        {
            "finetune": ds.select(index[: len(ds) * 3 // 10]),
            "reward": ds.select(index[len(ds) * 3 // 10 : len(ds) * 6 // 10]),
            "rl": ds.select(index[len(ds) * 6 // 10 : len(ds) * 9 // 10]),
            "evaluation": ds.select(index[len(ds) * 9 // 10 :]),
        }
    )

    ds_result = ds_splits.map(preprocess, batch_size=1000, batched=True, num_proc=9)
    ds_result = ds_result.remove_columns(["answers"])

    for key in ds_result:
        save_manual_shards(ds_result[key], subfolder=key)


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    TOPIC_NAMES = config["stackexchange_topics"]
    download_and_prepare_dataset(TOPIC_NAMES)

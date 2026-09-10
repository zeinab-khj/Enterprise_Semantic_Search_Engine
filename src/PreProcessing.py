from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sklearn.model_selection import train_test_split
from datasets import Dataset

def prepare_training_data():

    corpus = load_dataset(
    "BeIR/scidocs",
    "corpus"
    )
    
    queries = load_dataset(
        "BeIR/scidocs",
        "queries"
    )
    corpus_df = corpus["corpus"].to_pandas()
    queries_df = queries["queries"].to_pandas()
    
    corpus_df["text_words"] = corpus_df["text"].str.split().str.len()
    corpus_df["title_words"] = corpus_df["title"].str.split().str.len()
    
    queries_df["text_words"] = queries_df["text"].str.split().str.len()
    queries_df["title_words"] = queries_df["title"].str.split().str.len()


    dataset = "scidocs"

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    
    data_path = util.download_and_unzip(
        url,
        "datasets"
    )
    
    corpus_beir, queries_beir, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")

    qrels_records = []
    for query_id, doc_scores in qrels.items():
        for doc_id, score in doc_scores.items():
            qrels_records.append({'query_id': query_id, 'doc_id': doc_id, 'score': score})
    
    qrels_df = pd.DataFrame(qrels_records)

    qrels_dict = (
        qrels_df.groupby("query_id")["doc_id"]
        .apply(set)
        .to_dict()
    )

    corpus_id_to_doc_id = dict(
        enumerate(corpus_df["_id"])
    )

    positive_qrels = qrels_df[qrels_df["score"] == 1]

    qrels_dict = (
        positive_qrels.groupby("query_id")["doc_id"]
        .apply(set)
        .to_dict()
    )

    query_ids = queries_df["_id"].tolist()

    train_query_ids, temp_query_ids = train_test_split(
        query_ids,
        test_size=0.2,
        random_state=42
    )
    
    val_query_ids, test_query_ids = train_test_split(
        temp_query_ids,
        test_size=0.5,
        random_state=42
    )

    train_qrels = qrels_df[
        qrels_df["query_id"].isin(train_query_ids)
    ].copy()
    
    val_qrels = qrels_df[
        qrels_df["query_id"].isin(val_query_ids)
    ].copy()
    
    test_qrels = qrels_df[
        qrels_df["query_id"].isin(test_query_ids)
    ].copy()

    train_positive = train_qrels[
        train_qrels["score"] == 1
    ][["query_id", "doc_id"]].copy()

    query_lookup = queries_df.set_index("_id")["text"].to_dict()
    doc_lookup = corpus_df.set_index("_id")["text"].to_dict()

    train_hard_negatives = train_hard_negatives.sort_values(
        ["query_id", "score"],
        ascending=[True, False]
    )

    hard_negative_pool = (
        train_hard_negatives
        .groupby("query_id")
        .head(5)
        .reset_index(drop=True)
    )

    train_pairs = train_positive.copy()

    train_pairs["query"] = train_pairs["query_id"].map(
        query_lookup
    )
    
    train_pairs["positive"] = train_pairs["doc_id"].map(
        doc_lookup
    )
    
    train_pairs = train_pairs[
        ["query", "positive"]
    ].reset_index(drop=True)

    val_positive = val_qrels[
        val_qrels["score"] == 1
    ][["query_id", "doc_id"]].copy()
    
    val_pairs = val_positive.copy()
    
    val_pairs["query"] = val_pairs["query_id"].map(
        query_lookup
    )
    
    val_pairs["positive"] = val_pairs["doc_id"].map(
        doc_lookup
    )
    
    val_pairs = val_pairs[
        ["query", "positive"]
    ].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(
        train_pairs[["query", "positive"]],
        preserve_index=False
    )
    
    val_dataset = Dataset.from_pandas(
        val_pairs[["query", "positive"]],
        preserve_index=False
    )

    return train_dataset, val_dataset

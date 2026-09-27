# src/evaluation.py

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# F0.5 FOR ONE S1 ENTITY
# ---------------------------------------------------------

def f05_from_sets(true_ids, predicted_ids):

    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    # True empty + predicted empty
    if len(true_ids) == 0 and len(predicted_ids) == 0:
        return 1.0

    # True empty + prediction made
    if len(true_ids) == 0 and len(predicted_ids) > 0:
        return 0.0

    # True matches exist + predicted nothing
    if len(true_ids) > 0 and len(predicted_ids) == 0:
        return 0.0

    true_positive = len(
        true_ids & predicted_ids
    )

    precision = (
        true_positive / len(predicted_ids)
    )

    recall = (
        true_positive / len(true_ids)
    )

    if precision == 0 and recall == 0:
        return 0.0

    f05 = (
        1.25 * precision * recall
    ) / (
        0.25 * precision + recall
    )

    return f05


# ---------------------------------------------------------
# S1-LEVEL MACRO F0.5
# ---------------------------------------------------------

def calculate_entity_f05(
    ground_truth,
    predictions,
    all_s1_ids
):

    scores = []

    for s1_id in all_s1_ids:

        true_ids = ground_truth.get(
            s1_id,
            set()
        )

        predicted_ids = predictions.get(
            s1_id,
            set()
        )

        score = f05_from_sets(
            true_ids,
            predicted_ids
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


# ---------------------------------------------------------
# GROUND TRUTH LOADER
# ---------------------------------------------------------

def load_ground_truth(path):

    df = pd.read_csv(
        path,
        sep="\t"
    )

    result = {}

    for _, row in df.iterrows():

        s1_id = row["source1_entity_id"]

        if (
            pd.isna(row["matched_entity_ids"])
            or str(row["matched_entity_ids"]).strip() == ""
        ):
            result[s1_id] = set()

        else:

            result[s1_id] = set(
                x.strip()
                for x in str(
                    row["matched_entity_ids"]
                ).split(",")
                if x.strip()
            )

    return result


# ---------------------------------------------------------
# CONVERT PROBABILITIES -> PREDICTIONS
# ---------------------------------------------------------

def predictions_from_probabilities(
    pair_df,
    probabilities,
    threshold
):

    temp = pair_df[
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ].copy()

    temp["probability"] = probabilities

    selected = temp[
        temp["probability"] >= threshold
    ]

    predictions = (
        selected
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    return predictions


# ---------------------------------------------------------
# THRESHOLD SEARCH
# ---------------------------------------------------------

def find_best_threshold(
    pair_df,
    probabilities,
    ground_truth,
    all_s1_ids,
    start=0.05,
    stop=0.99,
    step=0.01
):

    thresholds = np.arange(
        start,
        stop + step,
        step
    )

    results = []

    best_threshold = None
    best_score = -1

    for threshold in thresholds:

        predictions = predictions_from_probabilities(
            pair_df,
            probabilities,
            threshold
        )

        score = calculate_entity_f05(
            ground_truth,
            predictions,
            all_s1_ids
        )

        results.append({
            "threshold": threshold,
            "f05": score
        })

        if score > best_score:

            best_score = score
            best_threshold = threshold

    results_df = pd.DataFrame(results)

    return (
        best_threshold,
        best_score,
        results_df
    )
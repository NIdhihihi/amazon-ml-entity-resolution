# src/submission.py

import pandas as pd


# ---------------------------------------------------------
# EXPLODE CANDIDATE FILE
# ---------------------------------------------------------

def explode_candidate_file(candidate_df):

    rows = []

    for _, row in candidate_df.iterrows():

        s1_id = row["source1_entity_id"]
        candidate_ids = row["candidate_entity_ids"]

        if pd.isna(candidate_ids):
            continue

        for candidate_id in str(
            candidate_ids
        ).split(","):

            candidate_id = candidate_id.strip()

            if candidate_id:

                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# CREATE SUBMISSION
# ---------------------------------------------------------

def generate_submission(
    test_source1,
    candidate_pairs,
    probabilities,
    threshold,
    output_path
):

    scored = candidate_pairs[
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ].copy()

    scored["probability"] = probabilities

    selected = scored[
        scored["probability"] >= threshold
    ]

    prediction_map = (
        selected
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .apply(
            lambda x: sorted(set(x))
        )
        .to_dict()
    )

    output = pd.DataFrame({
        "source1_entity_id":
            test_source1["entity_id"]
    })

    output["matched_entity_ids"] = (
        output["source1_entity_id"]
        .map(prediction_map)
    )

    output["matched_entity_ids"] = (
        output["matched_entity_ids"]
        .apply(
            lambda x:
                ",".join(x)
                if isinstance(x, list)
                else ""
        )
    )

    output.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    return output


# ---------------------------------------------------------
# VALIDATE PREDICTIONS ARE CANDIDATES
# ---------------------------------------------------------

def validate_prediction_subset(
    submission_df,
    candidate_pairs
):

    candidate_map = (
        candidate_pairs
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    errors = []

    for _, row in submission_df.iterrows():

        s1_id = row["source1_entity_id"]

        predicted = (
            set(
                x.strip()
                for x in str(
                    row["matched_entity_ids"]
                ).split(",")
                if x.strip()
            )
        )

        allowed = candidate_map.get(
            s1_id,
            set()
        )

        invalid = predicted - allowed

        if invalid:

            errors.append({
                "source1_entity_id": s1_id,
                "invalid_ids": invalid
            })

    if errors:

        raise ValueError(
            f"Found {len(errors)} S1 entities "
            "with predictions outside candidate set."
        )

    return True
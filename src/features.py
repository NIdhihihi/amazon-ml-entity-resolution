# src/features.py

import re
import numpy as np
import pandas as pd

from rapidfuzz import fuzz


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(value):
    """
    Normalize text while preserving Unicode characters.

    This is important because the dataset contains
    multilingual business names and addresses.
    """

    if pd.isna(value):
        return ""

    value = str(value).casefold().strip()

    # Replace punctuation/symbols with spaces.
    # \w preserves Unicode letters and numbers.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def tokenize(value):
    text = normalize_text(value)

    if not text:
        return set()

    return set(text.split())


def jaccard_similarity(a, b):

    a_tokens = tokenize(a)
    b_tokens = tokenize(b)

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def safe_length_difference(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    return abs(len(a) - len(b))


# ---------------------------------------------------------
# ONE PAIR -> FEATURES
# ---------------------------------------------------------

def create_single_pair_features(row):

    s1_name = row["business_name_s1"]
    c_name = row["business_name_candidate"]

    s1_address = row["business_address_s1"]
    c_address = row["business_address_candidate"]

    s1_country = row["country_s1"]
    c_country = row["country_candidate"]

    # Normalize
    s1_name_norm = normalize_text(s1_name)
    c_name_norm = normalize_text(c_name)

    s1_address_norm = normalize_text(s1_address)
    c_address_norm = normalize_text(c_address)

    # Missing indicators
    name_missing_s1 = int(not s1_name_norm)
    name_missing_candidate = int(not c_name_norm)

    address_missing_s1 = int(not s1_address_norm)
    address_missing_candidate = int(not c_address_norm)

    # -------------------------
    # NAME FEATURES
    # -------------------------

    if s1_name_norm and c_name_norm:

        name_ratio = fuzz.ratio(
            s1_name_norm,
            c_name_norm
        ) / 100

        name_partial_ratio = fuzz.partial_ratio(
            s1_name_norm,
            c_name_norm
        ) / 100

        name_token_sort_ratio = fuzz.token_sort_ratio(
            s1_name_norm,
            c_name_norm
        ) / 100

        name_token_set_ratio = fuzz.token_set_ratio(
            s1_name_norm,
            c_name_norm
        ) / 100

    else:

        name_ratio = 0.0
        name_partial_ratio = 0.0
        name_token_sort_ratio = 0.0
        name_token_set_ratio = 0.0

    name_jaccard = jaccard_similarity(
        s1_name_norm,
        c_name_norm
    )

    # -------------------------
    # ADDRESS FEATURES
    # -------------------------

    if s1_address_norm and c_address_norm:

        address_ratio = fuzz.ratio(
            s1_address_norm,
            c_address_norm
        ) / 100

        address_partial_ratio = fuzz.partial_ratio(
            s1_address_norm,
            c_address_norm
        ) / 100

        address_token_sort_ratio = fuzz.token_sort_ratio(
            s1_address_norm,
            c_address_norm
        ) / 100

        address_token_set_ratio = fuzz.token_set_ratio(
            s1_address_norm,
            c_address_norm
        ) / 100

    else:

        address_ratio = 0.0
        address_partial_ratio = 0.0
        address_token_sort_ratio = 0.0
        address_token_set_ratio = 0.0

    address_jaccard = jaccard_similarity(
        s1_address_norm,
        c_address_norm
    )

    # -------------------------
    # COUNTRY FEATURES
    # -------------------------

    country_s1 = (
        str(s1_country).casefold().strip()
        if not pd.isna(s1_country)
        else ""
    )

    country_candidate = (
        str(c_country).casefold().strip()
        if not pd.isna(c_country)
        else ""
    )

    country_equal = int(
        bool(country_s1)
        and bool(country_candidate)
        and country_s1 == country_candidate
    )

    both_country_missing = int(
        not country_s1 and not country_candidate
    )

    # -------------------------
    # RETURN FEATURE ROW
    # -------------------------

    return {
        "name_ratio": name_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_jaccard": name_jaccard,

        "address_ratio": address_ratio,
        "address_partial_ratio": address_partial_ratio,
        "address_token_sort_ratio": address_token_sort_ratio,
        "address_token_set_ratio": address_token_set_ratio,
        "address_jaccard": address_jaccard,

        "name_length_diff": safe_length_difference(
            s1_name,
            c_name
        ),

        "address_length_diff": safe_length_difference(
            s1_address,
            c_address
        ),

        "name_missing_s1": name_missing_s1,
        "name_missing_candidate": name_missing_candidate,

        "address_missing_s1": address_missing_s1,
        "address_missing_candidate": address_missing_candidate,

        "country_equal": country_equal,
        "both_country_missing": both_country_missing,

        "name_address_mean": (
            name_ratio + address_ratio
        ) / 2,

        "name_address_min": min(
            name_ratio,
            address_ratio
        ),

        "name_address_max": max(
            name_ratio,
            address_ratio
        ),
    }


# ---------------------------------------------------------
# COMPLETE DATAFRAME -> FEATURE MATRIX
# ---------------------------------------------------------

def create_pair_features(pair_df):

    required_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "business_name_s1",
        "business_address_s1",
        "country_s1",
        "business_name_candidate",
        "business_address_candidate",
        "country_candidate",
    ]

    missing = [
        col for col in required_columns
        if col not in pair_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    features = [
        create_single_pair_features(row)
        for _, row in pair_df.iterrows()
    ]

    return pd.DataFrame(
        features,
        index=pair_df.index
    )
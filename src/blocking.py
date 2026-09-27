"""
Memory-conscious blocking / candidate generation for Amazon ML
Business Entity Resolution.

Final strategy:
    1. Country + compact normalized name prefix (3 chars)
    2. Country + normalized address prefix (6 chars)
    3. Country + informative address tokens with frequency 1-30

The rules are OR-ed together and candidates are deduplicated.

Public API:
    build_blocking_indexes(s2, s3)
    generate_candidates(s1_chunk, indexes)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


NAME_PREFIX_LEN = 3
ADDRESS_PREFIX_LEN = 6

MIN_TOKEN_FREQ = 1
MAX_TOKEN_FREQ = 30

COMMON_ADDRESS_TOKENS = {
    "street", "st", "road", "rd", "avenue", "ave",
    "drive", "dr", "lane", "ln", "boulevard", "blvd",
    "highway", "hwy", "way", "place", "pl", "court", "ct",
    "circle", "cir", "parkway", "pkwy", "building", "block",
    "unit", "floor", "flat", "apartment", "apt", "house",
    "no", "number",
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def _compact_name(name: str) -> str:
    return name.replace(" ", "")


def _name_prefix_key(
    name: str,
    country: str,
    n: int = NAME_PREFIX_LEN,
) -> Optional[str]:

    if not name:
        return None

    compact = _compact_name(name)

    if len(compact) < n:
        return None

    return f"{country}|{compact[:n]}"


def _address_prefix_key(
    address: str,
    country: str,
    n: int = ADDRESS_PREFIX_LEN,
) -> Optional[str]:

    if not address:
        return None

    return f"{country}|{address[:n]}"


def _informative_address_tokens(
    address: str,
    country: str,
) -> List[str]:

    if not address:
        return []

    tokens = [
        token
        for token in address.split()
        if (
            len(token) >= 3
            and token not in COMMON_ADDRESS_TOKENS
            and not token.isdigit()
        )
    ]

    tokens = list(dict.fromkeys(tokens))

    return [
        f"{country}|{token}"
        for token in tokens
    ]


def _iter_rows(df: pd.DataFrame):

    required = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    for row in df.itertuples(index=False):
        yield (
            getattr(row, "entity_id"),
            getattr(row, "business_name"),
            getattr(row, "business_address"),
            getattr(row, "country"),
        )


def build_blocking_indexes(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    min_token_freq: int = MIN_TOKEN_FREQ,
    max_token_freq: int = MAX_TOKEN_FREQ,
) -> Dict:

    if min_token_freq < 1:
        raise ValueError("min_token_freq must be >= 1")

    if max_token_freq < min_token_freq:
        raise ValueError(
            "max_token_freq must be >= min_token_freq"
        )

    entity_ids = []
    address_token_frequency = Counter()

    # First pass:
    #   - collect entity IDs
    #   - calculate address-token frequencies
    #
    # We intentionally do NOT store normalized rows here.
    # The source DataFrames are already available for the second pass.

    for df in (s2, s3):

        for entity_id, raw_name, raw_address, raw_country in _iter_rows(df):

            address = normalize_text(raw_address)
            country = normalize_text(raw_country)

            entity_ids.append(str(entity_id))

            token_keys = _informative_address_tokens(
                address,
                country,
            )

            address_token_frequency.update(set(token_keys))

    allowed_address_tokens = {
        token
        for token, frequency in address_token_frequency.items()
        if min_token_freq <= frequency <= max_token_freq
    }

    # Second pass:
    # Build posting lists directly from the source DataFrames.

    key_to_indices = defaultdict(list)
    row_idx = 0

    for df in (s2, s3):

        for entity_id, raw_name, raw_address, raw_country in _iter_rows(df):

            name = normalize_text(raw_name)
            address = normalize_text(raw_address)
            country = normalize_text(raw_country)

            # Name prefix
            name_key = _name_prefix_key(
                name,
                country,
            )

            if name_key is not None:
                key_to_indices[name_key].append(row_idx)

            # Address prefix
            address_key = _address_prefix_key(
                address,
                country,
            )

            if address_key is not None:
                key_to_indices[address_key].append(row_idx)

            # Frequency-filtered address tokens
            token_keys = _informative_address_tokens(
                address,
                country,
            )

            for token_key in token_keys:

                if token_key in allowed_address_tokens:
                    key_to_indices[token_key].append(row_idx)

            row_idx += 1

    return {
        "entity_ids": np.asarray(entity_ids, dtype=object),
        "key_to_indices": dict(key_to_indices),
        "address_token_frequency": dict(address_token_frequency),
        "allowed_address_tokens": allowed_address_tokens,
        "min_token_freq": min_token_freq,
        "max_token_freq": max_token_freq,
        "name_prefix_len": NAME_PREFIX_LEN,
        "address_prefix_len": ADDRESS_PREFIX_LEN,
    }


def generate_candidates(
    s1_chunk: pd.DataFrame,
    indexes: Dict,
) -> pd.DataFrame:

    required = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing = required.difference(s1_chunk.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    entity_ids = indexes["entity_ids"]
    key_to_indices = indexes["key_to_indices"]

    output = []

    for row in s1_chunk.itertuples(index=False):

        source1_id = getattr(row, "entity_id")

        name = normalize_text(
            getattr(row, "business_name")
        )

        address = normalize_text(
            getattr(row, "business_address")
        )

        country = normalize_text(
            getattr(row, "country")
        )

        candidate_positions = set()

        # Name prefix
        name_key = _name_prefix_key(
            name,
            country,
        )

        if name_key is not None:
            positions = key_to_indices.get(name_key)

            if positions:
                candidate_positions.update(positions)

        # Address prefix
        address_key = _address_prefix_key(
            address,
            country,
        )

        if address_key is not None:
            positions = key_to_indices.get(address_key)

            if positions:
                candidate_positions.update(positions)

        # Frequency-filtered address tokens
        for token_key in _informative_address_tokens(
            address,
            country,
        ):

            if token_key not in indexes["allowed_address_tokens"]:
                continue

            positions = key_to_indices.get(token_key)

            if positions:
                candidate_positions.update(positions)

        candidate_ids = sorted(
            {
                str(entity_ids[position])
                for position in candidate_positions
            }
        )

        output.append(
            {
                "source1_entity_id": str(source1_id),
                "candidate_entity_ids": ",".join(candidate_ids),
            }
        )

    return pd.DataFrame(
        output,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids",
        ],
    )


def generate_candidates_in_chunks(
    s1,
    indexes: Dict,
    chunk_size: int = 100_000,
) -> Iterable[pd.DataFrame]:

    if isinstance(s1, pd.DataFrame):

        for start in range(
            0,
            len(s1),
            chunk_size,
        ):

            chunk = s1.iloc[
                start:start + chunk_size
            ]

            yield generate_candidates(
                chunk,
                indexes,
            )

    else:

        for chunk in s1:

            yield generate_candidates(
                chunk,
                indexes,
            )

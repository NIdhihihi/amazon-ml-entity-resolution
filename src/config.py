import os

# Project paths
DATA_PATH = "/home/sagemaker-user/amazonml_data/student_resource"

TRAIN_PATH = os.path.join(DATA_PATH, "dataset", "train")
TEST_PATH = os.path.join(DATA_PATH, "dataset", "test")

OUTPUT_PATH = "/home/sagemaker-user/amazon-ml-entity-resolution/output"


# Training files
TRAIN_SOURCE1 = os.path.join(
    TRAIN_PATH, "train_source1.tsv"
)

TRAIN_SOURCE2 = os.path.join(
    TRAIN_PATH, "train_source2.tsv"
)

TRAIN_SOURCE3 = os.path.join(
    TRAIN_PATH, "train_source3.tsv"
)

GROUND_TRUTH = os.path.join(
    TRAIN_PATH, "train_ground_truth.tsv"
)


# Test files
TEST_SOURCE1 = os.path.join(
    TEST_PATH, "test_source1.tsv"
)

TEST_SOURCE2 = os.path.join(
    TEST_PATH, "test_source2.tsv"
)

TEST_SOURCE3 = os.path.join(
    TEST_PATH, "test_source3.tsv"
)


# Processing
CHUNK_SIZE = 100_000

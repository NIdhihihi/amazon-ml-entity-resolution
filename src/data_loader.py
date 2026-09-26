import os
import pandas as pd


DATA_PATH = "/home/sagemaker-user/amazonml_data/student_resource/dataset"

TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")


def load_source1_train(nrows=None):
    path = os.path.join(TRAIN_PATH, "train_source1.tsv")
    return pd.read_csv(path, sep="\t", nrows=nrows)


def load_source2_train(nrows=None):
    path = os.path.join(TRAIN_PATH, "train_source2.tsv")
    return pd.read_csv(path, sep="\t", nrows=nrows)


def load_source3_train(nrows=None):
    path = os.path.join(TRAIN_PATH, "train_source3.tsv")
    return pd.read_csv(path, sep="\t", nrows=nrows)


def load_ground_truth(nrows=None):
    path = os.path.join(TRAIN_PATH, "train_ground_truth.tsv")
    return pd.read_csv(path, sep="\t", nrows=nrows)


def read_source1_train_chunks(chunksize=100_000):
    path = os.path.join(TRAIN_PATH, "train_source1.tsv")
    return pd.read_csv(path, sep="\t", chunksize=chunksize)


def read_source2_train_chunks(chunksize=100_000):
    path = os.path.join(TRAIN_PATH, "train_source2.tsv")
    return pd.read_csv(path, sep="\t", chunksize=chunksize)


def read_source3_train_chunks(chunksize=100_000):
    path = os.path.join(TRAIN_PATH, "train_source3.tsv")
    return pd.read_csv(path, sep="\t", chunksize=chunksize)
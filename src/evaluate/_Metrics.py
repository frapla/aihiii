import itertools
import multiprocessing
import sys
from collections import defaultdict
import logging
from pathlib import Path
from typing import Literal
from tqdm import trange
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import RobustScaler

proj_dir = str(Path(__file__).absolute().parents[2])
if proj_dir not in set(sys.path):
    sys.path.append(proj_dir)
del proj_dir
from src._StandardNames import StandardNames
from src.utils.iso18571 import rating_iso_18571_short

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


def rating_iso_parallel(y_true: np.ndarray, y_pred: np.ndarray, queue: multiprocessing.Queue) -> None:
    rating = rating_iso_18571_short(signal_ref=y_true, signal_comp=y_pred)

    LOG.debug("Got ISO rating %s", rating)

    queue.put(rating)


def save_sample_score(sample_score: pd.DataFrame) -> None:
    LOG.info("Save sample score to %s", STR.fname_sample_score)
    sample_score.to_parquet(STR.fname_sample_score, index=True)


class Metrics:
    def __init__(self, fold_id: int, mode: Literal["Train", "Test"]) -> None:
        """A bunch of metrics to evaluate the performance of a classifier"""
        self.fold_id: int = fold_id
        self.mode: Literal["Train", "Test"] = mode

    def r2_score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        """R2 score, modified with - capping

        Args:
            y_true (pd.DataFrame): true values of shape (n_samples, n_labels)
            y_pred (pd.DataFrame): predicted values of shape (n_samples, n_labels)

        Returns:
            pd.DataFrame: scores [0; 1] of shape (1, n_labels), multi index with fold and mode
        """
        LOG.debug("Calculate capped R2 score for shape %s", y_true.shape)
        # calculate scores
        scores = r2_score(y_true=y_true, y_pred=y_pred.loc[y_true.index], multioutput="raw_values")

        # capping
        scores = np.where(scores < 0, 0, scores)

        # format output
        scores = pd.DataFrame(scores, columns=[0], index=y_true.columns).T
        scores.index = self.__get_multi_index()

        LOG.debug("Got capped R2 scores of shape %s", scores.shape)

        # per sample score
        if self.mode == "Test" and self.fold_id == -1:
            # absolute error per normalized sample with [0, inf) where 0 is best
            scaler = RobustScaler()
            y_true_scaled = scaler.fit_transform(y_true)
            y_pred_scaled = scaler.transform(y_pred.loc[y_true.index])
            save_sample_score(
                sample_score=pd.DataFrame(
                    np.abs(y_true_scaled - y_pred_scaled),
                    index=y_true.index,
                    columns=y_true.columns,
                )
            )

        return scores

    def f1_score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame, n_classes: int) -> pd.DataFrame:
        """Calculate the F1 score

        Args:
            y_true (pd.DataFrame): true values of shape (n_samples, n_labels)
            y_pred (pd.DataFrame): predicted values of shape (n_samples, n_labels)
            n_classes (int): number of classes

        Returns:
            pd.DataFrame: f1 scores [0; 1] of shape (n_classes, n_labels), multi index with class, fold, and mode
        """
        LOG.debug("Calculate F1 score for label shape %s with %s classen", y_true.shape, n_classes)
        classes = list(range(n_classes))
        # calculate scores
        score: pd.DataFrame = pd.concat(
            [
                pd.DataFrame(
                    f1_score(
                        y_true[c],
                        y_pred.loc[y_true.index, c],
                        average="weighted",
                        labels=classes,
                        zero_division=0,
                    ),
                    columns=[c],
                    index=classes,
                )
                for c in y_true.columns
            ],
            axis=1,
        )
        score.fillna(0, inplace=True)
        score.index.name = STR.target_class

        # format
        score[STR.data] = self.mode
        score[STR.fold] = self.fold_id
        score.set_index([STR.fold, STR.data], inplace=True, append=True)

        LOG.debug("Got F1 scores of shape %s:\s%s", score.shape, score)

        if self.mode == "Test" and self.fold_id == -1:
            # absolute error per sample with [0, inf) where 0 is best
            save_sample_score(
                sample_score=pd.DataFrame(
                    np.abs(y_true - y_pred.loc[y_true.index]),
                    index=y_true.index,
                    columns=y_true.columns,
                )
            )

        return score

    def iso1871_score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        """Calculate the ISO 1871 score

        Args:
            y_true (pd.DataFrame): true values with multindex with levels TIME and ID
            y_pred pd.DataFrame): predicted values with multindex with levels TIME and ID
        Returns:
            pd.DataFrame: iso 1871 scores [0; 1] of shape (1, n_channels)
        """
        LOG.debug("Calculate ISO18571 score for label shape %s", y_true.shape)
        # sample
        sample_ids = y_true.index.unique(STR.id)
        n_samples_max = sample_ids.shape[0]  # 100 # this is a quick switch for debugging / testing purpose

        if sample_ids.shape[0] > n_samples_max:
            # down sampling 2,000 seem statistically acceptable
            rng = np.random.default_rng(seed=42)
            idxs = rng.choice(sample_ids, n_samples_max, replace=False)
            LOG.debug("Calculate ISO18571 score for label shape %s - reduced to 2000 samples", sample_ids.shape[0])
        else:
            idxs = sample_ids.to_numpy()

        scores = defaultdict(list)
        LOG.info(
            "Calculate ISO18571 score for %s samples x %s channels = %s signal pairs",
            idxs.shape[0],
            y_true.shape[1],
            idxs.shape[0] * y_true.shape[1],
            # batch_size,
        )
        with multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=multiprocessing.cpu_count()) as pool:
            scores = pool.starmap(
                func=rating_iso_18571_short,
                iterable=zip(
                    y_true.loc[(slice(60, 120), idxs), :].unstack(STR.id).values.T,
                    y_pred.loc[(slice(60, 120), idxs), :].unstack(STR.id).values.T,
                ),
            )

        # per sample score
        if self.mode == "Test" and self.fold_id == -1:
            # invert ISO score per sample to [0, inf) where 0 is best
            save_sample_score(
                sample_score=pd.DataFrame(
                    1 - np.array(scores).reshape(-1, idxs.shape[0]).T,
                    index=idxs,
                    columns=y_true.columns,
                )
            )

        # format
        scores = pd.DataFrame(
            data=np.median(np.array(scores).reshape(-1, idxs.shape[0]), axis=1),
            index=y_true.columns,
            columns=[0],
        ).T
        scores.index = self.__get_multi_index()

        LOG.info("Got ISO scores of shape %s", scores.shape)
        return scores

    def __get_multi_index(self):
        return pd.MultiIndex.from_tuples([(self.fold_id, self.mode)], names=[STR.fold, STR.data])

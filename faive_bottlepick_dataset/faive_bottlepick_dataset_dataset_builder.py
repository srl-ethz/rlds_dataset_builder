from typing import Iterator, Tuple, Any
import random
import os
import glob
from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from dataset_parser.faive import load_dataset
from dataset_parser.conversion_config import conversion_config


class FaiveBottlePickDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Collection of teleop datasets with delta EEF actions and absolute joint angles. Data can include wrist images, if availeble, otherwise, zeros are used.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Zed camera full desk view, RGB observation",
                                    ),
                                    "wrist": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="OAK-D wrist camera, RGB observation",
                                    ),
                                    "secondary": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="OAK-D top camera, RGB observation",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(17,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [6-dim EEF pose (Euler + translation) relative to the robot base,"
                                        "11-dim Faive joint angles]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(17,),
                                dtype=np.float32,
                                doc="Robot action, [3-dim eulerXYZ rot, 3-dim translation, faive joint angles],"
                                "11-dim Faive joint angles]. First 6 are absolute deltas in the world frame (robot base), last 11 are absolute.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        cfg = conversion_config['faive']
        return {
            "train": self._generate_examples(cfg),
            # "val": self._generate_examples(test_set),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, cfg) -> Iterator[Tuple[str, Any]]:
        """
        cfg: config dict

        Generator of examples for each split.
        The input path will be globbed.

        """

        def _parse_example(episode_path, episode_data):

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(episode_data):
                # compute Kona language embedding
                # language_embedding = self._embed(
                #     [step['language_instruction']])[0].numpy()
                language_desc = episode_data[i]['language_instruction']
                language_embedding = self._embed([language_desc])[0].numpy()
                episode.append(
                    {
                        **episode_data[i],
                        "discount": 1.0,
                        "reward": float(i == (len(episode_data) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(episode_data) - 1),
                        "is_terminal": i == (len(episode_data) - 1),
                        # "language_instruction": language_desc,
                        "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # for smallish datasets, use single-thread parsing
        for episode_path, episode_data in load_dataset(cfg):
            yield _parse_example(episode_path, episode_data)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

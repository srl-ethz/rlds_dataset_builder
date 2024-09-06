from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tqdm
import time
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from dataset_parser import build_obs_action_list
from dataset_parser.hoi4d import parse_dataset



class Hoi4dDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
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
                                        doc="RGB image of the scene.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(102,),
                                        dtype=np.float32,
                                        doc="At the moment, just zeros.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(102,),
                                dtype=np.float32,
                                doc="Two human hands, represented as delta poses (using XYZ Euler) and 45-dimensional MANO parameters. [pose_r, pose_l, mano_r, mano_l]",
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
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(
                root_dir="/data2/erbauer/hoi4d"
            ),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, root_dir) -> Iterator[Tuple[str, Any]]:
        """
        Generator of examples for each split.
        The input path will be globbed.

        """

        def _parse_example(episode_id, data_dict):

            episode_data = build_obs_action_list(
                dataset_name="hoi4d", human_data_dict=data_dict
            )

            #
            # # TODO: get better language descriptions
            language_desc = ""
            #
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(episode_data):
                # compute Kona language embedding
                # language_embedding = self._embed([step["language_instruction"]])[
                # 0
                # ].numpy()
                language_embedding = self._embed([language_desc])[0].numpy()
                episode.append(
                    {
                        **step,
                        "discount": 1.0,
                        "reward": float(i == (len(episode_data) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(episode_data) - 1),
                        "is_terminal": i == (len(episode_data) - 1),
                        "language_instruction": language_desc,
                        "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_id}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_id, sample

        # for smallish datasets, use single-thread parsing
        for episode_id, data_dict in parse_dataset(root_dir):
            yield _parse_example(episode_id, data_dict)

        # for seq_path, seq_data, image_
        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(configs)
        #         | beam.Map(_parse_example)
        # )

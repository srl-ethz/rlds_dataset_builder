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


class FaivePokemonPickDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.4")
    RELEASE_NOTES = {
        "1.0.4": "Pokemon pick data without wrist images, actions are absolute poses [euler, trans] and gripper angles. Includes separate keys for robot and hand proprio.",
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
                                    "robot_pose": tfds.features.Tensor(
                                        shape=(6,),
                                        dtype=np.float32,
                                        doc="Robot pose, [3-dim eulerXYZ rot, 3-dim translation]",
                                    ),
                                    "gripper_angles": tfds.features.Tensor(
                                        shape=(11,),
                                        dtype=np.float32,
                                        doc="Gripper angles (absolute).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(17,),
                                dtype=np.float32,
                                doc="Robot action, [3-dim eulerXYZ rot, 3-dim translation, faive joint angles],"
                                "11-dim Faive joint angles]",
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
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """
        episode_paths: List of file paths.

        Generator of examples for each split.
        The input path will be globbed.

        """

        def _parse_example(episode_path, episode_data):
            # load raw data --> this should change for your dataset
            # this is a list of dicts in our case
            # data = np.load(episode_path, allow_pickle=True)

            # compute robot_actions[i] (deltas that the model should learn) = robot_actions[i+1] - robot_actions[i]
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(episode_data):
                # compute Kona language embedding
                # language_embedding = self._embed(
                #     [step['language_instruction']])[0].numpy()
                language_embedding = self._embed([step["language_instruction"]])[0].numpy()
                episode.append(
                    {
                        **step,
                        "discount": 1.0,
                        "reward": float(i == (len(episode_data) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(episode_data) - 1),
                        "is_terminal": i == (len(episode_data) - 1),
                        "language_embedding": language_embedding,
                    }
               )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample


        # for smallish datasets, use single-thread parsing
        dataset_names = ["pokemon_pick_basic", "pokemon_pick_v2", "pokemon_pick_v3"]
        for episode_path, episode_data in load_dataset(conversion_config["faive"], selected_dataset_names=dataset_names):
            yield _parse_example(episode_path, episode_data)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

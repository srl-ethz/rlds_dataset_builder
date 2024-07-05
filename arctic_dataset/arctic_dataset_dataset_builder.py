from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tqdm
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from dexformer.dataset import h5_tools
from dexformer.utils.dataset_utils import (
    parse_state_from_sync_df,
    parse_actions_from_sync_df,
)


from human_data_preprocessing import (
    parse_sequences,
    extract_sequence_data,
    build_obs_action_list,
)


class ArcticDataset(tfds.core.GeneratorBasedBuilder):
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
                                        shape=(540, 960, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="RGB image of the scene.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(17,),
                                        dtype=np.float32,
                                        doc="Two human hands, absolute poses (using XYZ Euler) in the workspace and 45-dimensional MANO parameters. [pose_r, pose_l, mano_r, mano_l]",
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
                            # "language_instruction": tfds.features.Text(
                            #     doc="Language Instruction."
                            # ),
                            # "language_embedding": tfds.features.Tensor(
                            #     shape=(512,),
                            #     dtype=np.float32,
                            #     doc="Kona language embedding. "
                            #     "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            # ),
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
                seq_dir="/home/erbauer/srl-nas-faive/Datasets/ARCTIC/outputs/processed/seqs/",
                im_dir="/home/erbauer/srl-nas-faive/Datasets/ARCTIC/data/arctic_data/data/images/",
            ),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, seq_dir, im_dir) -> Iterator[Tuple[str, Any]]:
        """
        Generator of examples for each split.
        The input path will be globbed.

        """

        def _parse_example(seq_data, image_generator):
            # load raw data --> this should change for your dataset
            # this is a list of dicts in our case
            # data = np.load(episode_path, allow_pickle=True)
            view_ids = [str(i) for i in range(9)]

            data_dict = extract_sequence_data(seq_data, image_generator, view_ids)

            episode_data = build_obs_action_list(data_dict)

            #
            # # TODO: get better language descriptions
            # language_desc = task_name
            #
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(episode_data):
                # compute Kona language embedding
                # language_embedding = self._embed(
                #     [step['language_instruction']])[0].numpy()
                # language_embedding = self._embed([language_desc])[0].numpy()
                episode.append(
                    {
                        **step,
                        "discount": 1.0,
                        "reward": float(i == (len(sync_dataframe) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(sync_dataframe) - 1),
                        "is_terminal": i == (len(sync_dataframe) - 1),
                        # "language_instruction": language_desc,
                        # "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # for smallish datasets, use single-thread parsing
        for seq_data, im_data in parse_sequences(seq_dir, im_dir):
            image_generator = load_images(im_data)
            yield _parse_example(seq_data, image_generator)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

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

import sys

file_dir = os.path.abspath(__file__)
sys.path.append(os.path.join(file_dir, "..", "..", "human_data_preprocessing/"))
from read_actic_processed_seqs import parse_sequences, load_images


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
                                        doc="Zed camera full desk view, RGB observation",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(360, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="OAK-D wrist camera, RGB observation",
                                    ),
                                    "top_image": tfds.features.Image(
                                        shape=(360, 640, 3),
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
                                doc="Robot action, consists of [6-dim EEF twist relative to robot base,"
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
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(seq_dir="", im_dir=""),
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

            world_coord_info = seq_data["world_coord"]
            cam_coord_info = seq_data["cam_coord"]
            seq_info_2d = seq_data["2d"]
            bbox_data = seq_data["bbox"]
            params = seq_data["params"]

            # ## GEOMETRY PARAMS
            # # all in world world frame
            #
            # # mano rotations
            # rot_r = params["rot_r"]
            # rot_l = params["rot_l"]
            #
            # # mano translations
            # trans_r = params["trans_r"]
            # trans_l = params["trans_l"]
            #
            # # smplx translation and rotation
            # rot_body = params["smplx_global_orient"]
            # trans_body = params["smplx_trans"]
            #
            # # 4x4 transform from world to ego camera coordinates
            #
            # arctic stuff
            num_frames = seq_info_2d["joints.right"].shape[0]

            # TODO: Before implementing more, we must discuss what to put in the dataset

            for idx in tqdm.tqdm(range(num_frames)):
                imgs = next(image_generator)
                out_imgs = []
                for view_id in view_ids:
                    keypoints_right_2d = seq_info_2d["joints.right"][
                        idx, int(view_id)
                    ]  # 21 x 2
                    keypoints_left_2d = seq_info_2d["joints.left"][
                        idx, int(view_id)
                    ]  # 21 x 2

                    img = imgs[view_id]

            #
            # topic_arrays = h5_tools.load_topic_arrays_h5(episode_path)
            # sync_dataframe = h5_tools.build_synchronized_dataframe(
            #     '/franka_pose', topic_arrays, 10)
            #
            # # parse state from sync dataframe (EEF euler angles + translation, Faive joint angles)
            # robot_states = parse_state_from_sync_df(
            #     sync_dataframe).astype(np.float32)
            #
            # # parse actions from sync dataframe (EEF twist, Faive joint angles)
            # robot_actions = parse_actions_from_sync_df(
            #     sync_dataframe).astype(np.float32)
            #
            # task_name = os.path.dirname(episode_path).split(
            #     '/')[-1].replace('_', ' ')
            #
            # # TODO: get better language descriptions
            # language_desc = task_name
            #
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in sync_dataframe.iterrows():
                # compute Kona language embedding
                # language_embedding = self._embed(
                #     [step['language_instruction']])[0].numpy()
                language_embedding = self._embed([language_desc])[0].numpy()
                episode.append(
                    {
                        "observation": {
                            "image": step["/zed/zed_node/rgb/image_rect_color"],
                            "wrist_image": step["/oakd_wrist_view/color"],
                            "top_image": step["/oakd_top_view/color"],
                            "state": robot_states[i],
                        },
                        "action": robot_actions[i],
                        "discount": 1.0,
                        "reward": float(i == (len(sync_dataframe) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(sync_dataframe) - 1),
                        "is_terminal": i == (len(sync_dataframe) - 1),
                        "language_instruction": language_desc,
                        "language_embedding": language_embedding,
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

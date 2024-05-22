from typing import Iterator, Tuple, Any
import random
import os
import glob
from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from dexformer.dataset import h5_tools
from dexformer.utils.dataset_utils import (
    parse_state_from_sync_df,
    parse_actions_from_sync_df,
)


class FaiveDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.4")
    RELEASE_NOTES = {
        "1.0.4": "Plush pick v2 data without wrist images, actions are absolute poses [euler, trans] and gripper angles. Includes separate keys for robot and hand proprio.",
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
        file_pattern = "/home/erbauer/srl-nas-faive/Datasets/Dexformer/datasets/plush_pick_v2/converted/episode_*_success.h5"
        files = glob.glob(file_pattern)

        train_test_ratio = 0.95

        random.shuffle(files)

        num_samples = len(files)
        num_train = int(num_samples * train_test_ratio)

        train_set = files[:num_train]
        test_set = files[num_train:]

        return {
            "train": self._generate_examples(train_set),
            "val": self._generate_examples(test_set),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """
        episode_paths: List of file paths.

        Generator of examples for each split.
        The input path will be globbed.

        """

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # this is a list of dicts in our case
            # data = np.load(episode_path, allow_pickle=True)
            topic_arrays = h5_tools.load_topic_arrays_h5(episode_path)
            sync_dataframe = h5_tools.build_synchronized_dataframe(
                "/franka_pose", topic_arrays, 10
            )

            # parse state from sync dataframe (EEF euler angles + translation, Faive joint angles)
            robot_states = parse_state_from_sync_df(sync_dataframe).astype(np.float32)

            # parse actions from sync dataframe (EEF twist, Faive joint angles)
            # NOTE: we don't use twist for now, it does not seem to perform well
            # robot_actions = parse_actions_from_sync_df(sync_dataframe).astype(
            #     np.float32
            # )

            # NOTE: instead, use the state as action
            robot_actions = deepcopy(robot_states)

            # compute robot_actions[i] (deltas that the model should learn) = robot_actions[i+1] - robot_actions[i]
            # for last element, pad with 0

            robot_poses, gripper_angles = robot_actions[:, :6], robot_actions[:, 6:]
            # robot_pose_diffs = np.diff(robot_poses, axis=0)
            # robot_pose_diffs = np.vstack(
            #     (robot_pose_diffs, np.zeros_like(robot_pose_diffs[-1]))
            # )
            #
            # robot_actions = np.hstack((robot_pose_diffs, gripper_angles))
            assert robot_actions.shape == robot_states.shape
            #
            task_name = os.path.dirname(episode_path).split("/")[-1].replace("_", " ")

            # TODO: get better language descriptions
            language_desc = "pick up the plush toy"

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
                            "robot_pose": robot_poses[i],
                            "gripper_angles": gripper_angles[i],
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

        # create list of all examples
        print(f"Selected {len(episode_paths)}")

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

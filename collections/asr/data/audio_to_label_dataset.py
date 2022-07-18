# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_label
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list


def get_classification_label_dataset(featurizer, config: dict) -> audio_to_label.AudioToClassificationLabelDataset:
    """
    Instantiates a Classification AudioLabelDataset.

    Args:
        config: Config of the AudioToClassificationLabelDataset.

    Returns:
        An instance of AudioToClassificationLabelDataset.
    """
    dataset = audio_to_label.AudioToClassificationLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        is_regression_task=config.get('is_regression_task', False),
    )
    return dataset


def get_speech_label_dataset(featurizer, config: dict) -> audio_to_label.AudioToSpeechLabelDataset:
    """
    Instantiates a Speech Label (e.g. VAD, speaker recognition) AudioLabelDataset.

    Args:
        config: Config of the AudioToSpeechLabelDataSet.

    Returns:
        An instance of AudioToSpeechLabelDataset.
    """
    dataset = audio_to_label.AudioToSpeechLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        window_length_in_sec=config.get('window_length_in_sec', 0.31),
        shift_length_in_sec=config.get('shift_length_in_sec', 0.01),
        normalize_audio=config.get('normalize_audio', False),
    )
    return dataset


def get_tarred_classification_label_dataset(
    featurizer, config: dict, shuffle_n: int, global_rank: int, world_size: int
) -> audio_to_label.TarredAudioToClassificationLabelDataset:
    """
    Instantiates a Classification TarredAudioLabelDataset.

    Args:
        config: Config of the TarredAudioToClassificationLabelDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.

    Returns:
        An instance of TarredAudioToClassificationLabelDataset.
    """
    dataset = audio_to_label.TarredAudioToClassificationLabelDataset(
        audio_tar_filepaths=config['tarred_audio_filepaths'],
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        shuffle_n=shuffle_n,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        global_rank=global_rank,
        world_size=world_size,
        is_regression_task=config.get('is_regression_task', False),
    )
    return dataset


def get_tarred_speech_label_dataset(
    featurizer, config: dict, shuffle_n: int, global_rank: int, world_size: int,
) -> audio_to_label.TarredAudioToSpeechLabelDataset:
    """
    InInstantiates a Speech Label (e.g. VAD, speaker recognition) TarredAudioLabelDataset.

    Args:
        config: Config of the TarredAudioToSpeechLabelDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.

    Returns:
        An instance of TarredAudioToSpeechLabelDataset.
    """
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        dataset = audio_to_label.TarredAudioToSpeechLabelDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            labels=config['labels'],
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            trim=config.get('trim_silence', False),
            window_length_in_sec=config.get('window_length_in_sec', 8),
            shift_length_in_sec=config.get('shift_length_in_sec', 0.075),
            normalize_audio=config.get('normalize_audio', False),
            shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
            global_rank=global_rank,
            world_size=world_size,
        )

        datasets.append(dataset)

    if len(datasets) > 1:
        return ChainDataset(datasets)
    else:
        return datasets[0]

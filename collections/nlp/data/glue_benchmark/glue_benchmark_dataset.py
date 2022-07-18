# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

# Some code of this file was adapted from the HuggingFace library available at
# https://github.com/huggingface/transformers

import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.glue_benchmark.data_processors import (
    ColaProcessor,
    MnliMismatchedProcessor,
    MnliProcessor,
    MrpcProcessor,
    QnliProcessor,
    QqpProcessor,
    RteProcessor,
    Sst2Processor,
    StsbProcessor,
    WnliProcessor,
)
from nemo.core.classes import Dataset
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType
from nemo.utils import logging

__all__ = ['GLUEDataset']

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}
output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


class GLUEDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(
                tuple('B'), RegressionValuesType() if self.task_name == 'sts-b' else CategoricalValuesType()
            ),
        }

    def __init__(
        self, file_name: str, task_name: str, tokenizer: TokenizerSpec, max_seq_length: str, use_cache: bool = True,
    ):
        """
        Processes GLUE datasets
        Args:
            file_name: path to file
            task_name: GLUE task name
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            use_cache: whether to use data cache
        """
        logging.info(f'Processing {file_name}')
        data_dir, file_name = os.path.split(file_name)
        file_name = file_name[:-4]
        self.tokenizer = tokenizer
        evaluate = False if 'train' in file_name else True

        if task_name not in processors:
            raise ValueError(f'{task_name} not supported. Choose from {processors.keys()}')

        if task_name == 'mnli' and 'dev_mismatched' in file_name:
            self.task_name = 'mnli-mm'
        else:
            self.task_name = task_name

        processor = processors[self.task_name]()
        output_mode = output_modes[self.task_name]
        self.label_list = processor.get_labels()

        self.examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        processor_name = type(processor).__name__
        vocab_size = getattr(tokenizer, "vocab_size", 0)
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                processor_name, file_name, tokenizer.name, str(max_seq_length), str(vocab_size)
            ),
        )

        if use_cache and os.path.exists(cached_features_file):
            logging.info(f"loading from {cached_features_file}")
            with open(cached_features_file, "rb") as reader:
                self.features = pickle.load(reader)
        else:
            token_params = {
                'bos_token': None,
                'eos_token': tokenizer.eos_token,
                'pad_token': tokenizer.pad_token,
                'cls_token': tokenizer.cls_token,
                'sep_token_extra': tokenizer.eos_token if 'roberta' in tokenizer.name.lower() else None,
            }

            self.features = self.convert_examples_to_features(
                self.examples, self.label_list, max_seq_length, tokenizer, output_mode, **token_params
            )
            master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if master_device:
                logging.info(f'Saving train features into {cached_features_file}')
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(self.features, writer)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return (
            np.array(feature.input_ids),
            np.array(feature.segment_ids),
            np.array(feature.input_mask, dtype=np.long),
            np.array(feature.label_id),
        )

    def convert_examples_to_features(
        self,
        examples: List[str],
        label_list: List[int],
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        output_mode: str,
        bos_token: str = None,
        eos_token: str = '[SEP]',
        pad_token: str = '[PAD]',
        cls_token: str = '[CLS]',
        sep_token_extra: str = None,
        cls_token_at_end: bool = False,
        cls_token_segment_id: int = 0,
        pad_token_segment_id: int = 0,
        pad_on_left: bool = False,
        mask_padding_with_zero: bool = True,
        sequence_a_segment_id: int = 0,
        sequence_b_segment_id: int = 1,
    ):
        """
        Loads a data file into a list of `InputBatch`s.
        The `cls_token_at_end` defines the location of the CLS token:

            * False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            * True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]

        The `cls_token_segment_id` defines the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        
        The convention in BERT is:
        
            a. For sequence pairs:
                * tokens:   [CLS] is this jack ##ville ? [SEP] no it is not . [SEP]
                * type_ids:   0   0  0    0    0       0   0   1  1  1  1   1   1
            b. For single sequences:
                * tokens:   [CLS] the dog is hairy . [SEP]
                * type_ids:   0   0   0   0  0     0   0

        Where "type_ids" are used to indicate whether this is the first
        sequence or the second sequence. The embedding vectors for `type=0`
        and `type=1` were learned during pre-training and are added to the
        wordpiece embedding vector (and position vector). This is
        not *strictly* necessarysince the [SEP] token unambiguously separates
        the sequences, but it makes it easier for the model to learn
        the concept of sequences.
        For classification tasks, the first vector (corresponding to [CLS])
        is used as as the "sentence vector". Note that this only makes sense
        because the entire model is fine-tuned.
        
        The convention for NMT is:
        
            a. For sequence pairs:
                * tokens:<BOS> is this jack ##ville ? <EOS> <BOS> no it is not . <EOS>
                * type_ids:0   0  0    0    0       0   0     1   1  1  1  1   1   1
            b. For single sequences:
                * tokens:   <BOS> the dog is hairy . <EOS>
                * type_ids:   0   0   0   0  0     0   0

        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            if example.label == "-":  # skip examples without a consensus label (e.g. in SNLI data set)
                continue
            if ex_index % 10000 == 0:
                logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.text_to_tokens(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.text_to_tokens(example.text_b)

                special_tokens_count = 2 if eos_token else 0
                special_tokens_count += 1 if sep_token_extra else 0
                special_tokens_count += 2 if bos_token else 0
                special_tokens_count += 1 if cls_token else 0
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            else:
                special_tokens_count = 1 if eos_token else 0
                special_tokens_count += 1 if sep_token_extra else 0
                special_tokens_count += 1 if bos_token else 0
                if len(tokens_a) > max_seq_length - special_tokens_count:
                    tokens_a = tokens_a[: max_seq_length - special_tokens_count]
            # Add special tokens to sequence_a
            tokens = tokens_a
            if bos_token:
                tokens = [bos_token] + tokens
            if eos_token:
                tokens += [eos_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            # Add sequence separator between sequences
            if tokens_b and sep_token_extra:
                tokens += [sep_token_extra]
                segment_ids += [sequence_a_segment_id]

            # Add special tokens to sequence_b
            if tokens_b:
                if bos_token:
                    tokens += [bos_token]
                    segment_ids += [sequence_b_segment_id]
                tokens += tokens_b
                segment_ids += [sequence_b_segment_id] * (len(tokens_b))
                if eos_token:
                    tokens += [eos_token]
                    segment_ids += [sequence_b_segment_id]

            # Add classification token - for BERT models
            if cls_token:
                if cls_token_at_end:
                    tokens += [cls_token]
                    segment_ids += [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids
            input_ids = tokenizer.tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            pad_token_id = tokenizer.tokens_to_ids([pad_token])[0]
            if pad_on_left:
                input_ids = ([pad_token_id] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token_id] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            if len(input_ids) != max_seq_length:
                raise ValueError("input_ids must be of length max_seq_length")
            if len(input_mask) != max_seq_length:
                raise ValueError("input_mask must be of length max_seq_length")
            if len(segment_ids) != max_seq_length:
                raise ValueError("segment_ids must be of length max_seq_length")
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = np.float32(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index < 5:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % " ".join(list(map(str, tokens))))
                logging.info("input_ids: %s" % " ".join(list(map(str, input_ids))))
                logging.info("input_mask: %s" % " ".join(list(map(str, input_mask))))
                logging.info("segment_ids: %s" % " ".join(list(map(str, segment_ids))))
                logging.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)
            )
        return features

    def _truncate_seq_pair(self, tokens_a: str, tokens_b: str, max_length: int):
        """Truncates a sequence pair in place to the maximum length.

        This will always truncate the longer sequence one token at a time.
        This makes more sense than truncating an equal percent
        of tokens from each, since if one sequence is very short then each token
        that's truncated likely contains more information than a longer sequence.
        """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data.

    Args:
        input_ids: input/token ids
        input_mask: masks out subword tokens
        segment_ids: distinguish one sentence from the other one (if present)
        label_ids: label for the current example
    """

    def __init__(
        self, input_ids: List[int], input_mask: List[int], segment_ids: List[int], label_id: Union[float, int]
    ):
        """Initialized InputFeatures."""
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

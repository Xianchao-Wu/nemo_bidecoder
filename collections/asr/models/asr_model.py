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
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from omegaconf import OmegaConf

from nemo.core.classes import ModelPT, exportable, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.utils import model_utils

__all__ = ['ASRModel']


class ASRModel(ModelPT, ABC):
    @abstractmethod
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        """
        Takes paths to audio files and returns text transcription
        Args:
            paths2audio_files: paths to audio fragment to be transcribed

        Returns:
            transcription texts
        """
        pass

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        #import ipdb; ipdb.set_trace()
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_wer': wer_num / wer_denom}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        #import ipdb; ipdb.set_trace()
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean, 'test_wer': wer_num / wer_denom}
        return {'test_loss': val_loss_mean, 'log': tensorboard_logs}

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        # recursively walk the subclasses to generate pretrained model info
        list_of_models = model_utils.resolve_subclass_pretrained_model_info(cls)
        return list_of_models


class ExportableEncDecModel(Exportable):
    """
    Simple utiliy mix-in to export models that consist of encoder/decoder pair 
    plus pre/post processor, but have to be exported as encoder/decoder pair only
    (covers most ASR classes)
    """

    @property
    def input_module(self):
        return self.encoder

    @property
    def output_module(self):
        return self.decoder

    def forward_for_export(self, input, length=None):
        if hasattr(self.input_module, 'forward_for_export'):
            encoder_output = self.input_module.forward_for_export(input, length)
        else:
            encoder_output = self.input_module(input, length)
        if isinstance(encoder_output, tuple):
            decoder_input = encoder_output[0]
        else:
            decoder_input = encoder_output
        if hasattr(self.output_module, 'forward_for_export'):
            return self.output_module.forward_for_export(decoder_input)
        else:
            return self.output_module(decoder_input)

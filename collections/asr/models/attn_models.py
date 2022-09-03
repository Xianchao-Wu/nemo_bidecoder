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
import copy
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import ChainDataset
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
#from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.attn_ctc import CTC # from wenet
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

from nemo.collections.asr.utils.common import (IGNORE_ID, BLANK_ID, BLANK_STR, UNK_ID, UNK_STR,  
                                               SOSEOS_STR, add_sos_eos, log_add,
                                               remove_duplicates_and_blank, th_accuracy,
                                               reverse_pad_list) # from wenet
from nemo.collections.asr.modules.label_smoothing_loss import LabelSmoothingLoss # from wenet
from nemo.collections.asr.modules.cmvn_model import GlobalCMVN
from nemo.collections.asr.utils.cmvn import load_cmvn

from nemo.collections.asr.utils.mask import (make_pad_mask, mask_finished_scores,
                                             mask_finished_preds, subsequent_mask)# from wenet
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import (AudioSignal, LabelsType, LengthsType, 
                                    LogprobsType, NeuralType, SpectrogramType,
                                    LossType, ElementType)
from nemo.utils import logging

__all__ = ['EncDecCTCAttnModel']


class EncDecCTCAttnModel(ASRModel, ExportableEncDecModel, ASRModuleMixin):
    """Base class for encoder decoder CTC-based models."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5Base-En",
            description="QuartzNet15x5 model trained on six datasets: LibriSpeech, Mozilla Common Voice (validated clips from en_1488h_2019-12-10), WSJ, Fisher, Switchboard, and NSC Singapore English. It was trained with Apex/Amp optimization level O1 for 600 epochs. The model achieves a WER of 3.79% on LibriSpeech dev-clean, and a WER of 10.05% on dev-other. Please visit https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels for further details.",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_jasper10x5dr",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_jasper10x5dr",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_jasper10x5dr/versions/1.0.0rc1/files/stt_en_jasper10x5dr.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ca_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ca_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_quartznet15x5/versions/1.0.0rc1/files/stt_ca_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_quartznet15x5/versions/1.0.0rc1/files/stt_it_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_quartznet15x5/versions/1.0.0rc1/files/stt_fr_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_quartznet15x5/versions/1.0.0rc1/files/stt_es_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_quartznet15x5/versions/1.0.0rc1/files/stt_de_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_pl_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_pl_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_quartznet15x5/versions/1.0.0rc1/files/stt_pl_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_quartznet15x5/versions/1.0.0rc1/files/stt_ru_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_512/versions/1.0.0rc1/files/stt_zh_citrinet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_zh_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_zh_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="asr_talknet_aligner",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:asr_talknet_aligner",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/asr_talknet_aligner/versions/1.0.0rc1/files/qn5x5_libri_tts_phonemes.nemo",
        )
        results.append(model)

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size
        # add <blank>, <unk> and <sos/eos> to self.cfg.decoder.vocabulary
        self.extend_vocab(cfg.decoder.vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)
        #import ipdb; ipdb.set_trace()
        #cmvn_file = self._cfg.get('cmvn', None)
        #if cmvn_file is not None and os.path.exists(cmvn_file):
        #    cmvn_mean, cmvn_istd = load_cmvn(cmvn_file, True)
        #    self.global_cmvn = GlobalCMVN(
        #        torch.from_numpy(cmvn_mean).float(),
        #        torch.from_numpy(cmvn_istd).float()
        #    )
        #else:
        #    self.global_cmvn = None

        self.preprocessor = EncDecCTCAttnModel.from_config_dict(self._cfg.preprocessor)
        #import ipdb; ipdb.set_trace()
        self.encoder = EncDecCTCAttnModel.from_config_dict(self._cfg.encoder)

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                (not self._cfg.decoder.feat_in or self._cfg.decoder.feat_in == -1)
                and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.decoder.num_classes < 1 and self.cfg.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.decoder.num_classes, len(self.cfg.decoder.vocabulary)
                    )
                )
                self.cfg.decoder["num_classes"] = len(self.cfg.decoder.vocabulary) 
                # NOTE already includes <sos/eos>

        #import ipdb; ipdb.set_trace()
        self.decoder = EncDecCTCAttnModel.from_config_dict(self._cfg.decoder)

        self.ctc_weight = self._cfg.get('ctc_weight', 0.0)
        self.reverse_weight = self._cfg.get('reverse_weight', 0.0)
        self.sos = self.decoder.num_classes_with_blank - 1 # 3608 - 1 = 3607
        self.eos = self.decoder.num_classes_with_blank - 1 # 3608 - 1 = 3607
        self.ignore_id = IGNORE_ID

        #import ipdb; ipdb.set_trace()
        #self.loss = CTCLoss(
        #    num_classes=self.decoder.num_classes_with_blank - 1,
        #    zero_infinity=True,
        #    reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        #)
        self.ctc = CTC(self.decoder.num_classes_with_blank, # nemo.collections.asr.losses.attn_ctc.CTC 
            self.encoder._feat_out,
            normalize_length = self._cfg.get('length_normalized_loss', False))

        self.criterion_att = LabelSmoothingLoss(
            size = self.decoder.num_classes_with_blank, 
            padding_idx = self.ignore_id, 
            #smoothing=self.cfg.model.lsm_weight, # NOTE, use 'cfg' or '_cfg'? use _cfg!
            smoothing = self._cfg.get('lsm_weight', 0.1), 
            normalize_length = self._cfg.get('length_normalized_loss', False), # NOTE
        )

        #import ipdb; ipdb.set_trace()
        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecCTCAttnModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.char_dict = dict([(i, self.decoder.vocabulary[i]) for i in range(len(self.decoder.vocabulary))])

        # Setup metric objects
        #import ipdb; ipdb.set_trace()
        self._wer = WER(
            vocabulary=self.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
            blank_id=BLANK_ID
        )

    def extend_vocab(self, vocabulary):
        vocabulary.append(SOSEOS_STR) # <sos/eos> to the final position
        vocabulary.insert(BLANK_ID, BLANK_STR) # <blank>:0
        vocabulary.insert(UNK_ID, UNK_STR) # <unk>:1

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        #import ipdb; ipdb.set_trace() # TODO not checked yet
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                }

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            lg = logits[idx][: logits_len[idx]]
                            hypotheses.append(lg.cpu().numpy())
                    else:
                        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                        hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses

    def change_vocabulary(self, new_vocabulary: List[str]):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        self.extend_vocab(new_vocabulary)
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary) # NOTE already include <blank>, <unk>, <sos/eos>

            del self.decoder
            self.decoder = EncDecCTCAttnModel.from_config_dict(new_decoder_config)

            #del self.loss
            #self.loss = CTCLoss(
            #    num_classes=self.decoder.num_classes_with_blank - 1,
            #    zero_infinity=True,
            #    reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            #)
            del self.ctc
            self.ctc = CTC(self.decoder.num_classes_with_blank, self.encoder._feat_out)

            del self.criterion_att
            self.criterion_att = LabelSmoothingLoss(
                size = self.decoder.num_classes_with_blank, # NOTE
                padding_idx = self.ignore_id, 
                #smoothing=self.cfg.model.lsm_weight, # NOTE, use 'cfg' or '_cfg'? -> '_cfg'
                smoothing = self._cfg.get('lsm_weight', 0.1), 
                normalize_length = self._cfg.get('length_normalized_loss', False), # NOTE
            )

            self._wer = WER(
                vocabulary=self.decoder.vocabulary,
                batch_dim_index=0,
                use_cer=self._cfg.get('use_cer', False),
                ctc_decode=True,
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
            )

            # Update config
            OmegaConf.set_struct(self._cfg.decoder, False)
            self._cfg.decoder = new_decoder_config
            OmegaConf.set_struct(self._cfg.decoder, True)

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=config,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        if type(dataset) is ChainDataset:
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        #import ipdb; ipdb.set_trace()
        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        #import ipdb; ipdb.set_trace()
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        #import ipdb; ipdb.set_trace()
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
            "text": NeuralType(('B', 'T'), LabelsType(), optional=False),
            "text_lengths": NeuralType(tuple('B'), LengthsType(), optional=False)
        }
    # TODO do not understand...
    #@property
    #def output_types(self) -> Optional[Dict[str, ElementType]]:
    #    return {
    #        #"outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
    #        #"encoded_lengths": NeuralType(tuple('B'), LengthsType()),
    #        #"greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
    #        #return loss, loss_att, loss_ctc, acc_att
    #        "loss": LossType(),
    #        "loss_att": LossType(),
    #        "loss_ctc": LossType(),
    #        "acc_att": LossType(),
    #    }

    @typecheck()
    def forward(
        self, 
        input_signal: torch.Tensor = None, 
        input_signal_length: torch.Tensor = None, 
        processed_signal: torch.Tensor = None, 
        processed_signal_length: torch.Tensor = None,
        text: torch.Tensor = None, 
        text_lengths: torch.Tensor = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
            text: reference text, (Batch, Length) of token ids.
            text_lengths: length list of reference text, (Batch,)

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        #import ipdb; ipdb.set_trace()
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        # 1. encoder
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        #import ipdb; ipdb.set_trace()
        encoded = encoded.transpose(1, 2) # from (batch, hidden dim, length) to (batch, length, hidden dim)
        #encoded_out_mask = ~make_pad_mask(encoded_len).unsqueeze(1) # (batch, 1, hidden dim)
        encoded_out_mask = ~make_pad_mask(encoded_len, max_len=encoded.size(1)).unsqueeze(1) 
        # (batch, 1, length of source seq)

        #import ipdb; ipdb.set_trace() 
        # 2a. attention-decoder branch
        acc_att = 0.0 # accuracy from attention decoder
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoded, encoded_out_mask, text, text_lengths)
        else:
            loss_att = None
       
        #import ipdb; ipdb.set_trace()
        # 2b. ctc branch
        if self.ctc_weight != 0.0:
            #import ipdb; ipdb.set_trace()
            loss_ctc = self.ctc(encoded, encoded_len, text, text_lengths)      
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if torch.isnan(loss):
            #import ipdb; ipdb.set_trace()
            print('bad, nan loss=', loss)
        #import ipdb; ipdb.set_trace()
        return loss, loss_att, loss_ctc, acc_att

        #log_probs = self.decoder(encoder_output=encoded)
        #greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        #import ipdb; ipdb.set_trace()

        #return log_probs, encoded_len, greedy_predictions

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, self.ignore_id)
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        #import ipdb; ipdb.set_trace()
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        #import ipdb; ipdb.set_trace()
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.decoder.num_classes_with_blank), #self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        #import ipdb; ipdb.set_trace()
        return loss_att, acc_att


    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        #import ipdb; ipdb.set_trace()
        #signal, signal_len, transcript, transcript_len = batch
        signal, signal_len, transcript, transcript_len, sample_ids, fn_list = self.parse_batch(batch)
        debug = False
        if debug:
            # check the gradient of parameters in current model
            max_p, min_p, max_g, min_g = 0.0, 0.0, 0.0, 0.0
            mean_list = []
            for name, params in self.named_parameters():
                max_p = params.max().item() if params.max().item() > max_p else max_p
                min_p = params.min().item() if params.min().item() < max_p else min_p
                if params.requires_grad:
                    amean_value = params.mean().item()
                    mean_list.append(amean_value)
                    if torch.isnan(params.mean()):
                        #import ipdb; ipdb.set_trace()
                        print('nan mean, why??? (batch fn is for next)', name, params, fn_list)

                    grad_min = params.grad.min().item() if params.grad is not None else 'NA'
                    grad_max = params.grad.max().item() if params.grad is not None else 'NA'
                    #print(name, params.min(), params.max(), grad_min, grad_max)
                    if grad_min != 'NA':
                        min_g = grad_min if grad_min < min_g else min_g
                        max_g = grad_max if grad_max > max_g else max_g

            mean_value = np.mean(mean_list)
            print(max_p, min_p, np.mean(mean_list), max_g, min_g)
            if mean_value == float('nan'):
                print('total mean is nan, (batch fn is for next)', mean_list, fn_list)

        if debug:
            print('fn_list:', fn_list)
        # use -1 to pad transcript_len! (was 0 for padding)
        transcript = pad_sequence([y[:i] for y, i in zip(transcript, transcript_len)], True, self.ignore_id)

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            loss, loss_attn, loss_ctc, acc_att = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
                text=transcript, text_lengths=transcript_len
            )
        else:
            #log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)
            loss, loss_attn, loss_ctc, acc_att = self.forward(
                input_signal=signal, input_signal_length=signal_len,
                text=transcript, text_lengths=transcript_len
            )

        #loss_value = self.loss(
        #    log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        #)

        tensorboard_logs = {'train_loss': loss, 
                'train_loss_attn': loss_attn,
                'train_loss_ctc': loss_ctc,
                'accuracy_attention': acc_att,
                'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        #if (batch_nb + 1) % log_every_n_steps == 0:
        #    self._wer.update(
        #        predictions=predictions,
        #        targets=transcript,
        #        target_lengths=transcript_len,
        #        predictions_lengths=encoded_len,
        #    )
        #    wer, _, _ = self._wer.compute()
        #    self._wer.reset()
        #    tensorboard_logs.update({'training_batch_wer': wer})
        if torch.isnan(loss):
            #import ipdb; ipdb.set_trace()
            print('bad nan loss=', loss)
            if debug:
                print('bad nan, current fn_list:', fn_list)
                os._exit(0)

        if debug:
            print(max_p, min_p, np.mean(mean_list), max_g, min_g, loss, tensorboard_logs)

        return {'loss': loss, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        #import ipdb; ipdb.set_trace() # TODO not checked yet
        #signal, signal_len, transcript, transcript_len, sample_id = batch
        signal, signal_len, transcript, transcript_len, sample_ids, fn_list = self.parse_batch(batch)
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        transcribed_texts = self._wer.ctc_decoder_predictions_tensor(
            predictions=predictions, predictions_len=encoded_len, return_hypotheses=False,
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, transcribed_texts))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        #import ipdb; ipdb.set_trace()
        #signal, signal_len, transcript, transcript_len = batch
        signal, signal_len, transcript, transcript_len, sample_ids, fn_list = self.parse_batch(batch)
        # use -1 to pad transcript_len! (was 0 for padding)
        transcript = pad_sequence([y[:i] for y, i in zip(transcript, transcript_len)], True, self.ignore_id)

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            #log_probs, encoded_len, predictions = self.forward(
            #    processed_signal=signal, processed_signal_length=signal_len
            #)
            loss, loss_attn, loss_ctc, acc_att = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
                text=transcript, text_lengths=transcript_len
            )
        else:
            #log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)
            loss, loss_attn, loss_ctc, acc_att = self.forward(
                input_signal=signal, input_signal_length=signal_len,
                text=transcript, text_lengths=transcript_len
            )

        #loss_value = self.loss(
        #    log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        #)
        #self._wer.update(
        #    predictions=predictions, targets=transcript, 
        #    target_lengths=transcript_len, predictions_lengths=encoded_len
        #)
        #wer, wer_num, wer_denom = self._wer.compute()
        #self._wer.reset()
        return {
            'val_loss': loss, #_value, # linear combination of "loss_attn" and "loss_ctc"
            'val_loss_attn': loss_attn, # bi-decoder attention loss
            'val_loss_ctc': loss_ctc, # ctc loss
            'val_wer_num': torch.FloatTensor([0.0]), #wer_num,
            'val_wer_denom': torch.FloatTensor([0.0]), #wer_denom,
            'val_wer': torch.FloatTensor([0.0]), #wer,
            'val_acc': acc_att,
        }

    def parse_batch(self, batch):
        sample_ids, fn_list = None, None
        if len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch
        elif len(batch) == 5:
            id_or_fn = batch[-1]
            if id_or_fn is None or len(id_or_fn) == 0:
                signal, signal_len, transcript, transcript_len, _ = batch
            else:
                if isinstance(batch[-1][0], str):
                    signal, signal_len, transcript, transcript_len, fn_list = batch
                else:
                    signal, signal_len, transcript, transcript_len, sample_ids = batch
        elif len(batch) == 6:
            signal, signal_len, transcript, transcript_len, sample_ids, fn_list = batch
        else:
            raise ValueError('expect 4, 5, or 6 elemments in a batch, got {}'.format(len(batch)))

        return signal, signal_len, transcript, transcript_len, sample_ids, fn_list

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        #import ipdb; ipdb.set_trace()
        # NOTE: the decoding algorithms can be called here! and do real decoding~~
        # such as (1) ctc greedy search (2) ctc prefix beam search 
        # (3) (auto-regressive) attention decoder (4) attention rescoring
        signal, signal_len, transcript, transcript_len, sample_ids, fn_list = self.parse_batch(batch)

        # use -1 to pad transcript_len! (was 0 for padding)
        transcript = pad_sequence([y[:i] for y, i in zip(transcript, transcript_len)], True, self.ignore_id)

        beam_size = self._cfg.get('inf_beam_size', 10) # NOTE: read from config
        hyps_nbest, scores_nbest = None, None # for n-best output
        
        out_beam = self._cfg.get('inf_out_beam', True) 
        # NOTE read from config, is output n-best beam output or not

        simulate_streaming = self._cfg.get('inf_simulate_streaming', False)

        inf_alg = self._cfg.get('inf_alg', 'attention_rescoring') # default inference algorithm
        if inf_alg == 'attention_rescoring':
            assert (signal.size(0) == 1)

            hyp, _, hyps_nbest, scores_nbest = self.attention_rescoring(
                signal,
                signal_len,
                beam_size,
                decoding_chunk_size = -1,
                num_decoding_left_chunks = -1,
                ctc_weight = self.ctc_weight,
                simulate_streaming = simulate_streaming,  
                reverse_weight = self.reverse_weight
            )
            
            hyps = [hyp]
            if out_beam:
                hyps_nbest = [hyps_nbest]
                scores_nbest = [scores_nbest]
        elif inf_alg == 'ctc_prefix_beam_search':
            assert (signal.size(0) == 1)
            hyp, hyps_nbest = self.ctc_prefix_beam_search(
                signal,
                signal_len,
                beam_size,
                decoding_chunk_size = -1,
                num_decoding_left_chunks = -1,
                simulate_streaming = simulate_streaming
            )
            hyps = [hyp[0]] # TODO original was hyps=[hyp], yet hyp=((1396, ..., 1670), logp)
            if out_beam:
                scores_nbest = [[y[1] for y in hyps_nbest]]
                hyps_nbest = [hyps_nbest]
        elif inf_alg == 'ctc_greedy_search':
            hyps, _ = self.ctc_greedy_search(
                signal,
                signal_len,
                decoding_chunk_size = -1,
                num_decoding_left_chunks = -1,
                simulate_streaming = simulate_streaming
            )
        elif inf_alg == 'attention':
            # auto-regressive decoding
            hyps, _, hyps_nbest, scores_nbest = self.recognize(
                signal,
                signal_len,
                beam_size = beam_size,
                decoding_chunk_size = -1,
                num_decoding_left_chunks = -1,
                simulate_streaming = simulate_streaming
            )
            hyps = [hyp.tolist() for hyp in hyps]

        for i in range(signal.size(0)):
            # TODO need batch with "keys" (a list of keys to label outputs!)
            content = ''
            for w_id in hyps[i]:
                if w_id == self.eos:
                    break
                content += self.char_dict[w_id]
            ref = ''.join([self.char_dict[w_id.item()] for w_id in transcript[i]])
            fn = fn_list[i] if fn_list and len(fn_list) > i else 'fn=NA'
            print('tstout={}\tref={}\t{}'.format(content, ref, fn))

            # TODO think about n-best output as well
            # write n-best result to fout!
            if out_beam and hyps_nbest is not None and scores_nbest is not None:
                for j, hyp in enumerate(hyps_nbest[i]):
                    # (prefix, prob from ctc)
                    hyp_prefix = hyp.cpu().numpy() if isinstance(hyp, 
                            torch.Tensor) else hyp[0]
                    content = ''
                    for w in hyp_prefix:
                        if w == self.eos:
                            break
                        content += self.char_dict[w]
                    #out_row = 'beam: {} {} {} {}'.format(key, j, scores_nbest[i][j], content)
                    out_row = 'beam: {} {} {} {}'.format(
                            j, scores_nbest[i][j], content, fn)
                    #logging.info(out_row)
                    #fout.write(out_row + '\n')
                    print(out_row)


        #import ipdb; ipdb.set_trace()
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_loss': logs['val_loss'],
            'test_loss_attn': logs['val_loss_attn'],
            'test_loss_ctc': logs['val_loss_ctc'],
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            'test_wer': logs['val_wer'],
            'test_acc': logs['val_acc']
        }
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.decoder.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=speech, length=speech_lengths,
        )

        # Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_out_len = self.encoder.forward_chunk_by_chunk( # TODO not implemented!
                processed_signal,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_out_len = self.encoder( # TODO
                audio_signal = processed_signal,
                length = processed_signal_length,
                #decoding_chunk_size=decoding_chunk_size,
                #num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        #import ipdb; ipdb.set_trace()
        encoder_out = encoder_out.transpose(1, 2) # from (B, hidden dim, len) to (B, len, hidden dim)
        encoder_out_mask = ~make_pad_mask(encoder_out_len).unsqueeze(1) # (B, 1, hidden dim)

        return encoder_out, encoder_out_mask
        

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, 
            speech_lengths, 
            decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)

        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]

        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank -> TODO s=0 (' ') is not appended to the output sequence???
                        # TODO should use blank_id, not 0, here! blank (epsilon) is different with space (' ')
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb) 
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]

        #import ipdb; ipdb.set_trace()
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out # TODO hyps does not include token_id=0...

    
    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        #import ipdb; ipdb.set_trace()
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search( # TODO
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1) # (1, L, H) -> (beam, L, H)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device) # (beam, 1, L), all 'True' since it is repeating!
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size+1=num_classes)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        scores = list()
        for i, hyp in enumerate(hyps):
            # hyp = (prefix string, prob)
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            scores.append(score)
            if score > best_score:
                best_score = score
                best_index = i
        # TODO return all the n-best for system ensemble
        #import ipdb; ipdb.set_trace()
        return hyps[best_index][0], best_score, hyps, scores

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]: # TODO output type re-check!
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        # return n-best for system ensemble
        return hyps[0], hyps

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, 
            speech_lengths, 
            decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)

        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        # TODO should explicitly tell blank_id here in "remove_duplicates_and_blank"
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores


    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply autoregressive (step-by-step) beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)

        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            #import ipdb; ipdb.set_trace()
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            #best_hyps_index = best_k_index // beam_size
            best_hyps_index = torch.div(best_k_index, beam_size, rounding_mode='floor')

            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        #import ipdb; ipdb.set_trace()
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        hyps = hyps.view(batch_size, beam_size, -1)
        hyps = hyps[:, :, 1:]
        #import ipdb; ipdb.set_trace()
        return best_hyps, best_scores, hyps, scores


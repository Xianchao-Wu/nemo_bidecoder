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

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor,
    CropOrPadSpectrogramAugmentation,
    SpectrogramAugmentation,
)
from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.lstm_decoder import LSTMDecoder
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint

# TODO @blisc: Perhaps refactor instead of import guarding
try:
    from nemo.collections.asr.modules.conv_asr import (
        ConvASRDecoder,
        ConvASRDecoderClassification,
        ConvASRDecoderReconstruction,
        ConvASREncoder,
        ECAPAEncoder,
        ParallelConvASREncoder,
        SpeakerDecoder,
    )
except ModuleNotFoundError:
    from nemo.utils.exceptions import CheckInstall

    # fmt: off
    class ConvASRDecoder(CheckInstall): pass
    class ConvASRDecoderClassification(CheckInstall): pass
    class ConvASREncoder(CheckInstall): pass
    class ECAPAEncoder(CheckInstall): pass
    class ParallelConvASREncoder(CheckInstall): pass
    class SpeakerDecoder(CheckInstall): pass
    class ConvASRDecoderReconstruction(CheckInstall): pass
    # fmt: on

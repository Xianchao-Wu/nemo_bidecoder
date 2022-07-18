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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter

import os
import random

import librosa
import numpy as np
import soundfile as sf

from nemo.utils import logging

# TODO @blisc: Perhaps refactor instead of import guarding
HAVE_KALDI_PYDUB = True
try:
    from kaldiio.matio import read_kaldi
    from kaldiio.utils import open_like_kaldi
    from pydub import AudioSegment as Audio
    from pydub.exceptions import CouldntDecodeError
except ModuleNotFoundError:
    HAVE_KALDI_PYDUB = False


available_formats = sf.available_formats()
sf_supported_formats = ["." + i.lower() for i in available_formats.keys()]


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=None, trim=False, trim_db=60, orig_sr=None):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

        self._orig_sr = orig_sr if orig_sr is not None else sample_rate

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB" % (
            type(self),
            self.num_samples,
            self.sample_rate,
            self.duration,
            self.rms_db,
        )

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(
        cls, audio_file, target_sr=None, int_values=False, offset=0, duration=0, trim=False, orig_sr=None,
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param audio_file: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        samples = None
        if not isinstance(audio_file, str) or os.path.splitext(audio_file)[-1] in sf_supported_formats:
            try:
                with sf.SoundFile(audio_file, 'r') as f:
                    dtype = 'int32' if int_values else 'float32'
                    sample_rate = f.samplerate
                    if offset > 0:
                        f.seek(int(offset * sample_rate))
                    if duration > 0:
                        samples = f.read(int(duration * sample_rate), dtype=dtype)
                    else:
                        samples = f.read(dtype=dtype)
                samples = samples.transpose()
            except RuntimeError as e:
                logging.error(
                    f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`. "
                    f"NeMo will fallback to loading via pydub."
                )
        elif HAVE_KALDI_PYDUB and isinstance(audio_file, str) and audio_file.strip()[-1] == "|":
            f = open_like_kaldi(audio_file, "rb")
            sample_rate, samples = read_kaldi(f)
            if offset > 0:
                samples = samples[int(offset * sample_rate) :]
            if duration > 0:
                samples = samples[: int(duration * sample_rate)]
            if not int_values:
                abs_max_value = np.abs(samples).max()
                samples = np.array(samples, dtype=np.float) / abs_max_value

        if HAVE_KALDI_PYDUB and samples is None:
            try:
                samples = Audio.from_file(audio_file)
                sample_rate = samples.frame_rate
                if offset > 0:
                    # pydub does things in milliseconds
                    seconds = offset * 1000
                    samples = samples[int(seconds) :]
                if duration > 0:
                    seconds = duration * 1000
                    samples = samples[: int(seconds)]
                samples = np.array(samples.get_array_of_samples())
            except CouldntDecodeError as err:
                logging.error(f"Loading {audio_file} via pydub raised CouldntDecodeError: `{err}`.")

        if samples is None:
            libs = "soundfile, kaldiio, and pydub" if HAVE_KALDI_PYDUB else "soundfile"
            raise Exception(f"Your audio file {audio_file} could not be decoded. We tried using {libs}.")

        return cls(samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr)

    @classmethod
    def segment_from_file(cls, audio_file, target_sr=None, n_segments=0, trim=False, orig_sr=None):
        """Grabs n_segments number of samples from audio_file randomly from the
        file as opposed to at a specified offset.

        Note that audio_file can be either the file path, or a file-like object.
        """
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                sample_rate = f.samplerate
                if n_segments > 0 and len(f) > n_segments:
                    max_audio_start = len(f) - n_segments
                    audio_start = random.randint(0, max_audio_start)
                    f.seek(audio_start)
                    samples = f.read(n_segments, dtype='float32')
                else:
                    samples = f.read(dtype='float32')
            samples = samples.transpose()
        except RuntimeError as e:
            logging.error(f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`.")

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    @property
    def orig_sr(self):
        return self._orig_sr

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number
        of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false,
        `pad_size`
        zeros will be added only to the end.
        """
        self._samples = np.pad(self._samples, (pad_size if symmetric else 0, pad_size), mode='constant',)

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set,
        e.g. out
                           of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time
        if start_time < 0.0:
            start_time = self.duration + start_time
        if end_time < 0.0:
            end_time = self.duration + end_time
        if start_time < 0.0:
            raise ValueError("The slice start position (%f s) is out of bounds." % start_time)
        if end_time < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." % end_time)
        if start_time > end_time:
            raise ValueError(
                "The slice start position (%f s) is later than the end position (%f s)." % (start_time, end_time)
            )
        if end_time > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds (> %f s)" % (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

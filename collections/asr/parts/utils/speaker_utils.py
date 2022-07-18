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
import json
import math
import os
from copy import deepcopy

import numpy as np
import omegaconf
import soundfile as sf
import torch
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

from nemo.collections.asr.parts.utils.nmesc_clustering import COSclustering
from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental


"""
This file contains all the utility functions required for speaker embeddings part in diarization scripts
"""


def get_uniqname_from_filepath(filepath):
    "return base name from provided filepath"
    if type(filepath) is str:
        basename = os.path.basename(filepath).rsplit('.', 1)[0]
        return basename
    else:
        raise TypeError("input must be filepath string")


def audio_rttm_map(manifest):
    """
    This function creates AUDIO_RTTM_MAP which is used by all diarization components to extract embeddings,
    cluster and unify time stamps
    input: manifest file that contains keys audio_filepath, rttm_filepath if exists, text, num_speakers if known and uem_filepath if exists

    returns:
    AUDIO_RTTM_MAP (dict) : A dictionary with keys of uniq id, which is being used to map audio files and corresponding rttm files
    """

    AUDIO_RTTM_MAP = {}
    with open(manifest, 'r') as inp_file:
        lines = inp_file.readlines()
        logging.info("Number of files to diarize: {}".format(len(lines)))
        for line in lines:
            line = line.strip()
            dic = json.loads(line)

            meta = {
                'audio_filepath': dic['audio_filepath'],
                'rttm_filepath': dic.get('rttm_filepath', None),
                'duration': dic.get('duration', None),
                'text': dic.get('text', None),
                'num_speakers': dic.get('num_speakers', None),
                'uem_filepath': dic.get('uem_filepath', None),
                'ctm_filepath': dic.get('ctm_filepath', None),
            }

            uniqname = get_uniqname_from_filepath(filepath=meta['audio_filepath'])

            if uniqname not in AUDIO_RTTM_MAP:
                AUDIO_RTTM_MAP[uniqname] = meta
            else:
                raise KeyError(
                    "file {} is already part AUDIO_RTTM_Map, it might be duplicated".format(meta['audio_filepath'])
                )

    return AUDIO_RTTM_MAP


@experimental
def parse_scale_configs(window_lengths_in_sec, shift_lengths_in_sec, multiscale_weights):
    """
    Check whether multiscale parameters are provided correctly. window_lengths_in_sec, shift_lengfhs_in_sec and
    multiscale_weights should be all provided in omegaconf.listconfig.ListConfig type. In addition, the scales
    should be provided in descending order, from the longest scale to the base scale (the shortest).

    Example:
        Single-scale setting:
            parameters.window_length_in_sec=1.5
            parameters.shift_length_in_sec=0.75
            parameters.multiscale_weights=null

        Multiscale setting (base scale - window_length 0.5 s and shift_length 0.25):
            parameters.window_length_in_sec=[1.5,1.0,0.5]
            parameters.shift_length_in_sec=[0.75,0.5,0.25]
            parameters.multiscale_weights=[0.33,0.33,0.33]
    """
    checkFloatConfig = [type(var) == float for var in (window_lengths_in_sec, shift_lengths_in_sec)]
    checkListConfig = [
        type(var) == type(omegaconf.listconfig.ListConfig([]))
        for var in (window_lengths_in_sec, shift_lengths_in_sec, multiscale_weights)
    ]
    if all(checkListConfig) or all(checkFloatConfig):

        # If bare floating numbers are provided, convert them to list format.
        if all(checkFloatConfig):
            window_lengths, shift_lengths, multiscale_weights = (
                [window_lengths_in_sec],
                [shift_lengths_in_sec],
                [1.0],
            )
        else:
            window_lengths, shift_lengths, multiscale_weights = (
                window_lengths_in_sec,
                shift_lengths_in_sec,
                multiscale_weights,
            )

        length_check = (
            len(set([len(window_lengths), len(shift_lengths), len(multiscale_weights)])) == 1
            and len(multiscale_weights) > 0
        )
        scale_order_check = (
            window_lengths == sorted(window_lengths)[::-1] and shift_lengths == sorted(shift_lengths)[::-1]
        )

        # Check whether window lengths are longer than shift lengths
        if len(window_lengths) > 1:
            shift_length_check = all([w > s for w, s in zip(window_lengths, shift_lengths)]) == True
        else:
            shift_length_check = window_lengths[0] > shift_lengths[0]

        multiscale_args_dict = {}
        if all([length_check, scale_order_check, shift_length_check]) == True:
            if len(window_lengths) > 1:
                multiscale_args_dict['scale_dict'] = {
                    k: (w, s) for k, (w, s) in enumerate(zip(window_lengths, shift_lengths))
                }
            else:
                multiscale_args_dict['scale_dict'] = {0: (window_lengths[0], shift_lengths[0])}
            multiscale_args_dict['multiscale_weights'] = multiscale_weights
            return multiscale_args_dict
        else:
            raise ValueError('Multiscale parameters are not properly setup.')

    elif any(checkListConfig):
        raise ValueError(
            'You must provide list config for all three parameters: window, shift and multiscale weights.'
        )
    else:
        return None


@experimental
def get_embs_and_timestamps(multiscale_embeddings_and_timestamps, multiscale_args_dict):
    """
    The embeddings and timestamps in multiscale_embeddings_and_timestamps dictionary are
    indexed by scale index. This function rearranges the extracted speaker embedding and
    timestamps by unique ID to make the further processing more convenient.

    Args:
        multiscale_embeddings_and_timestamps (dict):
            Dictionary of embeddings and timestamps for each scale.
        multiscale_args_dict (dict):
            Dictionary of scale information: window, shift and multiscale weights.

    Returns:
        embs_and_timestamps (dict)
            A dictionary containing embeddings and timestamps of each scale, indexed by unique ID.
    """
    embs_and_timestamps = {
        uniq_id: {'multiscale_weights': [], 'scale_dict': {}}
        for uniq_id in multiscale_embeddings_and_timestamps[0][0].keys()
    }
    for scale_idx in sorted(multiscale_args_dict['scale_dict'].keys()):
        embeddings, time_stamps = multiscale_embeddings_and_timestamps[scale_idx]
        for uniq_id in embeddings.keys():
            embs_and_timestamps[uniq_id]['multiscale_weights'] = multiscale_args_dict['multiscale_weights']
            assert len(embeddings[uniq_id]) == len(time_stamps[uniq_id])
            embs_and_timestamps[uniq_id]['scale_dict'][scale_idx] = {
                'embeddings': embeddings[uniq_id],
                'time_stamps': time_stamps[uniq_id],
            }

    return embs_and_timestamps


def get_contiguous_stamps(stamps):
    """
    Return contiguous time stamps
    """
    lines = deepcopy(stamps)
    contiguous_stamps = []
    for i in range(len(lines) - 1):
        start, end, speaker = lines[i].split()
        next_start, next_end, next_speaker = lines[i + 1].split()
        if float(end) > float(next_start):
            avg = str((float(next_start) + float(end)) / 2.0)
            lines[i + 1] = ' '.join([avg, next_end, next_speaker])
            contiguous_stamps.append(start + " " + avg + " " + speaker)
        else:
            contiguous_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = lines[-1].split()
    contiguous_stamps.append(start + " " + end + " " + speaker)
    return contiguous_stamps


def merge_stamps(lines):
    """
    merge time stamps of same speaker
    """
    stamps = deepcopy(lines)
    overlap_stamps = []
    for i in range(len(stamps) - 1):
        start, end, speaker = stamps[i].split()
        next_start, next_end, next_speaker = stamps[i + 1].split()
        if float(end) == float(next_start) and speaker == next_speaker:
            stamps[i + 1] = ' '.join([start, next_end, next_speaker])
        else:
            overlap_stamps.append(start + " " + end + " " + speaker)

    start, end, speaker = stamps[-1].split()
    overlap_stamps.append(start + " " + end + " " + speaker)

    return overlap_stamps


def labels_to_pyannote_object(labels, uniq_name=''):
    """
    converts labels to pyannote object to calculate DER and for visualization
    """
    annotation = Annotation(uri=uniq_name)
    for label in labels:
        start, end, speaker = label.strip().split()
        start, end = float(start), float(end)
        annotation[Segment(start, end)] = speaker

    return annotation


def uem_timeline_from_file(uem_file, uniq_name=''):
    """
    outputs pyannote timeline segments for uem file

     <UEM> file format
     UNIQ_SPEAKER_ID CHANNEL START_TIME END_TIME
    """
    timeline = Timeline(uri=uniq_name)
    with open(uem_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            speaker_id, channel, start_time, end_time = line.split()
            timeline.add(Segment(float(start_time), float(end_time)))

    return timeline


def labels_to_rttmfile(labels, uniq_id, out_rttm_dir):
    """
    write rttm file with uniq_id name in out_rttm_dir with time_stamps in labels
    """
    filename = os.path.join(out_rttm_dir, uniq_id + '.rttm')
    with open(filename, 'w') as f:
        for line in labels:
            line = line.strip()
            start, end, speaker = line.split()
            duration = float(end) - float(start)
            start = float(start)
            log = 'SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(uniq_id, start, duration, speaker)
            f.write(log)

    return filename


def rttm_to_labels(rttm_filename):
    """
    prepares time stamps label list from rttm file
    """
    labels = []
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            rttm = line.strip().split()
            start, end, speaker = float(rttm[3]), float(rttm[4]) + float(rttm[3]), rttm[7]
            labels.append('{} {} {}'.format(start, end, speaker))
    return labels


# embeddings, time_stamps, AUDIO_RTTM_MAP, out_rttm_dir, clustering_params, multi_scale_data=None


def perform_clustering(embs_and_timestamps, AUDIO_RTTM_MAP, out_rttm_dir, clustering_params):
    """
    performs spectral clustering on embeddings with time stamps generated from VAD output

    Args:
        embs_and_timestamps (dict): This dictionary contains the following items indexed by unique IDs.
            'embeddings' : Embeddings with key as unique_id
            'time_stamps' : Time stamps list for each audio recording
        AUDIO_RTTM_MAP (dict): AUDIO_RTTM_MAP for mapping unique id with audio file path and rttm path
        out_rttm_dir (str): Path to write predicted rttms
        clustering_params (dict): clustering parameters provided through config that contains max_num_speakers (int),
        oracle_num_speakers (bool), max_rp_threshold(float), sparse_search_volume(int) and enhance_count_threshold (int)

    Returns:
        all_reference (list[uniq_name,Annotation]): reference annotations for score calculation
        all_hypothesis (list[uniq_name,Annotation]): hypothesis annotations for score calculation

    """
    all_hypothesis = []
    all_reference = []
    no_references = False
    max_num_speakers = clustering_params['max_num_speakers']

    cuda = True
    if not torch.cuda.is_available():
        logging.warning("cuda=False, using CPU for Eigen decompostion. This might slow down the clustering process.")
        cuda = False

    for uniq_id, value in tqdm(AUDIO_RTTM_MAP.items()):
        if clustering_params.oracle_num_speakers:
            num_speakers = value.get('num_speakers', None)
            if num_speakers is None:
                raise ValueError("Provided option as oracle num of speakers but num_speakers in manifest is null")
        else:
            num_speakers = None

        cluster_labels = COSclustering(
            uniq_embs_and_timestamps=embs_and_timestamps[uniq_id],
            oracle_num_speakers=num_speakers,
            max_num_speaker=max_num_speakers,
            enhanced_count_thres=clustering_params.enhanced_count_thres,
            max_rp_threshold=clustering_params.max_rp_threshold,
            sparse_search_volume=clustering_params.sparse_search_volume,
            cuda=cuda,
        )

        base_scale_idx = max(embs_and_timestamps[uniq_id]['scale_dict'].keys())
        lines = embs_and_timestamps[uniq_id]['scale_dict'][base_scale_idx]['time_stamps']
        assert len(cluster_labels) == len(lines)
        for idx, label in enumerate(cluster_labels):
            tag = 'speaker_' + str(label)
            lines[idx] += tag

        a = get_contiguous_stamps(lines)
        labels = merge_stamps(a)
        if out_rttm_dir:
            labels_to_rttmfile(labels, uniq_id, out_rttm_dir)
        hypothesis = labels_to_pyannote_object(labels, uniq_name=uniq_id)
        all_hypothesis.append([uniq_id, hypothesis])

        rttm_file = value.get('rttm_filepath', None)
        if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            all_reference.append([uniq_id, reference])
        else:
            no_references = True
            all_reference = []

    return all_reference, all_hypothesis


def score_labels(AUDIO_RTTM_MAP, all_reference, all_hypothesis, collar=0.25, ignore_overlap=True):
    """
    calculates DER, CER, FA and MISS

    Args:
    AUDIO_RTTM_MAP : Dictionary containing information provided from manifestpath
    all_reference (list[uniq_name,Annotation]): reference annotations for score calculation
    all_hypothesis (list[uniq_name,Annotation]): hypothesis annotations for score calculation

    Returns:
    metric (pyannote.DiarizationErrorRate): Pyannote Diarization Error Rate metric object. This object contains detailed scores of each audiofile.
    mapping (dict): Mapping dict containing the mapping speaker label for each audio input

    < Caveat >
    Unlike md-eval.pl, "no score" collar in pyannote.metrics is the maximum length of
    "no score" collar from left to right. Therefore, if 0.25s is applied for "no score"
    collar in md-eval.pl, 0.5s should be applied for pyannote.metrics.

    """
    metric = None
    if len(all_reference) == len(all_hypothesis):
        metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=ignore_overlap)

        mapping_dict = {}
        for (reference, hypothesis) in zip(all_reference, all_hypothesis):
            ref_key, ref_labels = reference
            _, hyp_labels = hypothesis
            uem = AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None)
            if uem is not None:
                uem = uem_timeline_from_file(uem_file=uem, uniq_name=ref_key)
            metric(ref_labels, hyp_labels, uem=uem, detailed=True)
            mapping_dict[ref_key] = metric.optimal_mapping(ref_labels, hyp_labels)

        DER = abs(metric)
        CER = metric['confusion'] / metric['total']
        FA = metric['false alarm'] / metric['total']
        MISS = metric['missed detection'] / metric['total']

        logging.info(
            "Cumulative Results for collar {} sec and ignore_overlap {}: \n FA: {:.4f}\t MISS {:.4f}\t \
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                collar, ignore_overlap, FA, MISS, DER, CER
            )
        )

        return metric, mapping_dict
    else:
        logging.warning(
            "check if each ground truth RTTMs were present in provided manifest file. Skipping calculation of Diariazation Error Rate"
        )

        return None


def write_rttm2manifest(AUDIO_RTTM_MAP, manifest_file):
    """
    writes manifest file based on rttm files (or vad table out files). This manifest file would be used by 
    speaker diarizer to compute embeddings and cluster them. This function also takes care of overlap time stamps

    Args:
    AUDIO_RTTM_MAP: dict containing keys to uniqnames, that contains audio filepath and rttm_filepath as its contents,
    these are used to extract oracle vad timestamps.
    manifest (str): path to write manifest file

    Returns:
    manifest (str): path to write manifest file
    """

    with open(manifest_file, 'w') as outfile:
        for key in AUDIO_RTTM_MAP:
            rttm_filename = AUDIO_RTTM_MAP[key]['rttm_filepath']
            if rttm_filename and os.path.exists(rttm_filename):
                f = open(rttm_filename, 'r')
            else:
                raise FileNotFoundError(
                    "Requested to construct manifest from rttm with oracle VAD option or from NeMo VAD but received filename as {}".format(
                        rttm_filename
                    )
                )

            audio_path = AUDIO_RTTM_MAP[key]['audio_filepath']
            if AUDIO_RTTM_MAP[key].get('duration', None):
                max_duration = AUDIO_RTTM_MAP[key]['duration']
            else:
                sound = sf.SoundFile(audio_path)
                max_duration = sound.frames / sound.samplerate

            lines = f.readlines()
            time_tup = (-1, -1)
            for line in lines:
                vad_out = line.strip().split()
                if len(vad_out) > 3:
                    start, dur, _ = float(vad_out[3]), float(vad_out[4]), vad_out[7]
                else:
                    start, dur, _ = float(vad_out[0]), float(vad_out[1]), vad_out[2]
                start, dur = float("{:.3f}".format(start)), float("{:.3f}".format(dur))

                if time_tup[0] >= 0 and start > time_tup[1]:
                    dur2 = float("{:.3f}".format(time_tup[1] - time_tup[0]))
                    if time_tup[0] < max_duration and dur2 > 0:
                        meta = {"audio_filepath": audio_path, "offset": time_tup[0], "duration": dur2, "label": 'UNK'}
                        json.dump(meta, outfile)
                        outfile.write("\n")
                    else:
                        logging.warning(
                            "RTTM label has been truncated since start is greater than duration of audio file"
                        )
                    time_tup = (start, start + dur)
                else:
                    if time_tup[0] == -1:
                        end_time = start + dur
                        if end_time > max_duration:
                            end_time = max_duration
                        time_tup = (start, end_time)
                    else:
                        end_time = max(time_tup[1], start + dur)
                        if end_time > max_duration:
                            end_time = max_duration
                        time_tup = (min(time_tup[0], start), end_time)
            dur2 = float("{:.3f}".format(time_tup[1] - time_tup[0]))
            if time_tup[0] < max_duration and dur2 > 0:
                meta = {"audio_filepath": audio_path, "offset": time_tup[0], "duration": dur2, "label": 'UNK'}
                json.dump(meta, outfile)
                outfile.write("\n")
            else:
                logging.warning("RTTM label has been truncated since start is greater than duration of audio file")
            f.close()
    return manifest_file


def segments_manifest_to_subsegments_manifest(
    segments_manifest_file: str,
    subsegments_manifest_file: str = None,
    window: float = 1.5,
    shift: float = 0.75,
    min_subsegment_duration: float = 0.05,
):
    """
    Generate subsegments manifest from segments manifest file
    Args
    input:
        segments_manifest file (str): path to segments manifest file, typically from VAD output
        subsegments_manifest_file (str): path to output subsegments manifest file (default (None) : writes to current working directory)
        window (float): window length for segments to subsegments length
        shift (float): hop length for subsegments shift
        min_subsegments_duration (float): exclude subsegments smaller than this duration value

    output:
        returns path to subsegment manifest file
    """
    if subsegments_manifest_file is None:
        pwd = os.getcwd()
        subsegments_manifest_file = os.path.join(pwd, 'subsegments.json')

    with open(segments_manifest_file, 'r') as segments_manifest, open(
        subsegments_manifest_file, 'w'
    ) as subsegments_manifest:
        segments = segments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)

            for subsegment in subsegments:
                start, dur = subsegment
                if dur > min_subsegment_duration:
                    meta = {"audio_filepath": audio, "offset": start, "duration": dur, "label": label}
                    json.dump(meta, subsegments_manifest)
                    subsegments_manifest.write("\n")

    return subsegments_manifest_file


def get_subsegments(offset: float, window: float, shift: float, duration: float):
    """
    return subsegments from a segment of audio file
    Args
    input:
        offset (float): start time of audio segment
        window (float): window length for segments to subsegments length
        shift (float): hop length for subsegments shift
        duration (float): duration of segment
    output:
        subsegments (List[tuple[float, float]]): subsegments generated for the segments as list of tuple of start and duration of each subsegment
    """
    subsegments = []
    start = offset
    slice_end = start + duration
    base = math.ceil((duration - window) / shift)
    slices = 1 if base < 0 else base + 1
    for slice_id in range(slices):
        end = start + window
        if end > slice_end:
            end = slice_end
        subsegments.append((start, end - start))
        start = offset + (slice_id + 1) * shift

    return subsegments


def embedding_normalize(embs, use_std=False, eps=1e-10):
    """
    mean and l2 length normalize the input speaker embeddings
    input:
        embs: embeddings of shape (Batch,emb_size)
    output:
        embs: normalized embeddings of shape (Batch,emb_size)
    """
    embs = embs - embs.mean(axis=0)
    if use_std:
        embs = embs / (embs.std(axis=0) + eps)
    embs_l2_norm = np.expand_dims(np.linalg.norm(embs, ord=2, axis=-1), axis=1)
    embs = embs / embs_l2_norm

    return embs

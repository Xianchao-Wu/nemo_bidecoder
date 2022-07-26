#########################################################################
# File Name: compute_cmvn_stats.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jul 26 20:52:53 2022
#########################################################################
#!/bin/bash

python compute_cmvn_stats.py \
	--num_workers=16 \
	--train_config="/workspace/asr/nemo1.6/NeMo/examples/asr/conf/conformer/conformer_ctc_char_librispeech_18blocks_bidecoder_full.yaml" \
	--in_train_json_file="/workspace/asr/LibriSpeech/LibriSpeech/libri.manifest.train.json" \
	--out_cmvn="/workspace/asr/LibriSpeech/LibriSpeech/cmvn/libri_global_cmvn.json" \
	--min_duration=0.1 

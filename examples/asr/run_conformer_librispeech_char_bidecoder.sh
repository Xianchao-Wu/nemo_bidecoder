#########################################################################
# File Name: run_csj.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Sep 20 11:28:39 2021
#########################################################################
#!/bin/bash
#	model.train_ds.manifest_filepath="/workspace/asr/csj/csj.manifest.train.json" \
#	model.train_ds.manifest_filepath="/workspace/asr/csj/csj.manifest.test1.json" \
#	model.validation_ds.manifest_filepath="[/workspace/asr/csj/csj.manifest.test1.json]" \
#	model.train_ds.manifest_filepath="/workspace/asr/csj/testsets/csj.manifest.test1.rno.txtnorm.dir.json" \
#	model.validation_ds.manifest_filepath="[/workspace/asr/csj/testsets/csj.manifest.test1.rno.txtnorm.dir.json,/workspace/asr/csj/testsets/csj.manifest.test2.rno.txtnorm.dir.json]" \
#	model.test_ds.manifest_filepath="[/workspace/asr/csj/testsets/csj.manifest.test1.rno.txtnorm.dir.json,/workspace/asr/csj/testsets/csj.manifest.test2.rno.txtnorm.dir.json]" \

#python -m ipdb speech_to_text_bpe.py \
#--config-path="/workspace/asr/nemo1.6/NeMo/examples/asr/conf/citrinet" \
#--config-name="citrinet_384_attn" \

#python asr_ctc/speech_to_text_ctc_bpe.py \
#python -m ipdb asr_ctc/speech_to_text_ctc.py \

#CUDA_VISIBLE_DEVICES=1 python -m ipdb asr_ctc/speech_to_text_ctc_bidecoder.py \
CUDA_VISIBLE_DEVICES=1,2 python asr_ctc/speech_to_text_ctc_bidecoder.py \
	--config-path="/workspace/asr/nemo1.6/NeMo/examples/asr/conf/conformer" \
	--config-name="conformer_ctc_char_librispeech_18blocks_bidecoder" \
	model.train_ds.manifest_filepath="/workspace/asr/LibriSpeech/LibriSpeech/libri.manifest.train.100lines.json" \
	model.validation_ds.manifest_filepath="[/workspace/asr/LibriSpeech/LibriSpeech/libri.manifest.dev.clean.30lines.json,/workspace/asr/LibriSpeech/LibriSpeech/libri.manifest.dev.other.30lines.json]" \
	model.test_ds.manifest_filepath="[/workspace/asr/LibriSpeech/LibriSpeech/libri.manifest.dev.clean.30lines.json,/workspace/asr/LibriSpeech/LibriSpeech/libri.manifest.dev.other.30lines.json]" \
	trainer.accelerator="ddp" \
	trainer.gpus=2 \
	trainer.max_epochs=10 \
	model.optim.name="adamw" \
	model.optim.lr=0.001 \
	model.optim.betas=[0.8,0.25] \
	model.optim.weight_decay=0.0001 \
	model.optim.sched.warmup_steps=20 \


	#trainer.accelerator="ddp" \
	#model.validation_ds.manifest_filepath="[/workspace/asr/csj/csj.manifest.test1.json,/workspace/asr/csj/csj.manifest.test1.json,/workspace/asr/csj/csj.manifest.test1.json]" \

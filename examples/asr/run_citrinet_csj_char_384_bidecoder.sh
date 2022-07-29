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
#python -m ipdb asr_ctc/speech_to_text_ctc_bpe.py \
#python -m ipdb asr_ctc/speech_to_text_ctc.py \
#asr_ctc/speech_to_text_ctc_bidecoder.py
python -m ipdb asr_ctc/speech_to_text_ctc_bidecoder.py \
	--config-path="/workspace/asr/nemo1.6/NeMo/examples/asr/conf/citrinet" \
	--config-name="citrinet_384_same_filters_cer_gamma_0_25_jp_asr_char_confirmed_20220619_bidecoder" \
	model.train_ds.manifest_filepath="/workspace/asr/csj/testsets/csj.manifest.test1.rno.txtnorm.dir.81.json" \
	model.train_ds.batch_size=2 \
	model.validation_ds.manifest_filepath="[/workspace/asr/csj/testsets/csj.manifest.test1.rno.txtnorm.dir.81.json,/workspace/asr/csj/testsets/csj.manifest.test2.rno.txtnorm.dir.91.json]" \
	model.validation_ds.batch_size=2 \
	model.test_ds.manifest_filepath="[/workspace/asr/csj/testsets/csj.manifest.test1.rno.txtnorm.dir.81.json,/workspace/asr/csj/testsets/csj.manifest.test2.rno.txtnorm.dir.91.json]" \
	model.test_ds.batch_size=2 \
	trainer.accelerator="dp" \
	trainer.gpus=1 \
	trainer.max_epochs=2 \
	model.optim.name="adamw" \
	model.optim.lr=0.001 \
	model.optim.betas=[0.8,0.25] \
	model.optim.weight_decay=0.0001 \
	model.optim.sched.warmup_steps=20


	#trainer.accelerator="ddp" \
	#model.validation_ds.manifest_filepath="[/workspace/asr/csj/csj.manifest.test1.json,/workspace/asr/csj/csj.manifest.test1.json,/workspace/asr/csj/csj.manifest.test1.json]" \
	#model.tokenizer.dir="/workspace/asr/csj/nemo.tokenizer.output/tokenizer_spe_bpe_v4096" \
	#model.tokenizer.type="bpe" \

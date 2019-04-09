#!/bin/bash
stage="$1" # parse first argument 

if [ $stage -eq 0 ]; then
    # call main.py; CPC train on LibriSpeech
    CUDA_VISIBLE_DEVICES=`free-gpu` python main.py \
	--train-raw LibriSpeech/train-Librispeech.h5 \
	--validation-raw LibriSpeech/validation-Librispeech.h5 \
	--eval-raw LibriSpeech/eval-Librispeech.h5 \
	--train-list LibriSpeech/list/train.txt \
        --validation-list LibriSpeech/list/validation.txt \
        --eval-list LibriSpeech/list/eval.txt \
        --logging-dir snapshot/cdc/ \
	--log-interval 50 --audio-window 20480 --timestep 12 --masked-frames 10 --n-warmup-steps 1000
fi

if [ $stage -eq 1 ]; then
    # call spk_class.py
    CUDA_VISIBLE_DEVICES=`free-gpu` python spk_class.py \
	--raw-hdf5 LibriSpeech/train-clean-100.h5 \
	--train-list LibriSpeech/list/train.txt \
        --validation-list LibriSpeech/list/validation.txt \
        --eval-list LibriSpeech/list/eval.txt \
	--index-file LibriSpeech/spk2idx \
        --logging-dir snapshot/cdc/ --log-interval 5 \
	--model-path snapshot/cdc/cdc-2018-09-17_22_08_37-model_best.pth 
fi



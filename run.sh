#!/bin/bash
#source /export/c01/jlai/thesis/cdc/path.sh
#source /export/c01/jlai/thesis/cdc/cmd.sh
#source /export/b18/nchen/keras/bin/activate
stage="$1" # parse first argument 

###################################### Below is for SITW ##################################################

if [ $stage -eq 88 ]; then
    # call retrieve4.py; forward pass cpc features for SITW 
    source /export/c01/jlai/thesis/cdc/path.sh
    source /export/c01/jlai/thesis/cdc/cmd.sh
    nj=40
    cpc="cpc_1"
    data_list="data_list-"

    for mode in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test; do
    # first split data list to 4 machines 
    	utils/split_scp.pl /export/c01/jlai/thesis/data/sitw/${mode}_list.txt /export/c{03,04,05,06}/$USER/cpc-data/egs/voxceleb2/v2/$cpc/$data_list${mode}
    	mkdir -p /export/c{03,04,05,06}/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/
    	for dir in c03 c04 c05 c06; do
		wavs=""
    		for n in $(seq $nj); do
			wavs="$wavs /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/$n.scp"
		done
		utils/split_scp.pl /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/$data_list${mode} $wavs
		$cmd JOB=1:$nj snapshot/cdc/forward_pass_${mode}_${dir}.JOB.log \
   		 /export/b18/nchen/keras/bin/python retrieve_cdc4.py \
			--utt2h5-file /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/JOB.scp \
			--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
			--log-interval 50 --timestep 12 --audio-window 60000 --batch-size 1 \
			--output-ark /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/${cpc}_${mode}.JOB.ark \
			--output-scp /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/${cpc}_${mode}.JOB.scp \
			--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2019-02-28_14_24_48-model_best.pth
    	done
	cat /export/c{03,04,05,06}/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/${cpc}_${mode}.*.scp > \
		/export/c01/jlai/thesis/xvector/sitw/cpc_feats/${cpc}_${mode}.scp
    done 
fi

if [ $stage -eq 89 ]; then
    # call retrieve4.py; forward pass cpc features for SITW 
    source /export/c01/jlai/thesis/cdc/path.sh
    source /export/c01/jlai/thesis/cdc/cmd.sh
    nj=40
    cpc="cpc_1"
    data_list="data_list-"

    for mode in dev train; do
    # first split data list to 4 machines 
    	utils/split_scp.pl /export/c01/jlai/thesis/data/sitw/train_combined_list.txt-${mode} /export/c{03,04,05,06}/$USER/cpc-data/egs/voxceleb2/v2/$cpc/$data_list${mode}
    	mkdir -p /export/c{03,04,05,06}/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/
    	for dir in c03 c04 c05 c06; do
		wavs=""
    		for n in $(seq $nj); do
			wavs="$wavs /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/$n.scp"
		done
		utils/split_scp.pl /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/$data_list${mode} $wavs
		$cmd JOB=1:$nj snapshot/cdc/forward_pass_sitw_${mode}_${dir}.JOB.log \
   		 /export/b18/nchen/keras/bin/python retrieve_cdc4.py \
			--utt2h5-file /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/JOB.scp \
			--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
			--log-interval 50 --timestep 12 --audio-window 60000 --batch-size 1 \
			--output-ark /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/${cpc}_${mode}.JOB.ark \
			--output-scp /export/$dir/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/${cpc}_${mode}.JOB.scp \
			--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2019-02-28_14_24_48-model_best.pth
    	done
	cat /export/c{03,04,05,06}/$USER/cpc-data/egs/voxceleb2/v2/$cpc/storage/cpc_feats/${cpc}_${mode}.*.scp > \
		/export/c01/jlai/thesis/xvector/sitw/cpc_feats/${cpc}_${mode}.scp
    done 
fi

if [ $stage -eq 90 ]; then
    # call main.py; CPC trained for SITW
    source /export/b18/nchen/keras/bin/activate
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main4.py \
	--train-utt2h5-file /export/c01/jlai/thesis/data/sitw/train_combined_list.txt-train \
	--dev-utt2h5-file  /export/c01/jlai/thesis/data/sitw/train_combined_list.txt-dev \
	--train-utt2len-file /export/c01/jlai/thesis/data/sitw/train_combined_utt2len-train \
	--dev-utt2len-file /export/c01/jlai/thesis/data/sitw/train_combined_utt2len-dev \
	--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	--log-interval 150 --audio-window 60000 --timestep 12 \
	--n-warmup-steps 16000 --batch-size 64 --epochs 30
fi

exit 0 
###################################### Below is for SRE16 ##################################################

if [ $stage -eq 99 ]; then
    # call retrieve3.py; forward pass cpc features for librispeech
    #$cmd JOB=1:20 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_librispeech_${mode}.JOB.log \
 	#	    /export/b18/nchen/keras/bin/python 
    source /export/c01/jlai/thesis/cdc/path.sh
    source /export/c01/jlai/thesis/cdc/cmd.sh
for mode in validation eval train; do 
   $cmd JOB=1:20 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_librispeech_${mode}.JOB.log \
   	/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc3.py \
		--data-raw /export/c01/jlai/thesis/data/LibriSpeech/${mode}-Librispeech.h5 \
		--data-list /export/c01/jlai/thesis/data/LibriSpeech/list/split20${mode}/${mode}.JOB.scp \
		--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
		--log-interval 50 --timestep 12 --batch-size 1 \
		--output-ark /export/b18/jlai/cdc-data/librispeech/cpc-23/cpc_${mode}.JOB.ark \
		--output-scp /export/c01/jlai/thesis/ivector/librispeech/cpc/cpc-23/cpc_${mode}.JOB.scp \
		--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-12-11_10_26_13-model_best.pth
done 
fi

if [ $stage -eq 100 ]; then
    # call main.py; CPC train on Sre16
    source /export/b18/nchen/keras/bin/activate
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main3.py \
	--sre-raw /export/c01/jlai/thesis/data/sre16/sre.h5 \
	--swbd-raw /export/c01/jlai/thesis/data/sre16/swbd.h5 \
	--dev-raw /export/c01/jlai/thesis/data/sre16/sre16_major.h5 \
	--sre-list /export/c01/jlai/thesis/data/sre16/list/sre.txt \
        --swbd-list /export/c01/jlai/thesis/data/sre16/list/swbd.txt \
        --dev-list /export/c01/jlai/thesis/data/sre16/list/sre16_major.txt \
	--sre-len /export/c01/jlai/thesis/data/sre16/utt2len/sre.txt \
	--swbd-len /export/c01/jlai/thesis/data/sre16/utt2len/swbd.txt \
	--dev-len /export/c01/jlai/thesis/data/sre16/utt2len/sre16_major.txt \
	--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	--log-interval 50 --audio-window 81920 --timestep 12 \
	--n-warmup-steps 1000 --batch-size 128 --epochs 60
fi

exit 0 
###################################### Below is for LibriSpeech ###############################################

if [ $stage -eq 98 ]; then
    # DON'T DO THIS HERE. REFER TO ivector/librispeech/feature_plot.py
    # call feature_plot.py; cpc features for librispeech
    python /export/c01/jlai/thesis/cdc/feature_plot.py \
	--data-raw /export/c01/jlai/thesis/data/LibriSpeech/train-Librispeech.h5 \
	--data-list /export/c01/jlai/thesis/data/LibriSpeech/list/train.txt \
	--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	--log-interval 50 --timestep 12 --batch-size 1 \
	--plot-wd /export/c01/jlai/thesis/cdc/plots/cdc-2018-11-17_12_04_45/train/ \
	--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-11-17_12_04_45-model_best.pth
fi

if [ $stage -eq 99 ]; then
    # call retrieve3.py; forward pass cpc features for librispeech
    #$cmd JOB=1:20 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_librispeech_${mode}.JOB.log \
    #	    /export/b18/nchen/keras/bin/python 
for mode in validation eval train; do 
   $cmd JOB=1:20 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_librispeech_${mode}.JOB.log \
   	/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc3.py \
		--data-raw /export/c01/jlai/thesis/data/LibriSpeech/${mode}-Librispeech.h5 \
		--data-list /export/c01/jlai/thesis/data/LibriSpeech/list/split20${mode}/${mode}.JOB.scp \
		--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
		--log-interval 50 --timestep 12 --batch-size 1 \
		--output-ark /export/b18/jlai/cdc-data/librispeech/cpc-23/cpc_${mode}.JOB.ark \
		--output-scp /export/c01/jlai/thesis/ivector/librispeech/cpc/cpc-23/cpc_${mode}.JOB.scp \
		--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-12-11_10_26_13-model_best.pth
done 
fi

if [ $stage -eq 100 ]; then
    # call main.py; CPC train on LibriSpeech
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main.py \
	--train-raw /export/c01/jlai/thesis/data/LibriSpeech/train-Librispeech.h5 \
	--validation-raw /export/c01/jlai/thesis/data/LibriSpeech/validation-Librispeech.h5 \
	--eval-raw /export/c01/jlai/thesis/data/LibriSpeech/eval-Librispeech.h5 \
	--train-list /export/c01/jlai/thesis/data/LibriSpeech/list/train.txt \
        --validation-list /export/c01/jlai/thesis/data/LibriSpeech/list/validation.txt \
        --eval-list /export/c01/jlai/thesis/data/LibriSpeech/list/eval.txt \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	--log-interval 50 --audio-window 20480 --timestep 12 --masked-frames 10 --n-warmup-steps 1000
fi

if [ $stage -eq 101 ]; then
    # call spk_class.py
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/spk_class.py \
	--raw-hdf5 /export/c01/jlai/thesis/data/LibriSpeech/train-clean-100.h5 \
	--train-list /export/c01/jlai/thesis/data/LibriSpeech/list/train.txt \
        --validation-list /export/c01/jlai/thesis/data/LibriSpeech/list/validation.txt \
        --eval-list /export/c01/jlai/thesis/data/LibriSpeech/list/eval.txt \
	--index-file /export/c01/jlai/thesis/data/LibriSpeech/spk2idx \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ --log-interval 5 \
	--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-09-17_22_08_37-model_best.pth 
fi

if [ $stage -eq 102 ]; then
    # call main2.py; CPC train on swbd_sre_combined_20k (list/training.txt, list/val.txt)
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main2.py \
	--raw-hdf5 /export/c01/jlai/thesis/data/swbd_sre_combined_20k/swbd_sre_combined_20k_20480.h5 \
	--train-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/training.txt \
        --validation-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/val.txt \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ --log-interval 20
fi

if [ $stage -eq 103 ]; then
    # call main2.py; CPC train on swbd_sre_combined_20k with silence frames (list/training2.txt, list/val2.txt)
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main2.py \
	--raw-hdf5 /export/c01/jlai/thesis/data/swbd_sre_combined_20k/swbd_sre_combined_20k.h5 \
	--train-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/training2.txt \
        --validation-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/val2.txt \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ --log-interval 20 --epochs 300 --timestep 12
fi

if [ $stage -eq 104 ]; then
    # call retrieve_cdc.py; forward pass cpc features of swbd_sre_combined with silence frames
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/retrieve_cdc.py \
	--wav-dir /export/c01/jlai/thesis/data/swbd_sre_combined/wav/ \
	--scp-list /export/c01/jlai/thesis/data/swbd_sre_combined/list/test.scp \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	--output-ark /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/test.ark \
	--output-scp /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/test.scp \
	--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
fi

if [ $stage -eq 105 ]; then
    # stage 104 run on cpu 
    python /export/c01/jlai/thesis/cdc/retrieve_cdc.py \
	--wav-dir /export/c01/jlai/thesis/data/swbd_sre_combined/wav/ \
	--scp-list /export/c01/jlai/thesis/data/swbd_sre_combined/list/test.scp \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	--output-ark /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/test.ark \
	--output-scp /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/test.scp \
	--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
fi

if [ $stage -eq 106 ]; then
    # parallelize stage 105, cpc model: cdc-2018-10-02_15_03_24
    $cmd JOB=1:50 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass.JOB.log \
	/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc.py \
	    --wav-dir /export/c01/jlai/thesis/data/swbd_sre_combined/wav/ \
	    --scp-list /export/c01/jlai/thesis/data/swbd_sre_combined/list/log/swbd_sre_utt.JOB.scp \
	    --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	    --output-ark /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/swbd_sre_utt.JOB.ark \
	    --output-scp /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/swbd_sre_utt.JOB.scp \
	    --model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
fi

if [ $stage -eq 107 ]; then
    # parallelize stage 105, cpc model: cdc-2018-10-02_15_31_55
    $cmd JOB=1:50 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_cdc-2018-10-02_15_31_55.JOB.log \
	/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc.py \
	    --wav-dir /export/c01/jlai/thesis/data/swbd_sre_combined/wav/ \
	    --scp-list /export/c01/jlai/thesis/data/swbd_sre_combined/list/log/swbd_sre_utt.JOB.scp \
	    --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
	    --output-ark /export/b06/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_31_55/swbd_sre_utt.JOB.ark \
	    --output-scp /export/b06/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_31_55/swbd_sre_utt.JOB.scp \
	    --model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_31_55-model_best.pth
fi

if [ $stage -eq 108 ]; then
    # call main2.py; CPC train on swbd_sre_combined_20k without silence frames (list/training3.txt, list/val3.txt)
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main2.py \
	--raw-hdf5 /export/c01/jlai/thesis/data/swbd_sre_combined_20k/swbd_sre_combined_20k_20480.h5 \
	--train-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/training3.txt \
        --validation-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/val3.txt \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ --log-interval 20 --epochs 400 --timestep 12
fi

if [ $stage -eq 109 ]; then
    # call main2.py; CPC train on swbd_sre_combined_20k without silence frames (list/training3.txt, list/val3.txt)
    CUDA_VISIBLE_DEVICES=`free-gpu` python /export/c01/jlai/thesis/cdc/main2.py \
	--raw-hdf5 /export/c01/jlai/thesis/data/swbd_sre_combined_20k/swbd_sre_combined_20k_20480.h5 \
	--train-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/training3.txt \
        --validation-list /export/c01/jlai/thesis/data/swbd_sre_combined_20k/list/val3.txt \
        --logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ --log-interval 20 --epochs 400 --timestep 12
fi

if [ $stage -eq 110 ]; then
    # call dct_cdc.py; take 256 cpc features and apply dct and store the resulting features
    $cmd JOB=1:50 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_dct-2018-10-02_15_03_24.JOB.log \
    	/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/dct_cdc.py \
		--input-scp /export/b18/jlai/cdc-data/swbd-sre/cdc-2018-10-02_15_03_24/swbd_sre_utt.JOB.scp \
		--output-scp /export/c05/jlai/cdc-data/swbd-sre/dct-2018-10-02_15_03_24/swbd_sre_utt.JOB.scp \
		--output-ark /export/c05/jlai/cdc-data/swbd-sre/dct-2018-10-02_15_03_24/swbd_sre_utt.JOB.ark \
		--dct-dim 23
fi

if [ $stage -eq 111 ]; then
    # call retrieve_cdc.py; forward pass cpc features of sre16 with silence frames
    for name in sre16_eval_enroll sre16_eval_test sre16_dev_enroll sre16_dev_test sre16_major sre16_minor; do
	    $cmd JOB=1:20 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_sre16.JOB.log \
		/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc.py \
			--wav-dir /export/c01/jlai/thesis/data/$name/wav/ \
			--scp-list /export/c01/jlai/thesis/data/$name/list/list.JOB.scp \
			--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
			--output-ark /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.ark \
			--output-scp /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.scp \
			--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
    done
fi

if [ $stage -eq 112 ]; then
    # call dct_cdc.py; take 256 cpc features and apply dct and store the resulting features
    for name in sre16_eval_enroll sre16_eval_test sre16_dev_enroll sre16_dev_test sre16_major sre16_minor; do
	    $cmd JOB=1:20 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_dct_sre16.JOB.log \
		/export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/dct_cdc.py \
			--input-scp /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.scp \
			--output-scp /export/c01/jlai/thesis/data/$name/cdc-23/cdc-2018-10-02_15_03_24.JOB.scp \
			--output-ark /export/c01/jlai/thesis/data/$name/cdc-23/cdc-2018-10-02_15_03_24.JOB.ark \
			--dct-dim 23
    done 
fi

if [ $stage -eq 113 ]; then
    # call retrieve_cdc2.py; forward pass cpc features of sitw with silence frames
    for name in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test; do
	    $cmd JOB=1:60 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_sitw.JOB.log \
		     /export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc2.py \
			--wav-file /export/c01/jlai/thesis/data/$name/list/wav.JOB.scp \
			--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
			--output-ark /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.ark \
			--output-scp /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.scp \
			--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
    done
fi

if [ $stage -eq 114 ]; then
    # call retrieve_cdc2.py; forward pass cpc features of sitw with silence frames
    for name in sitw_eval_test; do
	    $cmd JOB=1:60 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_sitw.JOB.log \
		     /export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc2.py \
			--wav-file /export/c01/jlai/thesis/data/$name/list/wav.JOB.scp \
			--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
			--output-ark /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.ark \
			--output-scp /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.scp \
			--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
    done
fi

if [ $stage -eq 115 ]; then
    # rerun sitw_eval_enroll 6 8 38
    for name in sitw_eval_enroll; do
	    $cmd JOB=38 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_sitw.JOB.log \
		     /export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/retrieve_cdc2.py \
			--wav-file /export/c01/jlai/thesis/data/$name/list/wav.JOB.scp \
			--logging-dir /export/c01/jlai/thesis/cdc/snapshot/cdc/ \
			--output-ark /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.ark \
			--output-scp /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.scp \
			--model-path /export/c01/jlai/thesis/cdc/snapshot/cdc/cdc-2018-10-02_15_03_24-model_best.pth
    done
fi

if [ $stage -eq 116 ]; then
    # call dct_cdc.py; take 256 cpc features and apply dct and store the resulting features
    for name in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test; do
	    $cmd JOB=1:60 /export/c01/jlai/thesis/cdc/snapshot/cdc/forward_pass_dct_sitw.JOB.log \
		     /export/b18/nchen/keras/bin/python /export/c01/jlai/thesis/cdc/dct_cdc.py \
			--input-scp /export/c01/jlai/thesis/data/$name/cdc-256/cdc-2018-10-02_15_03_24.JOB.scp \
			--output-scp /export/c01/jlai/thesis/data/$name/cdc-23/cdc-2018-10-02_15_03_24.JOB.scp \
			--output-ark /export/c01/jlai/thesis/data/$name/cdc-23/cdc-2018-10-02_15_03_24.JOB.ark \
			--dct-dim 23
    done
fi


#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# librispeech trials
librispeech_trials=data/eval_test/trials
# set which cpc feature 
cpc=cpc_16
cpc2=cpc-16

stage=5
if [ $stage -le 0 ]; then 
  # prepare features 
  for name in train; do 
	utils/fix_data_dir.sh data/${name}_cpc_1/  
	utils/copy_data_dir.sh data/${name}_cpc_1/ data/${name}_${cpc}/
	cp cpc/${cpc2}/cpc_${name}.scp data/${name}_${cpc}/feats.scp
	utils/fix_data_dir.sh data/${name}_${cpc}/
  done
  for name in eval_enroll eval_test; do 
	utils/fix_data_dir.sh data/${name}_cpc_1/  
	utils/copy_data_dir.sh data/${name}_cpc_1/ data/${name}_${cpc}/
	utils/filter_scp.pl data/${name}_${cpc}/utt2spk cpc/${cpc2}/cpc_eval.scp > data/${name}_${cpc}/feats.scp	
	utils/fix_data_dir.sh data/${name}_${cpc}/
  done
fi 

if [ $stage -le 1 ]; then
  # energy-based VAD is based on MFCC
  # remember to fix vad frames mismatches with adjust_vad.py!!!
  for name in eval_enroll eval_test; do 
	utils/filter_scp.pl data/${name}_${cpc}/utt2spk data/eval_cpc_1/vad.scp > data/${name}_${cpc}/vad.scp
	utils/fix_data_dir.sh data/${name}_${cpc}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    	--nj 40 --num-threads 8  --subsample 1 \
    	data/train_${cpc} 2048 \
    	exp/diag_ubm_${cpc}

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    	--nj 40 --remove-low-count-gaussians false --subsample 1 \
    	data/train_${cpc} \
    	exp/diag_ubm_${cpc} exp/full_ubm_${cpc}
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
	--ivector-dim 600 \
    	--num-iters 5 \
    	--num-threads 4 \
    	--num-processes 4 \
    	exp/full_ubm_${cpc}/final.ubm data/train_${cpc} \
    	exp/extractor_${cpc}
fi

if [ $stage -le 4 ]; then 
  # split data/train_${cpc} into data/train_${cpc}_subset_{n}
  sh split_list.sh ${cpc}
fi

if [ $stage -le 5 ]; then
  # extract i-vectors for train
  for n in $(seq 5 10); do
  for name in train; do
  	sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 30 \
    		exp/extractor_${cpc} data/${name}_${cpc}_subset_${n} \
    		exp/ivectors_${name}_${cpc}_subset_${n}
  done
  done
  if [ ! -d exp/ivectors_${name}_${cpc} ]; then
  	mkdir exp/ivectors_${name}_${cpc}	
  fi 
  name=train
  cat exp/ivectors_${name}_${cpc}_subset_{1..10}/ivector.scp > exp/ivectors_${name}_${cpc}/ivector.scp
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 30 \
	--stage 2 \
    	exp/extractor_${cpc} data/train_${cpc} \
    	exp/ivectors_train_${cpc}
fi

if [ $stage -le 6 ]; then
  # extract i-vectors for eval
  for name in eval_enroll eval_test; do
	sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 30 \
    		exp/extractor_${cpc} data/${name}_${cpc} \
    		exp/ivectors_${name}_${cpc}
  done
fi

if [ $stage -le 7 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train_${cpc}/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train_${cpc}/ivector.scp \
    exp/ivectors_train_${cpc}/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_train_${cpc}/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_${cpc}/ivector.scp ark:- |" \
    ark:data/train_${cpc}/utt2spk exp/ivectors_train_${cpc}/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd exp/ivectors_train_${cpc}/log/plda.log \
    ivector-compute-plda ark:data/train_${cpc}/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_${cpc}/ivector.scp ark:- | transform-vec exp/ivectors_train_${cpc}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train_${cpc}/plda || exit 1;
fi

if [ $stage -le 8 ]; then
  # Get results using the out-of-domain PLDA model
  $train_cmd exp/scores_${cpc}/log/eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_eval_enroll_${cpc}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_${cpc}/plda - |" \
    "ark:ivector-mean ark:data/eval_enroll_${cpc}/spk2utt scp:exp/ivectors_eval_enroll_${cpc}/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_train_${cpc}/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_${cpc}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${cpc}/mean.vec scp:exp/ivectors_eval_test_${cpc}/ivector.scp ark:- | transform-vec exp/ivectors_train_${cpc}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$librispeech_trials' | cut -d\  --fields=1,2 |" exp/scores_${cpc}/eval_scores || exit 1;

  pooled_eer=$(paste $librispeech_trials exp/scores_${cpc}/eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%"
  # EER: 3.712%
  exit
fi

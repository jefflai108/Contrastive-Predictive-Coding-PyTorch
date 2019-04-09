# Contrastive-Predictive-Coding-PyTorch
This repository contains (PyTorch) code to reproduce the core results for: 
* [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)
* [Contrastive Predictive Coding Based
Feature for Automatic Speaker Verification](https://arxiv.org/pdf/1904.01575.pdf)

<p align="center">
 <img src="img/CDCK2.png" width="60%">
</p>
<p align="center">
 <img src="img/CPC-ivector.png" width="80%">
</p>

## Getting Started
`./src/model/model.py` contains the CPC model implementation, `./src/main.py` is the code for training the CPC model, `./src/spk_class.py` trains a NN speaker classifier, `./ivector/` contains the scripts for running an i-vectors speaker verification system. 

## CPC Models 
CDCK2: base model from the paper 'Representation Learning with Contrastive Predictive Coding'
CDCK5: CDCK2 with a different decoder 
CDCK6: CDCK2 with a shared encoder and double decoders 

# Experimental Results 

## CPC Model Training 
|        CPC model ID         | number of epoch |   model size   |  dev NCE loss   |    dev acc.    |  
| :-------------------------: | :-------------: | :------------: | :-------------: | :------------: |
|           CDCK2             |        60       |     7.42M      |      1.6427     |      26.42     |  
|           CDCK5             |        60       |     5.58M      |      1.7818     |      22.48     |
|           CDCK6             |        30       |     7.33M      |      1.6484     |      28.24     |

## Speaker Verificaiton on LibriSpeech test-clean-100
Note trail list link 
|        CPC model ID         | number of epoch |   model size   |  dev NCE loss   |    dev acc.    |  
| :-------------------------: | :-------------: | :------------: | :-------------: | :------------: |
|           CDCK2             |        60       |     7.42M      |      1.6427     |      26.42     |  
|           CDCK5             |        60       |     5.58M      |      1.7818     |      22.48     |
|           CDCK6             |        30       |     7.33M      |      1.6484     |      28.24     |
|           CDCK6             |        30       |     7.33M      |      1.6484     |      28.24     |

## CPC with PCA 

```
./run_feature.sh.sh
```


## Authors 
Cheng-I Lai.

If you encouter any problem, feel free to contact me.

import numpy as np
import logging
import torch
import torch.nn.functional as F
import kaldi_io as ko
import scipy.fftpack as fft
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

## Get the same logger from main"
logger = logging.getLogger("cdc")

def forwardXXreverse(args, cpc_model, device, data_loader, output_ark, output_scp):
    logger.info("Starting Forward Passing")
    cpc_model.eval() # not training cdc model 

    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:' + output_ark + ',' + output_scp 

    with torch.no_grad():
        with ko.open_or_fd(ark_scp_output,'wb') as f:
            for [utt_id, data, data_r] in data_loader:
                data   = data.float().unsqueeze(1).to(device) # add channel dimension
                data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
                data   = data.contiguous()
                data_r = data.contiguous()
                hidden1 = cpc_model.init_hidden1(len(data))
                hidden2 = cpc_model.init_hidden2(len(data))
                output = cpc_model.predict(data, data_r, hidden1, hidden2)
                mat = output.squeeze(0).cpu().numpy() # kaldi io does not accept torch tensor
                ko.write_mat(f, mat, key=utt_id[0])

def forward_dct(args, cpc_model, device, data_loader, output_ark, output_scp, dct_dim=24):
    ''' forward with dct '''

    logger.info("Starting Forward Passing")
    cpc_model.eval() # not training cdc model 

    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:' + output_ark + ',' + output_scp 
    with torch.no_grad():
        with ko.open_or_fd(ark_scp_output,'wb') as f:
            for [utt_id, data] in data_loader:
                data = data.float().unsqueeze(1).to(device) # add channel dimension
                data = data.contiguous()
                hidden = cpc_model.init_hidden(len(data))
                output, hidden = cpc_model.predict(data, hidden)
                mat = output.squeeze(0).cpu().numpy() # kaldi io does not accept torch tensor
                dct_mat = fft.dct(mat, type=2, n=dct_dim) # apply dct 
                ko.write_mat(f, dct_mat, key=utt_id[0])


def forward(cpc_model, device, data_loader, output_ark, output_scp):
    logger.info("Starting Forward Passing")
    cpc_model.eval() # not training cdc model 

    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:' + output_ark + ',' + output_scp 

    with torch.no_grad():
        with ko.open_or_fd(ark_scp_output,'wb') as f:
            for [utt_id, data] in data_loader:
                data = data.float().unsqueeze(1).to(device) # add channel dimension
                data = data.contiguous()
                hidden = cpc_model.init_hidden(len(data), use_gpu=False)
                output, hidden = cpc_model.predict(data, hidden)
                mat = output.squeeze(0).cpu().numpy() # kaldi io does not accept torch tensor
                ko.write_mat(f, mat, key=utt_id[0])


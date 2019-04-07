import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cdc")

def prediction_spk(args, cdc_model, spk_model, device, data_loader, batch_size, frame_window):
    logger.info("Starting Evaluation")
    cdc_model.eval() # not training cdc model 
    spk_model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for [data, target] in data_loader:
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            target = target.to(device)
            hidden = cdc_model.init_hidden(len(data))
            output, hidden = cdc_model.predict(data, hidden)
            data = output.contiguous().view((-1,256))
            target = target.view((-1,))
            output = spk_model.forward(data)
            total_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            total_acc += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(data_loader.dataset)*frame_window # average loss
    total_acc  /= 1.*len(data_loader.dataset)*frame_window # average acc

    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> Evaluation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

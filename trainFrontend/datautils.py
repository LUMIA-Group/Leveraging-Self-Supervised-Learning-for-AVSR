import cv2 as cv
import numpy as np
import torch


def prepare_input(index, h5, trgt, wordToIx, transform):
    """
    Function to convert the data sample in the LRW dataset into appropriate tensors.
    """
    wordTrgt = wordToIx[trgt]
    wordTrgt = np.array([wordTrgt])

    # visual file
    vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
    vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
    vidInp = torch.tensor(vidInp).unsqueeze(1)
    vidInp = transform(vidInp)

    wordTrgt = torch.from_numpy(wordTrgt)

    return vidInp, wordTrgt


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    vis_seq_list = torch.cat([data[0] for data in dataBatch])
    wordtargetBatch = torch.cat([data[1] for data in dataBatch])

    return vis_seq_list, wordtargetBatch

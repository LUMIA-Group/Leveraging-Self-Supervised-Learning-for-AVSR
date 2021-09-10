from itertools import groupby

import numpy as np
import torch

np.seterr(divide="ignore")


def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0):
    """
    Greedy search technique for CTC decoding.
    This decoding method selects the most probable character at each time step. This is followed by the usual CTC decoding
    to get the predicted transcription.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

    outputBatch = outputBatch.cpu()
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:, :, blank] = torch.log(torch.exp(outputBatch[:, :, blank]) + torch.exp(outputBatch[:, :, eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:, :, reqIxs]

    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()
    inpLens = inputLenBatch.numpy()
    preds = list()
    predLens = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]
        pred = np.array([x[0] for x in groupby(pred)])
        pred = pred[pred != blank]
        pred = list(pred)
        pred.append(eosIx)
        preds.extend(pred)
        predLens.append(len(pred))
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch


def teacher_forcing_attention_decode(outputBatch, eosIx):
    outputBatch = outputBatch.cpu()
    predCharIxs = torch.argmax(outputBatch, dim=-1)
    seqLength = outputBatch.shape[1] - 1
    predictionBatch = []
    predictionLenBatch = []
    for pred in predCharIxs:
        firstEOSIx = seqLength if len((pred == eosIx).nonzero()) == 0 else (pred == eosIx).nonzero()[0]
        predictionBatch.append(pred[:firstEOSIx + 1] if pred[firstEOSIx] == eosIx else torch.cat((pred[:firstEOSIx + 1], torch.tensor([eosIx])), -1))
        predictionLenBatch.append(firstEOSIx + 1 if pred[firstEOSIx] == eosIx else firstEOSIx + 2)

    predictionBatch = torch.cat(predictionBatch, 0).int()
    predictionLenBatch = torch.tensor(predictionLenBatch)
    return predictionBatch, predictionLenBatch


def compute_CTC_prob(h, alpha, CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, numClasses, blank, eosIx):
    batch = h.shape[0]
    g = h[:, :, :, :-1]
    c = h[:, :, :, -1]
    alphaCTC = torch.zeros_like(alpha)
    eosIxMask = c == eosIx
    eosIxIndex = eosIxMask.nonzero()
    eosIxIndex = torch.cat((eosIxIndex[:, :1], torch.repeat_interleave((T - 1).unsqueeze(-1), numBeam, dim=0), eosIxIndex[:, 1:]), dim=-1).long()
    eosIxIndex[:, -1] = 0
    gamma_eosIxMask = torch.zeros_like(gamma_n).bool()
    gamma_eosIxMask.index_put_(tuple(map(torch.stack, zip(*eosIxIndex))), torch.tensor(True))
    alphaCTC[eosIxMask] = np.logaddexp(gamma_n[gamma_eosIxMask], gamma_b[gamma_eosIxMask])

    if g.shape[-1] == 1:
        gamma_n[:, 1, 0, 1:-1] = CTCOutLogProbs[:, 1, 1:-1]
    else:
        gamma_n[:, 1, :numBeam, 1:-1] = -np.inf
    gamma_b[:, 1, :numBeam, 1:-1] = -np.inf

    psi = gamma_n[:, 1, :numBeam, 1:-1]
    for t in range(2, T.max()):
        activeBatch = t < T
        gEndWithc = (g[:, :, :, -1] == c)[:, :, :-1].nonzero()
        added_gamma_n = torch.repeat_interleave(gamma_n[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1)
        if len(gEndWithc):
            added_gamma_n.index_put_(tuple(map(torch.stack, zip(*gEndWithc))), torch.tensor(-np.inf).float())
        phi = np.logaddexp(torch.repeat_interleave(gamma_b[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1), added_gamma_n)
        expandShape = [batch, numBeam, numClasses - 1]
        gamma_n[:, t, :numBeam, 1:-1][activeBatch] = np.logaddexp(gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch], phi[activeBatch]) \
                                                     + CTCOutLogProbs[:, t, None, 1:-1].expand(expandShape)[activeBatch]
        gamma_b[:, t, :numBeam, 1:-1][activeBatch] = \
            np.logaddexp(gamma_b[:, t - 1, :numBeam, 1:-1][activeBatch], gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch]) \
            + CTCOutLogProbs[:, t, None, None, blank].expand(expandShape)[activeBatch]
        psi[activeBatch] = np.logaddexp(psi[activeBatch], phi[activeBatch] + CTCOutLogProbs[:, t, None, 1:-1].expand(phi.shape)[activeBatch])
    return torch.cat((psi, alphaCTC[:, :, -1:]), dim=-1)

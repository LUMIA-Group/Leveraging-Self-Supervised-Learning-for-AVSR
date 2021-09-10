import editdistance
import numpy as np
import torch


def compute_error_ch(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):
    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    totalEdits = 0
    totalChars = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]
        trgt = trgts[n].numpy()[:-1]
        numEdits = editdistance.eval(pred, trgt)
        totalEdits = totalEdits + numEdits
        totalChars = totalChars + len(trgt)

    return totalEdits, totalChars


def compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):
    """
    Function to compute the Character Error Rate using the Predicted character indices and the Target character
    indices over a batch.
    CER is computed by dividing the total number of character edits (computed using the editdistance package)
    with the total number of characters (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the CER.
    """
    
    totalEdits, totalChars = compute_error_ch(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)

    return totalEdits / totalChars


def compute_error_word(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):
    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    totalEdits = 0
    totalWords = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]
        trgt = trgts[n].numpy()[:-1]

        predWords = np.split(pred, np.where(pred == spaceIx)[0])
        predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]

        trgtWords = np.split(trgt, np.where(trgt == spaceIx)[0])
        trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]

        numEdits = editdistance.eval(predWords, trgtWords)
        totalEdits = totalEdits + numEdits
        totalWords = totalWords + len(trgtWords)

    return totalEdits, totalWords


def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):
    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    totalEdits, totalWords = compute_error_word(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx)

    return totalEdits / totalWords

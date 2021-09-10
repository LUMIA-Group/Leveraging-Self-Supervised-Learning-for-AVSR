import os

import torch
from tqdm import tqdm

from config import args
from .decoders import ctc_greedy_decode, teacher_forcing_attention_decode
from .metrics import compute_error_ch, compute_error_word


def index_to_string(indexBatch):
    return "".join([args["INDEX_TO_CHAR"][ix] if ix > 0 else "" for ix in indexBatch.tolist()])


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def inference(model, evalLoader, device, logger, inferenceParams):
    evalCER = 0
    evalWER = 0
    evalPER = 0
    evalCCount = 0
    evalWCount = 0
    evalPCount = 0

    Lambda = inferenceParams["Lambda"]
    if os.path.exists(args["CODE_DIRECTORY"] + "pred_%s.txt" % inferenceParams["decodeType"]):
        os.remove(args["CODE_DIRECTORY"] + "pred_%s.txt" % inferenceParams["decodeType"])
    if os.path.exists(args["CODE_DIRECTORY"] + "trgt.txt"):
        os.remove(args["CODE_DIRECTORY"] + "trgt.txt")

    model.eval()
    for batch, (inputBatch, targetinBatch, targetoutBatch, targetLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval", ncols=75)):
        if inferenceParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float().to(device), inputBatch[1].to(device), None, None)
        elif inferenceParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float().to(device), inputBatch[3].to(device))
        else:
            inputBatch = \
                (inputBatch[0].float().to(device), inputBatch[1].to(device), inputBatch[2].float().to(device), inputBatch[3].to(device))
        targetinBatch = targetinBatch.int().to(device)
        targetoutBatch = targetoutBatch.int().to(device)
        targetLenBatch = targetLenBatch.int().to(device)
        targetMask = torch.zeros((targetLenBatch.shape[0], targetLenBatch.max()), device=targetLenBatch.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]

        with torch.no_grad():
            if inferenceParams["decodeType"] == "HYBRID":
                predictionBatch, predictionLenBatch = \
                    model.inference(inputBatch, False, device, Lambda, inferenceParams["beamWidth"], inferenceParams["eosIx"], 0)
            elif inferenceParams["decodeType"] == "ATTN":
                predictionBatch, predictionLenBatch = \
                    model.attentionAutoregression(inputBatch, False, device, inferenceParams["eosIx"])
            elif inferenceParams["decodeType"] == "TFATTN":
                inputLenBatch, outputBatch = model(inputBatch, targetinBatch, targetLenBatch.long(), False)
                predictionBatch, predictionLenBatch = teacher_forcing_attention_decode(outputBatch[1], inferenceParams["eosIx"])
            elif inferenceParams["decodeType"] == "CTC":
                inputLenBatch, outputBatch = model(inputBatch, targetinBatch, targetLenBatch.long(), False)
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0], inputLenBatch, inferenceParams["eosIx"])
            else:
                predictionBatch, predictionLenBatch = None, None

            predictionStr = index_to_string(predictionBatch).replace('<EOS>', '\n')
            targetStr = index_to_string(concatTargetoutBatch).replace('<EOS>', '\n')

            with open("pred_%s.txt" % inferenceParams["decodeType"], "a") as f:
                f.write(predictionStr)

            with open("trgt.txt", "a") as f:
                f.write(targetStr)

            c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
            evalCER += c_edits
            evalCCount += c_count
            w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch,
                                                  inferenceParams["spaceIx"])
            evalWER += w_edits
            evalWCount += w_count
            print("batch%d || Test CER: %.3f || Test WER: %.3f" % (batch + 1, evalCER / evalCCount, evalWER / evalWCount))
            logger.info("batch%d || Test CER: %.3f || Test WER: %.3f" % (batch + 1, evalCER / evalCCount, evalWER / evalWCount))

    evalCER /= evalCCount if evalCCount > 0 else 1
    evalWER /= evalWCount if evalWCount > 0 else 1
    evalPER /= evalPCount if evalPCount > 0 else 1
    return evalCER, evalWER, evalPER

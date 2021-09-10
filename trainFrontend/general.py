import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, trainLoader, optimizer, loss_function, device):
    trainingLoss = 0
    trainingCorr = 0
    cnt = 0
    model.train()

    for batch, (inputBatch, wordtargetBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train", ncols=75)):
        inputBatch = inputBatch.float().to(device)
        wordtargetBatch = wordtargetBatch.long().to(device)
        optimizer.zero_grad()
        outputBatch = model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = loss_function(outputBatch, wordtargetBatch)
        loss.backward()
        optimizer.step()
        trainingLoss += loss.item()
        predictionBatch = torch.argmax(F.softmax(outputBatch, dim=1), 1)
        cnt += len(wordtargetBatch)
        trainingCorr += torch.sum(predictionBatch == wordtargetBatch)

    trainingLoss /= len(trainLoader)
    return trainingLoss, trainingCorr / cnt


def evaluate(model, evalLoader, loss_function, device):
    evalLoss = 0
    evalCorr = 0
    cnt = 0
    model.eval()

    for batch, (inputBatch, wordtargetBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval", ncols=75)):
        inputBatch = inputBatch.float().to(device)
        wordtargetBatch = wordtargetBatch.long().to(device)
        with torch.no_grad():
            outputBatch = model(inputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, wordtargetBatch)
        evalLoss += loss.item()
        predictionBatch = torch.argmax(F.softmax(outputBatch, dim=1), 1)
        cnt += len(wordtargetBatch)
        evalCorr += torch.sum(predictionBatch == wordtargetBatch)

    evalLoss /= len(evalLoader)
    return evalLoss, evalCorr / cnt

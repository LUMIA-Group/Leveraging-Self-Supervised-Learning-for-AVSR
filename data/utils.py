import cv2 as cv
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from config import args


def prepare_main_input(index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR):
    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """
    with open(targetFile, "r") as f:
        trgt = f.readline().strip()[7:]

    trgtin = [charToIx[item] for item in trgt]
    trgtin.insert(0, charToIx["<EOS>"])
    trgtout = [charToIx[item] for item in trgt]
    trgtout.append(charToIx["<EOS>"])
    trgtin = np.array(trgtin)
    trgtout = np.array(trgtout)
    trgtLen = len(trgtout)

    # audio file
    if not modal == "VO":
        audInp = np.array(h5["flac"][index])
        audInp = (audInp - audInp.mean()) / audInp.std()
        if noise is not None:
            pos = np.random.randint(0, len(noise) - len(audInp) + 1)
            noise = noise[pos:pos + len(audInp)]
            noise = noise / np.max(np.abs(noise))
            gain = 10 ** (noiseSNR / 10)
            noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
            audInp = audInp + noise
        audInp = torch.from_numpy(audInp)
    else:
        audInp = None

    # visual file
    if not modal == "AO":
        vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
        vidInp = torch.tensor(vidInp).unsqueeze(1)
        vidInp = transform(vidInp)
    else:
        vidInp = None

    inp = (audInp, vidInp)
    trgtin = torch.from_numpy(trgtin)
    trgtout = torch.from_numpy(trgtout)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgtin, trgtout, trgtLen


def prepare_pretrain_input(index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR, numWordsRange, maxLength):
    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    # reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    numWords = len(words) // 3
    if numWords < numWordsRange[0]:
        numWords = numWordsRange[0]
    elif numWords > numWordsRange[1]:
        numWords = numWordsRange[1]

    while True:
        # if number of words in target is less than the required number of words, consider the whole target
        if len(words) <= numWords:
            trgtNWord = trgt
            # audio file
            if not modal == "VO":
                audInp = np.array(h5["flac"][index])
                audInp = (audInp - audInp.mean()) / audInp.std()
                if noise is not None:
                    pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                    noise = noise[pos:pos + len(audInp)]
                    noise = noise / np.max(np.abs(noise))
                    gain = 10 ** (noiseSNR / 10)
                    noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                    audInp = audInp + noise
                audInp = torch.from_numpy(audInp)
            else:
                audInp = None

            # visual file
            if not modal == "AO":
                vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
                vidInp = torch.tensor(vidInp).unsqueeze(1)
                vidInp = transform(vidInp)
            else:
                vidInp = None

        else:
            # make a list of all possible sub-sequences with required number of words in the target
            nWords = [" ".join(words[i:i + numWords])
                      for i in range(len(words) - numWords + 1)]
            nWordLens = np.array(
                [len(nWord) + 1 for nWord in nWords]).astype(np.float)

            # choose the sub-sequence for target according to a softmax distribution of the lengths
            # this way longer sub-sequences (which are more diverse) are selected more often while
            # the shorter sub-sequences (which appear more frequently) are not entirely missed out
            ix = np.random.choice(np.arange(len(nWordLens)), p=nWordLens / nWordLens.sum())
            trgtNWord = nWords[ix]

            # reading the start and end times in the video corresponding to the selected sub-sequence
            startTime = float(lines[4 + ix].split(" ")[1])
            endTime = float(lines[4 + ix + numWords - 1].split(" ")[2])
            # audio file
            if not modal == "VO":
                samplerate = 16000
                audInp = np.array(h5["flac"][index])
                audInp = (audInp - audInp.mean()) / audInp.std()
                if noise is not None:
                    pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                    noise = noise[pos:pos + len(audInp)]
                    noise = noise / np.max(np.abs(noise))
                    gain = 10 ** (noiseSNR / 10)
                    noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                    audInp = audInp + noise
                audInp = torch.from_numpy(audInp)
                audInp = audInp[int(samplerate * startTime):int(samplerate * endTime)]
            else:
                audInp = None

            # visual file
            if not modal == "AO":
                videoFPS = 25
                vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
                vidInp = torch.tensor(vidInp).unsqueeze(1)
                vidInp = transform(vidInp)
                vidInp = vidInp[int(np.floor(videoFPS * startTime)): int(np.ceil(videoFPS * endTime))]
            else:
                vidInp = None

        # converting each character in target to its corresponding index
        trgtin = [charToIx[item] for item in trgtNWord]
        trgtout = [charToIx[item] for item in trgtNWord]
        trgtin.insert(0, charToIx["<EOS>"])
        trgtout.append(charToIx["<EOS>"])
        trgtin = np.array(trgtin)
        trgtout = np.array(trgtout)
        trgtLen = len(trgtout)

        inp = (audInp, vidInp)
        trgtin = torch.from_numpy(trgtin)
        trgtout = torch.from_numpy(trgtout)
        trgtLen = torch.tensor(trgtLen)
        inpLen = len(vidInp) if not args["MODAL"] == "AO" else len(audInp) / 640
        if inpLen <= maxLength:
            break
        elif inpLen > maxLength + 80:
            numWords -= 2
        else:
            numWords -= 1

    return inp, trgtin, trgtout, trgtLen


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    # audio & mask
    if not args["MODAL"] == "VO":
        aud_seq_list = [data[0][0] for data in dataBatch]
        aud_padding_mask = torch.zeros((len(aud_seq_list), len(max(aud_seq_list, key=len))), dtype=torch.bool)
        for i, seq in enumerate(aud_seq_list):
            aud_padding_mask[i, len(seq):] = True
        aud_seq_list = pad_sequence(aud_seq_list, batch_first=True)
    else:
        aud_seq_list = None
        aud_padding_mask = None
    # visual & len
    if not args["MODAL"] == "AO":
        vis_seq_list = pad_sequence([data[0][1] for data in dataBatch], batch_first=True)
        vis_len = torch.tensor([len(data[0][1]) for data in dataBatch])
    else:
        vis_seq_list = None
        vis_len = None

    inputBatch = (aud_seq_list, aud_padding_mask, vis_seq_list, vis_len)

    targetinBatch = pad_sequence([data[1] for data in dataBatch], batch_first=True)
    targetoutBatch = pad_sequence([data[2] for data in dataBatch], batch_first=True)
    targetLenBatch = torch.stack([data[3] for data in dataBatch])

    return inputBatch, targetinBatch, targetoutBatch, targetLenBatch

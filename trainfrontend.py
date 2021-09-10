import logging
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_warmup_lr import WarmupLR

from config import args
from trainFrontend.dataset import LRW
from trainFrontend.datautils import collate_fn
from trainFrontend.general import train, evaluate
from trainFrontend.label_smoothing import SmoothCrossEntropy
from trainFrontend.model import MoCoVisualFrontend
from utils.general import num_params


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='info.log', filemode='w')
    logger = logging.getLogger(__name__)
    # set seed
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    # check device
    torch.cuda.set_device(args["GPU_ID"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # declaring the train and validation datasets and their corresponding dataloaders
    trainData = LRW("train", args["LRW_DATA_DIRECTORY"], args["LRW_HDF5_FILE"], args["WORD_TO_INDEX"], args["STEP_SIZE"], True)
    trainLoader = DataLoader(trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)
    valData = LRW("val", args["LRW_DATA_DIRECTORY"], args["LRW_HDF5_FILE"], args["WORD_TO_INDEX"], args["STEP_SIZE"], False)
    valLoader = DataLoader(valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)

    # declaring the model, optimizer, scheduler and the loss function
    model = MoCoVisualFrontend(args["FRONTEND_DMODEL"], args["WORD_NUM_CLASSES"], args["FRAME_LENGTH"], args['MOCO_FILE'],
                               args["VIDEO_FEATURE_SIZE"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))

    scheduler_reduce = ReduceLROnPlateau(optimizer, mode="max", factor=args["LR_SCHEDULER_FACTOR"], patience=args["LR_SCHEDULER_WAIT"],
                                         threshold=args["LR_SCHEDULER_THRESH"], threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
    scheduler = WarmupLR(scheduler_reduce, init_lr=args["FINAL_LR"], num_warmup=args["LRW_WARMUP_PERIOD"], warmup_strategy='cos')

    loss_function = SmoothCrossEntropy()
    # removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "checkpoints"):
        shutil.rmtree(args["CODE_DIRECTORY"] + "checkpoints")

    os.mkdir(args["CODE_DIRECTORY"] + "checkpoints")
    os.mkdir(args["CODE_DIRECTORY"] + "checkpoints/models")

    # loading the pretrained weights
    if args["TRAIN_LRW_MODEL_FILE"] is not None:
        logger.info("\n\nPre-trained Model File: %s" % (args["TRAIN_LRW_MODEL_FILE"]))
        logger.info("\nLoading the pre-trained model .... \n")
        state_dict = torch.load(args["TRAIN_LRW_MODEL_FILE"], map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        model.to(device)
        print(msg)
        logger.info(msg)
        logger.info("Loading Done.\n")

    train_writer = SummaryWriter('log/train')
    val_writer = SummaryWriter('log/val')
    bestCkpSavePath = args["CODE_DIRECTORY"] + "checkpoints/models/LRWbestCkp.pt"

    # logger.infoing the total and trainable parameters in the model
    numTotalParams, numTrainableParams = num_params(model)
    logger.info("\nNumber of total parameters in the model = %d" % numTotalParams)
    logger.info("Number of trainable parameters in the model = %d\n" % numTrainableParams)
    logger.info("\nTraining the model .... \n")

    # evaluate the model on validation set first
    validationLoss, validationAcc = evaluate(model, valLoader, loss_function, device)
    val_writer.add_scalar("info/loss", validationLoss, -1)
    val_writer.add_scalar("info/acc", validationAcc, -1)
    scheduler.step(validationAcc)
    bestCkp = validationAcc
    val_writer.add_scalar("info/beskCkp", bestCkp, -1)
    val_writer.add_scalar("info/lr", optimizer.param_groups[0]['lr'], -1)

    for step in range(args["NUM_STEPS"]):

        # train the model for one step
        trainingLoss, trainingAcc = train(model, trainLoader, optimizer, loss_function, device)
        train_writer.add_scalar("info/loss", trainingLoss, step)
        train_writer.add_scalar("info/acc", trainingAcc, step)

        # evaluate the model on validation set
        validationLoss, validationAcc = evaluate(model, valLoader, loss_function, device)
        val_writer.add_scalar("info/loss", validationLoss, step)
        val_writer.add_scalar("info/acc", validationAcc, step)

        # logger.infoing the stats after each step
        logger.info("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.Acc: %.3f  Val.Acc: %.3f" % (
            step, trainingLoss, validationLoss, trainingAcc, validationAcc))
        # make a scheduler step
        scheduler.step(validationAcc)

        if validationAcc > bestCkp:
            bestCkp = validationAcc
            torch.save(model.state_dict(), bestCkpSavePath)

        val_writer.add_scalar("info/beskCkp", bestCkp, step)
        val_writer.add_scalar("info/lr", optimizer.param_groups[0]['lr'], step)

        # saving the model weights and loss/metric curves in the checkpoints directory after every few steps
        if ((step % args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"] - 1)) and (step != 0):
            savePath = args["CODE_DIRECTORY"] + "checkpoints/models/train-step_{:04d}-Acc_{:.3f}.pt".format(step, validationAcc)
            torch.save(model.state_dict(), savePath)

    logger.info("\nTraining Done.\n")
    return


if __name__ == '__main__':
    main()

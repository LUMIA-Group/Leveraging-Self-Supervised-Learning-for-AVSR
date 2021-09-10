import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from data.lrs2_dataset import LRS2
from data.utils import collate_fn
from models.av_net import AVNet
from utils.general import inference


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='info.log', filemode='w')
    logger = logging.getLogger(__name__)
    # set seed
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    # check device
    torch.set_num_threads(args["NUM_CPU_CORE"])
    torch.cuda.set_device(args["GPU_ID"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # declaring the test dataset and test dataloader
    noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 1 if args["TEST_WITH_NOISE"] else 0, "noiseSNR": args["TEST_NOISE_SNR_DB"]}
    testData = LRS2(args['MODAL'], "test", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False, noiseParams)
    testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **kwargs)

    if args["EVAL_LRS2_MODEL_FILE"] is not None:
        print("\nTrained Model File: %s" % (args["EVAL_LRS2_MODEL_FILE"]))
        logger.info("\nTrained Model File: %s" % (args["EVAL_LRS2_MODEL_FILE"]))

        # declaring the model,loss function and loading the trained model weights
        modelargs = (args["DMODEL"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"], args["AUDIO_FEATURE_SIZE"],
                     args["VIDEO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["CHAR_NUM_CLASSES"])
        model = AVNet(args['MODAL'], args['WAV2VEC_FILE'], args['MOCO_FRONTEND_FILE'], args["MAIN_REQ_INPUT_LENGTH"], modelargs)
        stateDict = torch.load(args["EVAL_LRS2_MODEL_FILE"], map_location=device)['state_dict']
        msg = model.load_state_dict(stateDict, strict=False)
        print(msg)
        logger.info(msg)
        model.to(device)

        print("\nTesting the trained model .... \n")
        logger.info("\nTesting the trained model .... \n")

        inferenceParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"], "decodeType": args["DECODE_TYPE"],
                           "beamWidth": args["BEAM_WIDTH"], "modal": args["MODAL"], "Lambda": args["LAMBDA"]}

        testCER, testWER, testPER = inference(model, testLoader, device, logger, inferenceParams)

        print("%sMODAL || Test CER: %.3f || Test WER: %.3f" % (args["MODAL"], testCER, testWER))
        logger.info("%sMODAL || Test CER: %.3f || Test WER: %.3f" % (args["MODAL"], testCER, testWER))

        print("\nTesting Done.\n")
        logger.info("\nTesting Done.\n")

    else:
        print("Path to the trained model file not specified.\n")
        logger.info("Path to the trained model file not specified.\n")

    return


if __name__ == "__main__":
    main()

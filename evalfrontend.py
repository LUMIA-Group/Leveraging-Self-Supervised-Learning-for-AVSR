import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from trainFrontend.dataset import LRW
from trainFrontend.datautils import collate_fn
from trainFrontend.general import evaluate
from trainFrontend.label_smoothing import SmoothCrossEntropy
from trainFrontend.model import MoCoVisualFrontend


def main():
    # set seed
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    # check device
    torch.set_num_threads(args["NUM_CPU_CORE"])
    torch.cuda.set_device(args["GPU_ID"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # declaring the train and test datasets and their corresponding dataloaders
    testData = LRW("test", args["LRW_DATA_DIRECTORY"], args["LRW_HDF5_FILE"], args["WORD_TO_INDEX"], args["STEP_SIZE"], False)
    testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)

    if args["EVAL_LRW_MODEL_FILE"] is not None:
        print("\nTrained Model File: %s" % (args["EVAL_LRW_MODEL_FILE"]))

        # declaring the model, optimizer, scheduler and the loss function
        model = MoCoVisualFrontend(args["FRONTEND_DMODEL"], args["WORD_NUM_CLASSES"], args["FRAME_LENGTH"], args['MOCO_FILE'],
                                   args["VIDEO_FEATURE_SIZE"])
        msg = model.load_state_dict(torch.load(args["EVAL_LRW_MODEL_FILE"], map_location="cpu"), strict=False)
        model.to(device)
        loss_function = SmoothCrossEntropy()

        print("\nTesting the trained model .... \n")
        testLoss, testAcc = evaluate(model, testLoader, loss_function, device)
        print("Test Loss: %.6f || Test Acc: %.3f" % (testLoss, testAcc))

    else:
        print("Path to the trained model file not specified.\n")

    print("\nTesting Done.\n")

    return


if __name__ == '__main__':
    main()

import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import prepare_pretrain_input, prepare_main_input
from .vision_transform import ToTensor, Normalize, RandomCrop, CenterCrop, RandomHorizontalFlip


def get_files(datadir, dataset, fold):
    with open(datadir + "/" + dataset + ".txt", "r") as f:
        lines = f.readlines()
    datalist = [datadir + "/" + fold + "/" + line.strip().split(" ")[0] for line in lines]
    return datalist


class LRS2(Dataset):
    """
    A custom dataset class for the LRS2 dataset.
    """

    def __init__(self, modal, dataset, datadir, h5file, charToIx, stepSize, lrs2Aug, noiseParams):
        super(LRS2, self).__init__()
        self.modal = modal

        self.dataset = dataset
        if self.dataset == "train":
            self.datalist = get_files(datadir, 'pretrain', 'pretrain') + get_files(datadir, 'train', 'main')
        elif self.dataset == "val":
            self.datalist = get_files(datadir, 'val', 'main')
        else:
            self.dataset = "test"
            self.datalist = get_files(datadir, 'test', 'main')

        self.h5file = h5file
        with h5py.File(noiseParams["noiseFile"], "r") as f:
            self.noise = f["noise"][0]
        self.noiseSNR = noiseParams["noiseSNR"]
        self.noiseProb = noiseParams["noiseProb"]
        self.charToIx = charToIx
        self.stepSize = stepSize
        if lrs2Aug:
            self.transform = transforms.Compose([
                ToTensor(),
                RandomCrop(112),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.4161], std=[0.1688])
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor(),
                CenterCrop(112),
                Normalize(mean=[0.4161], std=[0.1688])
            ])
        return

    def open_h5(self):
        self.h5 = h5py.File(self.h5file, "r")

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        if self.dataset == "train":
            # index goes from 0 to stepSize-1
            # dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            # fetch the sample at position 'index' in this randomly selected partition
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)
        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        targetFile = self.datalist[index] + ".txt"
        if self.dataset == "val":
            index += 142157
        elif self.dataset == "test":
            index += 143239

        if np.random.choice([True, False], p=[self.noiseProb, 1 - self.noiseProb]):
            noise = self.noise
        else:
            noise = None

        if index < 96318:
            inp, trgtin, trgtout, trgtLen = prepare_pretrain_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise,
                                                                   self.noiseSNR, (3, 21), 160)
        else:
            inp, trgtin, trgtout, trgtLen = prepare_main_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise,
                                                               self.noiseSNR)

        return inp, trgtin, trgtout, trgtLen

    def __len__(self):
        # each iteration covers only a random subset of all the training samples whose size is given by the step size
        # this is done only for the pretrain set, while the whole val/test set is considered
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)

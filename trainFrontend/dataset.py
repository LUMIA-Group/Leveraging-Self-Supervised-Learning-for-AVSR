import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from data.vision_transform import ToTensor, Normalize, RandomCrop, CenterCrop, RandomHorizontalFlip
from .datautils import prepare_input


class LRW(Dataset):
    """
    A custom dataset class for the LRW main (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, h5file, wordToIx, stepSize, lrwaug):
        super(LRW, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [line.strip() for line in lines]
        self.h5file = h5file
        self.dataset = dataset
        self.wordToIx = wordToIx
        self.stepSize = stepSize
        if lrwaug:
            self.transform = transforms.Compose([
                ToTensor(),
                RandomCrop(112),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.413621], std=[0.1700239])
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor(),
                CenterCrop(112),
                Normalize(mean=[0.413621], std=[0.1700239])
            ])
        return

    def open_h5(self):
        self.h5 = h5py.File(self.h5file, "r")

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()
        # using the same procedure as in pretrain dataset class only for the train dataset
        if self.dataset == "train":
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)
        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        target = self.datalist[index].split("/")[-3]
        if self.dataset == "train":
            pass
        elif self.dataset == "val":
            index += 488766
        elif self.dataset == "test":
            index += 513766

        inp, wordTrgt = prepare_input(index, self.h5, target, self.wordToIx, self.transform)
        return inp, wordTrgt

    def __len__(self):
        # using step size only for train dataset and not for val and test datasets because
        # the size of val and test datasets is smaller than step size and we generally want to validate and test
        # on the complete dataset
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)

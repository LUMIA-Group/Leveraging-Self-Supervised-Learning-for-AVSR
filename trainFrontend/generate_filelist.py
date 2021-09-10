import glob
import os

from config import args


def generate_filelist():
    # Generating prtest.txt for splitting the pretrain set into train and validation sets
    print("\n\nGenerating filelist ....")

    train_lines = glob.glob(os.path.join("..", args["LRW_DATA_DIRECTORY"], '*', "train", '*.mp4'))
    val_lines = glob.glob(os.path.join("..", args["LRW_DATA_DIRECTORY"], '*', "val", '*.mp4'))
    test_lines = glob.glob(os.path.join("..", args["LRW_DATA_DIRECTORY"], '*', "test", '*.mp4'))
    train_lines = [line[:-4] + '\n' for line in train_lines]
    val_lines = [line[:-4] + '\n' for line in val_lines]
    test_lines = [line[:-4] + '\n' for line in test_lines]

    with open("../" + args["LRW_DATA_DIRECTORY"] + "/train.txt", "w") as f:
        f.writelines(train_lines)
    with open("../" + args["LRW_DATA_DIRECTORY"] + "/val.txt", "w") as f:
        f.writelines(val_lines)
    with open("../" + args["LRW_DATA_DIRECTORY"] + "/test.txt", "w") as f:
        f.writelines(test_lines)

    print("\nfilelist generated.\n")

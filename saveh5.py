import cv2 as cv
import dlib
import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import args
from preprocess.find_mean import find_mean_face
from preprocess.preprocessing import preprocessing


def get_files(datadir, dataset, fold):
    with open(datadir + "/" + dataset + ".txt", "r") as f:
        lines = f.readlines()
    datalist = [datadir + "/" + fold + "/" + line.strip().split(" ")[0] for line in lines]
    return datalist


def fill_in_data(datalist, flac, png):
    for i, item in enumerate(tqdm(datalist, leave=False, desc="saveh5", ncols=75)):
        inputAudio, sampFreq = sf.read(item + '.flac')
        flac[i] = np.array(inputAudio)
        vidInp = cv.imread(item + '.png')
        vidInp = cv.cvtColor(vidInp, cv.COLOR_BGR2GRAY)
        vidInp = cv.imencode(".png", vidInp)[1].tobytes()
        png[i] = np.frombuffer(vidInp, np.uint8)


def main():
    """
        Preparation for model and filelist
    """
    datadir = args['DATA_DIRECTORY']
    pretrain_datalist = get_files(datadir, 'pretrain', 'pretrain')
    train_datalist = get_files(datadir, 'train', 'main')
    val_datalist = get_files(datadir, 'val', 'main')
    test_datalist = get_files(datadir, 'test', 'main')
    filesList = pretrain_datalist + train_datalist + val_datalist + test_datalist

    landmark_detector = dlib.shape_predictor(args["SHAPE_PREDICTOR_FILE"])
    mean_face_landmarks = find_mean_face(train_datalist, args["PREPROCESSING_NUM_OF_PROCESS"], landmark_detector)
    preprocessing(filesList, args["PREPROCESSING_NUM_OF_PROCESS"], landmark_detector, mean_face_landmarks, True, "lrs")

    """
        Create dataset and Load data
    """

    f = h5py.File(args["HDF5_FILE"], "w")
    dt = h5py.vlen_dtype(np.dtype('float32'))
    flac = f.create_dataset('flac', (len(filesList),), dtype=dt)
    dt = h5py.vlen_dtype(np.dtype('uint8'))
    png = f.create_dataset('png', (len(filesList),), dtype=dt)

    fill_in_data(filesList, flac, png)
    f.close()


if __name__ == "__main__":
    main()

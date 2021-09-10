from multiprocessing import Process, Queue

import cv2 as cv
import dlib
import numpy as np
from tqdm import tqdm


def shape_to_array(shape):
    coords = np.empty((68, 2))
    for i in range(0, 68):
        coords[i][0] = shape.part(i).x
        coords[i][1] = shape.part(i).y
    return coords


def preprocess_sample(file, face_detector, landmark_detector):
    """
    Function to preprocess each data sample.
    """

    videoFile = file + ".mp4"

    # for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    landmark_buffer = list()
    while captureObj.isOpened():
        ret, frame = captureObj.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if not len(frame) == 224:
                frame = cv.resize(frame, (224, 224))
            face_rects = face_detector(frame, 0)  # Detect face
            if len(face_rects) < 1:  # No face detected
                continue
            rect = face_rects[0]  # Proper number of face
            landmark = landmark_detector(frame, rect)
            landmark = shape_to_array(landmark)
            landmark_buffer.append(landmark)
        else:
            break
    captureObj.release()

    return np.array(landmark_buffer).sum(0), len(landmark_buffer)


def preprocess_sample_list(filesList, face_detector, landmark_detector, queue):
    sumed_landmarks = np.zeros((68, 2))
    cnt = 0
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        currsumed, currcnt = preprocess_sample(file, face_detector, landmark_detector)
        sumed_landmarks += currsumed
        cnt += currcnt
    ret = queue.get()
    ret['sumed_landmarks'] += sumed_landmarks
    ret['cnt'] += cnt

    queue.put(ret)


def find_mean_face(filesList, processes, landmark_detector):
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    face_detector = dlib.get_frontal_face_detector()

    # multi processing
    queue = Queue()
    queue.put({'sumed_landmarks': np.zeros((68, 2)), 'cnt': 0, 'missed': 0})

    def splitlist(inlist, chunksize):
        return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]

    filesListSplitted = splitlist(filesList, int((len(filesList) / processes)))

    process_list = []
    for subFilesList in filesListSplitted:
        p = Process(target=preprocess_sample_list, args=(subFilesList, face_detector, landmark_detector, queue))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()

    return_dict = queue.get()
    return return_dict["sumed_landmarks"] / return_dict["cnt"]

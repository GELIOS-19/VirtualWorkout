from utils import *
import cv2
import time
import numpy as np

if __name__ == "__main__":

    webcam = WebcamImage()
    model = load_model()
    
    i = 0

    while True:

        image = webcam.get_image()
        coords, image = get_image_and_coords(image, model)
        cv2.imshow('posenet', image)

        if cv2.waitKey(33) == ord("a"):
            np.save("training_data/squat/{}.npy".format(i), coords)
            cv2.imwrite("training_data/squat/{}.png".format(i), image)
            i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

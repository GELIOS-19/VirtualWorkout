from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import (
    pyqtSignal,
    Qt,
    QThread,
    pyqtSlot
)
from PyQt5.QtWidgets import (
    QWidget, 
    QApplication, 
    QLabel, 
    QVBoxLayout 
) 
import cv2
import numpy as np
from utils import *
import utils

every = 35
current = 0

class Thread(QThread):
    _pyqtSignal: pyqtSignal = pyqtSignal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self.RunFlag = True

    def run(self) -> None:
        global current, every
        print("t")
        capture = cv2.VideoCapture(0)
        print("k")
        capture.set(3, 1280)
        capture.set(4, 720)

        model = load_model()
        
        embedder = BodyPoseEmbedding()
        
        while self.RunFlag:
            if utils.tracker != None and utils.tracker.waiting > 0:
                prev = utils.tracker.waiting
                utils.tracker.waiting -= 1

                if utils.tracker.waiting == 0 and prev > 0:
                    utils.tracker.reset()

            ret, cvImage = capture.read()

            coords, cvImage = get_image_and_coords(cvImage, model)
            embedding = embedder.get_embeddings(coords[0, :, :])

            if ret:
                self._pyqtSignal.emit(cvImage)

            if utils.tracker.progress == 3:
                continue

            avg, thresh, wait = utils.tracker.get_current_avg()
            score = np.linalg.norm(embedding - avg)

            if current <= 0 and score <= thresh:
                current = wait
                utils.tracker.step()
            
            current -= 1

    def stop(self) -> None:
        self.RunFlag = False
        self.wait()


class WebcamWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.w: int = 1280
        self.h: int = 720

        self.imageLabel: QLabel = QLabel(self)
        self.imageLabel.resize(self.w, self.h)

        qVBL: QVBoxLayout = QVBoxLayout()
        qVBL.addWidget(self.imageLabel)
        
        self.setLayout(qVBL)

        self.t = Thread()
        self.t._pyqtSignal.connect(self.updateImage)
        self.t.start()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.t.stop()
        a0.accept()

    @pyqtSlot(np.ndarray)
    def updateImage(self, cvImage) -> None:
        qtImage: QPixmap = self.convertCv2Qt(cvImage)
        self.imageLabel.setPixmap(qtImage)

    def convertCv2Qt(self, cvImage) -> QPixmap:
        rgbCvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbCvImage.shape
        bytesPerLine = ch * w
        cvt2Qt = QtGui.QImage(rgbCvImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        img = cvt2Qt.scaled(self.w, self.h, Qt.KeepAspectRatio)
        return QPixmap.fromImage(img)

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox

import cv2
import numpy as np

from camera_calibration import CameraCalibration


class MainWindow(QMainWindow):

    q1_calibration = CameraCalibration('Q1_Image/')
    q2_calibration = CameraCalibration('Q2_Image/')
    q3_calibration = CameraCalibration('Q3_Image/')
    q4_calibration = CameraCalibration('Q4_Image/')

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('CVDL 2020 HW1')

        main_box_layout = QVBoxLayout()
        main_box_widget = QWidget()
        main_box_widget.setLayout(main_box_layout)

        ### Q1
        q1_box_layout = QVBoxLayout()
        q1_box_widget = QWidget()
        q1_box_widget.setLayout(q1_box_layout)
        q1_label = QLabel('Q1. Calibration')
        q1_label.setAlignment(Qt.AlignLeft)
        q1_box_layout.addWidget(q1_label)

        #### Q1 Question
        q1_questions_layout = QHBoxLayout()
        q1_questions_widget = QWidget()
        q1_questions_widget.setLayout(q1_questions_layout)

        ##### Q1.1 Corner Detection
        self.q1_button_1 = QPushButton('Corner Detection')
        self.q1_button_1.clicked.connect(self.func_q1_corner_detection)
        q1_questions_layout.addWidget(self.q1_button_1)

        ##### Q1.2 Intrinsic Matrix
        self.q1_button_2 = QPushButton('Intrinsic Matrix')
        self.q1_button_2.clicked.connect(self.func_q1_intrinsic)
        q1_questions_layout.addWidget(self.q1_button_2)

        ##### Q1.3 Extrinsic Matrix
        q1_3_box_layout = QVBoxLayout()
        q1_3_box_container = QWidget()
        q1_3_box_container.setLayout(q1_3_box_layout)

        self.q1_combobox_3 = QComboBox()
        for i in range(1, 16):
            self.q1_combobox_3.addItem(str(i)+'.bmp')
        q1_3_box_layout.addWidget(self.q1_combobox_3)

        self.q1_button_3 = QPushButton('Extrinsic Matrix')
        self.q1_button_3.clicked.connect(self.func_q1_extrinsic)
        q1_3_box_layout.addWidget(self.q1_button_3)

        q1_questions_layout.addWidget(q1_3_box_container)

        ##### Q1.4 Distortion Matrix
        self.q1_button_4 = QPushButton('Distortion Matrix')
        self.q1_button_4.clicked.connect(self.func_q1_distortion)
        q1_questions_layout.addWidget(self.q1_button_4)

        q1_box_layout.addWidget(q1_questions_widget)

        ### Q2
        q2_box_layout = QVBoxLayout()
        q2_box_widget = QWidget()
        q2_box_widget.setLayout(q2_box_layout)
        q2_label = QLabel('Q2. Augmented Reality')
        q2_label.setAlignment(Qt.AlignLeft)
        q2_box_layout.addWidget(q2_label)

        #### Q2 Question
        q2_questions_layout = QHBoxLayout()
        q2_questions_widget = QWidget()
        q2_questions_widget.setLayout(q2_questions_layout)

        ##### Q2.1 Augmented Reality
        self.q2_button_1 = QPushButton('Augmented Reality')
        self.q2_button_1.clicked.connect(self.func_q2_augmented)
        q2_questions_layout.addWidget(self.q2_button_1)

        q2_box_layout.addWidget(q2_questions_widget)

        ### Q3
        q3_box_layout = QVBoxLayout()
        q3_box_widget = QWidget()
        q3_box_widget.setLayout(q3_box_layout)
        q3_label = QLabel('Q3. Stereo Disparity Map	')
        q3_label.setAlignment(Qt.AlignLeft)
        q3_box_layout.addWidget(q3_label)

        #### Q3 Question
        q3_questions_layout = QHBoxLayout()
        q3_questions_widget = QWidget()
        q3_questions_widget.setLayout(q3_questions_layout)

        ##### Q3.1 Stereo Disparity Map
        self.q3_button_1 = QPushButton('Stereo Disparity Map')
        self.q3_button_1.clicked.connect(self.func_q3_disparity)
        q3_questions_layout.addWidget(self.q3_button_1)

        q3_box_layout.addWidget(q3_questions_widget)

        ### Q4
        q4_box_layout = QVBoxLayout()
        q4_box_widget = QWidget()
        q4_box_widget.setLayout(q4_box_layout)
        q4_label = QLabel('Q4. SIFT')
        q4_label.setAlignment(Qt.AlignLeft)
        q4_box_layout.addWidget(q4_label)

        #### Q4 Question
        q4_questions_layout = QHBoxLayout()
        q4_questions_widget = QWidget()
        q4_questions_widget.setLayout(q4_questions_layout)

        ##### Q4.1 Keypoints
        self.q4_button_1 = QPushButton('Keypoints')
        self.q4_button_1.clicked.connect(self.func_q4_sift)
        q4_questions_layout.addWidget(self.q4_button_1)

        ##### Q4.2 Match Keypoints
        self.q4_button_2 = QPushButton('Match Keypoints')
        self.q4_button_2.clicked.connect(self.func_q4_match)
        q4_questions_layout.addWidget(self.q4_button_2)

        q4_box_layout.addWidget(q4_questions_widget)

        ### Q5
        q5_box_layout = QVBoxLayout()
        q5_box_widget = QWidget()
        q5_box_widget.setLayout(q5_box_layout)
        q5_label = QLabel('Q5. VGG16 on Cifar')
        q5_label.setAlignment(Qt.AlignLeft)
        q5_box_layout.addWidget(q5_label)

        #### Q5 Question
        q5_questions_layout = QHBoxLayout()
        q5_questions_widget = QWidget()
        q5_questions_widget.setLayout(q5_questions_layout)

        ##### Q5.1 Corner Detection
        self.q5_button_1 = QPushButton('Load Cifar Images')
        q5_questions_layout.addWidget(self.q5_button_1)

        ##### Q5.2 Intrinsic Matrix
        q5_2_box_layout = QVBoxLayout()
        q5_2_box_widget = QWidget()
        q5_2_box_widget.setLayout(q5_2_box_layout)

        q5_label_2 = QLabel('Hyperparameter')
        q5_2_box_layout.addWidget(q5_label_2)

        q5_label_2_epoch = QLabel('Epoch: 50')
        q5_2_box_layout.addWidget(q5_label_2_epoch)
        q5_label_2_optim = QLabel('Optimizer: Adam')
        q5_2_box_layout.addWidget(q5_label_2_optim)
        q5_label_2_batch = QLabel('Batch Size: 32')
        q5_2_box_layout.addWidget(q5_label_2_batch)
        q5_label_2_rate = QLabel('Learning Rate: 0.001')
        q5_2_box_layout.addWidget(q5_label_2_rate)

        q5_questions_layout.addWidget(q5_2_box_widget)

        ##### Q5.3 Extrinsic Matrix
        self.q5_button_3 = QPushButton('Model Structure')
        q5_questions_layout.addWidget(self.q5_button_3)

        ##### Q5.4 Distortion Matrix
        self.q5_button_4 = QPushButton('Training Record')
        q5_questions_layout.addWidget(self.q5_button_4)

        ##### Q5.5 Distortion Matrix
        self.q5_button_5 = QPushButton('Inference')
        q5_questions_layout.addWidget(self.q5_button_5)

        q5_box_layout.addWidget(q5_questions_widget)

        main_box_layout.addWidget(q1_box_widget)
        main_box_layout.addWidget(q2_box_widget)
        main_box_layout.addWidget(q3_box_widget)
        main_box_layout.addWidget(q4_box_widget)
        main_box_layout.addWidget(q5_box_widget)
        self.setCentralWidget(main_box_widget)

    def func_q1_corner_detection(self):
        # calibration = CameraCalibration('Q1_Image/')
        self.q1_calibration.chessboard_corner_detection(True)

    def func_q1_intrinsic(self):
        # calibration = CameraCalibration('Q1_Image/')
        # calibration.chessboard_corner_detection(False)
        print('INTRINSIC MATRIX')
        for index, m in enumerate(self.q1_calibration.mtx):
            if index == 0:
                print('[ %.6f %.6f %.6f ;' % (m[0], m[1], m[2]))
            elif index == 2:
                print('  %.6f %.6f %.6f ] \n' % (m[0], m[1], m[2]))
            else:
                print('  %.6f %.6f %.6f ;' % (m[0], m[1], m[2]))

    def func_q1_extrinsic(self):
        # calibration = CameraCalibration('Q1_Image/')
        # calibration.chessboard_corner_detection(False)

        index = self.q1_combobox_3.currentIndex()
        rvecs = self.q1_calibration.rvecs[index]
        tvecs = self.q1_calibration.tvecs[index]

        rvecs = cv2.Rodrigues(rvecs)[0]
        ext = np.hstack((rvecs, tvecs))
        print('EXTRINSIC MATRIX #', index + 1)
        for index, e in enumerate(ext):
            if index == 0:
                print('[', e[0], e[1], e[2], e[3], ';')
            elif index == 2:
                print(e[0], e[1], e[2], e[3], ']\n')
            else:
                print(e[0], e[1], e[2], e[3], ';')


    def func_q1_distortion(self):
        # calibration = CameraCalibration('Q1_Image/')
        # calibration.chessboard_corner_detection(False)
        print('DISTORTION MATRIX')
        print(self.q1_calibration.dist.flatten())

    def func_q2_augmented(self):
        # calibration = CameraCalibration('Q2_Image/')
        self.q2_calibration.augmented_reality()

    def func_q3_disparity(self):
        # calibration = CameraCalibration('Q3_Image/')
        self.q3_calibration.stereo_disparitymap()

    def func_q4_sift(self):
        self.q4_calibration.sift_keypoint()


    def func_q4_match(self):
        self.q4_calibration.draw_sift_match()



def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()


if __name__ == '__main__':
    main()




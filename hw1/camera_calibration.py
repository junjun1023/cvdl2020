import numpy as np
import cv2
import glob


class CameraCalibration():
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    width = 1000
    height = 800

    def __init__(self, path):
        self.path = path

    def chessboard_corner_detection(self, show=False):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        # objpoints = []  # 3d point in real world space
        # imgpoints = []  # 2d points in image plane.

        images = glob.glob(self.path + '*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                img_resize = cv2.resize(img, (self.width, self.height))

                if show:
                    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    cv2.imshow('img', img_resize)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def _draw(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)

        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)


        # img = cv2.drawContours(img, [imgpts], -1, (0, 0, 255), 5)
        return img

    def augmented_reality(self):
        self.chessboard_corner_detection(False)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        axis = np.float32([[1, 1, 0], [3, 5, 0], [5, 1, 0], [3, 3, -3]]).reshape(-1, 3)

        images = glob.glob(self.path + '*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vectors.
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)

                img = self._draw(img, corners2, imgpts)
                img_resize = cv2.resize(img, (self.width, self.height))

                cv2.imshow('img', img_resize)
                cv2.waitKey(500)
                # k = cv2.waitKey(0) & 0xff
                # if k == 's':
                #     cv2.imwrite(fname[:6] + '.png', img)

        cv2.destroyAllWindows()


c = CameraCalibration('./Q2_Image/')
c.augmented_reality()

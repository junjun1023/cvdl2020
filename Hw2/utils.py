import cv2
import numpy as np
import sklearn

def background_subtraction(path="./Q1_Image/bgSub.mp4"):
    cap = cv2.VideoCapture(path)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    frame_stack = []
    frame_mean = np.array([])
    frame_std = np.array([])
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            # cv2.imshow('video', frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if pos_frame < 50:
                frame_stack.append(frame_gray)
            elif pos_frame == 50:
                frame_stack.append(frame_gray)

                frame_stack = np.stack(frame_stack)

                frame_mean = np.mean(frame_stack, axis=0)
                frame_std = np.std(frame_stack, axis=0)

                # if std less than 5, set 5
                frame_std[frame_std < 5] = 5
            else:
                frame_diff = np.subtract(frame_gray, frame_mean)
                frame_diff = np.abs(frame_diff)
                frame_flag = frame_diff > (frame_std * 5)
                frame_gray[frame_flag] = 255
                frame_gray[np.invert(frame_flag)] = 0
                cv2.imshow('video', frame_gray)
                cv2.waitKey(1000//30)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)


        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break



def optical_flow(path="./Q2_Image/opticalFlow.mp4"):
    cap = cv2.VideoCapture(path)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            # cv2.imshow('video', frame)
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # Setup SimpleBlobDetector parameters.
            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 50
            params.maxThreshold = 200

            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.85

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)

            detector.empty()
            keypoints = detector.detect(frame)


            for keypoint in keypoints:
                x = keypoint.pt[0]
                y = keypoint.pt[1]
                top_left = [x-5.5, y-5.5]
                top_right = [x+5.5, y-5.5]
                bottom_left = [x-5.5, y+5.5]
                bottom_right = [x+5.5, y+5.5]
                x_top = (int(x), int(y-5.5))
                x_bottom = (int(x), int(y+5.5))
                y_left = (int(x-5.5), int(y))
                y_right = (int(x+5.5), int(y))
                pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 0, 255), 1)
                cv2.line(frame, x_top, x_bottom, (0, 0, 255), 1)
                cv2.line(frame, y_left, y_right, (0, 0, 255), 1)

            cv2.imshow("Keypoints", frame)
            cv2.waitKey(50)

        else:
            # The next frame is not ready, so we try to read it again
            # cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            # It is better to wait for a while for the next frame to be ready
            # cv2.waitKey(1000)
            break

        # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            # break


def video_tracking(path="./Q2_Image/opticalFlow.mp4"):
    cap = cv2.VideoCapture(path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = []
    prev_frame = None
    mask = None
    while True:
        flag, frame = cap.read()
        if flag:

            if prev_frame is None:

                # Setup SimpleBlobDetector parameters.
                params = cv2.SimpleBlobDetector_Params()

                # Change thresholds
                params.minThreshold = 50
                params.maxThreshold = 200

                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = 0.85

                # Create a detector with the parameters
                detector = cv2.SimpleBlobDetector_create(params)

                detector.empty()
                keypoints = detector.detect(frame)

                for keypoint in keypoints:
                    x = keypoint.pt[0]
                    y = keypoint.pt[1]
                    p0.append([x, y])
                p0 = np.array(p0, dtype='float32')

                p0 = p0[:, np.newaxis, :]


                prev_frame = frame
                mask = np.zeros_like(frame)
                continue

            else:
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk_params)
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
                    _frame = cv2.circle(prev_frame, (a, b), 5, (0, 0, 255), -1)
                img = cv2.add(_frame, mask)
                cv2.imshow('frame', img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                prev_frame = frame
                p0 = p1

        else:
            # The next frame is not ready, so we try to read it again
            # cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            # It is better to wait for a while for the next frame to be ready
            # cv2.waitKey(1000)
            break

        # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            # break



def perspective_transform(path="./Q3_Image/test4perspective.mp4", img_path="./Q3_Image/rl.jpg"):
    cap = cv2.VideoCapture(path)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    im_src = cv2.imread(img_path)

    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            # cv2.imshow('video', frame)
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # Load the dictionary that was used to generate the markers
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

            # Initialize the detector parameters using default values
            parameters = cv2.aruco.DetectorParameters_create()

            # Detect he markers in the image
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

            if len(markerIds) < 4:
                continue


            ### Id = 25
            index = np.squeeze(np.where(markerIds == 25))
            refPt1 = np.squeeze(markerCorners[index[0]])[1]

            ### Id = 33
            index = np.squeeze(np.where(markerIds == 33))
            refPt2 = np.squeeze(markerCorners[index[0]])[2]

            distance = np.linalg.norm(refPt1-refPt2)

            scalingFac = 0.02

            pts_dst = [
                [refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]
            ]
            pts_dst = pts_dst + [
                [refPt2[0] + round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]
            ]


            ### Id = 30
            index = np.squeeze(np.where(markerIds == 30))
            refPt3 = np.squeeze(markerCorners[index[0]])[0]
            pts_dst = pts_dst + [
                [refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)]
            ]

            ### Id = 23
            index = np.squeeze(np.where(markerIds == 23))
            refPt4 = np.squeeze(markerCorners[index[0]])[0]
            pts_dst = pts_dst + [
                [refPt4[0] - round(scalingFac*distance), refPt4[1] + round(scalingFac*distance)]
            ]

            pts_src = [
                [0, 0],
                [im_src.shape[1], 0],
                [im_src.shape[1], im_src.shape[0]],
                [0, im_src.shape[0]]
            ]

            retval, mask = cv2.findHomography(np.array(pts_src, dtype='float32'), np.array(pts_dst, dtype='float32'))
            dst = cv2.warpPerspective(im_src, retval, dsize=(frame.shape[1], frame.shape[0]))

            img = cv2.add(frame, dst)
            cv2.imshow("Keypoints", img)
            cv2.waitKey(1000//30)

        else:
            # The next frame is not ready, so we try to read it again
            # cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            # It is better to wait for a while for the next frame to be ready
            # cv2.waitKey(1000)
            break



def pca_reconstruction(dir_path="./Q4_Image/"):
    # https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
    from sklearn.decomposition import PCA
    import os
    import pandas as pd

    dirs = os.listdir(path=dir_path)
    imgs = pd.DataFrame([])
    _shape = []

    for file in dirs:
        img = cv2.imread(dir_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
        img = img / 255

        imgs.append( pd.Series(img.flatten(), name=file) )
        _shape.append(img.shape)
        # cv2.imshow('img', img)
        # cv2.waitKey(50)

    # plt
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 10, figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []}, gridspec_kw = dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs.iloc[i].values.reshape(100, 100), cmap="gray")

    imgs = np.array(imgs)
    pca = PCA(n_components=2)
    pca.fit_transform(imgs)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    print(len(pca.components_))
    cv2.imshow('eigen vector', pca.components_[1].reshape(_shape[1][0], _shape[1][1]) )
    cv2.waitKey()




# background_subtraction()
# optical_flow()
# video_tracking()
# perspective_transform()
pca_reconstruction()
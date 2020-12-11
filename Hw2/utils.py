import cv2
import numpy as np

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
                print(p0.shape)
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

                # Now update the previous frame and previous points
                # p0 = good_new.reshape(-1, 1, 2)
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

# background_subtraction()
video_tracking()
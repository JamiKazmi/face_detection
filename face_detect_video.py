# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2 as cv

# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe(
    'files/deploy.prototxt.txt',
    'files/res10_300x300_ssd_iter_140000.caffemodel'
)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # draw the bounding box of the face along with the associated
        # probability
        text = '{:.2f}%'.format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv.rectangle(
            frame, (startX, startY),
            (endX, endY), (0, 0, 255), 2
        )
        cv.putText(
            frame, text,
            (startX, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2
        )
    # show the output frame
    cv.imshow('Frame', frame)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv.destroyAllWindows()
vs.stop()

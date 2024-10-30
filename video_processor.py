# video_processor.py
import cv2
import time
import numpy as np
import logging
from utils import preprocess_frame


def make_predictions(net, classes, frame, confidence_threshold=0.5):
    """Makes predictions on the frame and overlays the top classes with probabilities."""
    blob = preprocess_frame(frame)
    net.setInput(blob)

    outp = net.forward()

    for r, i in enumerate(np.argsort(outp[0])[::-1][:5], start=1):
        confidence = outp[0][i]
        if confidence > confidence_threshold:
            txt = f'{classes[i]}: {confidence * 100:.2f}%'
            cv2.putText(frame, txt, (10, 25 + 40 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def process_video(video_path, net, classes, settings):
    """Processes a video file, applying the model on every 5th frame for efficiency."""
    process_every_n_frames = settings.get("process_every_n_frames", 5)
    confidence_threshold = settings.get("confidence_threshold", 0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Cannot open video stream")
        exit()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_count % process_every_n_frames == 0:
            make_predictions(net, classes, frame, confidence_threshold)

        cv2.imshow('Processed Frame', frame)
        frame_count += process_every_n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if cv2.waitKey(25) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

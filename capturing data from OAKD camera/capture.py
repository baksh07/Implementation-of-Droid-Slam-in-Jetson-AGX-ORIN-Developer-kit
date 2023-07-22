import sys
import numpy as np
import cv2
import os
import glob 
import time
import depthai as dai

if __name__ == '__main__':

    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setPreviewSize(300,300)
    camRgb.initialControl.setManualFocus(128)
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")

    camRgb.video.link(xoutRgb.input)
    
    device = dai.Device(pipeline)

    tstamps = []

    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    output_directory = 'frames'
    os.makedirs(output_directory, exist_ok=True)
    frame_counter = 0

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            image = inRgb.getCvFrame()
        else:
            continue

        print(image.shape)
        window_name = "image"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image / 255.0)

        # Save the frame as an image
        output_path = os.path.join(output_directory, f'{str(frame_counter).zfill(3)}.png')
        cv2.imwrite(output_path, image)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

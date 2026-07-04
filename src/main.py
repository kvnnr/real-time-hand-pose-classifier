#Camera Module
from camera.camera import Camera

import cv2 as cv


def main():

    #Create the object.
    cam = Camera()

    #Initialize the camera.
    cam.open_camera()

    while True:

        """
        Why:
            A continuous loop is used to fetch frames from 
            the video stream in real time. The loop runs 
            until an exit condition is met, 
            allowing controlled termination of the capture process.
        """

        frame = cam.get_frames() #Grab each frame
        rgb_frames = cam.bgr_to_rgb(frame) #Convert BGR to RGB

        #Show the window Screen
        cv.imshow("Live Stream", frame)

        #Initiate for exit key
        key = cv.waitKey(1) & 0xFFq
        if key == ord('q'):
            break
        
    cam.release_camera()

main()
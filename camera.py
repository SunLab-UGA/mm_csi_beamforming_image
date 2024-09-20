# test taking a picture with the webcam and saving it to a file with opencv

# taking a picture from cold start takes about 2.6 seconds!

import cv2
import time

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.resolution = (1920, 1080)
        self.cap.set(3, self.resolution[0])
        self.cap.set(4, self.resolution[1])
        self.first_frame = True
    
    def take_picture(self, filename):
        '''take a picture with the webcam and save it to a file
        the resolution of the picture is set by the resolution parameter'''
        # flush the frames to get the most recent picture
        if self.first_frame == False: # we don't need to flush the first frame
            self.flush_frames()
            print("flushed frames")
        self.first_frame = False
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(filename, frame)
            print(f"picture saved to {filename}")
        else:
            print("failed to take picture")

    def flush_frames(self, num_frames=10):
        '''Read and discard a number of frames to clear the buffer.'''
        for _ in range(num_frames):
            self.cap.read()

    def release(self):
        self.cap.release()

def take_test_picture(filename):
    '''take a picture with the webcam and save it to a file
    the resolution of the picture is the same as the webcam's default resolution'''
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"picture saved to {filename}")
    else:
        print("failed to take picture")
    cap.release()

def take_picture(filename, resolution:tuple=(1920, 1080)):
    '''take a picture with the webcam and save it to a file
    the resolution of the picture is set by the resolution parameter'''
    # HD resolution is (1280, 720)
    # Full HD resolution is (1920, 1080)
    cap = cv2.VideoCapture(0)
    cap.set(3, resolution[0])
    cap.set(4, resolution[1])
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"picture saved to {filename}")
    else:
        print("failed to take picture")
    cap.release()

if __name__ == "__main__":
    print("Welcome!")
    # print("taking test picture")
    # start_time = time.time()
    # take_picture("test.jpg", resolution=(1920, 1080))
    # picture_time = time.time() - start_time
    # print(f"picture_time: {picture_time}")
    # print("done")

    # test the camera class
    print("testing camera class")
    cam = Camera()
    start_time = time.time()
    cam.take_picture("test2.jpg")
    picture_time = time.time() - start_time
    print(f"picture_time: {picture_time}")
    print("waiting for 5 seconds")
    dialog = ['ready', 5, 4, 3, 2, 1, 'smile! ;D']
    for d in dialog:
        print(d)
        time.sleep(1)
    start_time = time.time()
    cam.take_picture("test3.jpg")
    picture_time = time.time() - start_time
    print(f"picture_time: {picture_time}")
    cam.release()
    print("camera released")
    print("Test complete :D")
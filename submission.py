from imageai.Detection import ObjectDetection
from PIL import Image
from PIL import ImageFilter
import numpy as np
import os
import os.path
from os import path
import sys
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import subprocess
import multiprocessing
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Task 1:
def fetch_and_extract(data):
    subprocess.call(["./script.sh", data])

def grab_timestamps(start, end, index="index.txt"):
    to_do = []

    # grabbing all of the timestamps and creating a list of images to download
    for img in range(start,end+1):
        name = str(img) + ".ts"
        with open(index) as f:
            if name in f.read():
                filename = "./first_frame/" + str(img) + ".jpg"
                if not path.exists(filename):
                    to_do.append(str(img))
    return to_do

def concurent_download(to_do):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(fetch_and_extract, to_do)


## Task 2:
# loads the RetinaNet model for detection 
def load_model():
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    return detector

# used to get information about the specifc parking spot
def crop_img(img):
    im = Image.open(img)
    
    left = 185
    top = 190
    right = 275
    bottom = 252

    im1 = np.array(im.crop((left, top, right, bottom)))
    return im1

# runs the loaded model to determine if there is a car in the picture
def has_car(img, detector):
    im1 = crop_img(img)
    img_name = img.split('/')[2]
    returned_image, detections = detector.detectObjectsFromImage(input_type="array", input_image=im1, output_type="array", minimum_percentage_probability=20)
    found = False
    for objects in detections:
        if objects["name"] == "car":
            found = True
            break
    if found:
        return True
    else:
        return False


## Task 3:
# uses SIFT and the Lowe's ratio test to determine if it is the same car
def is_same_car(img1, img2):
    img_name1 = img1.split('/')[2]
    img_name2 = img2.split('/')[2]
    
    region1 = crop_img(img1)
    region2 = crop_img(img2)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(region1,None)
    kp2, des2 = sift.detectAndCompute(region2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    number_keypoints = 0
    if len(kp1) <= len(kp2):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)
    
    how_similar = len(good) / number_keypoints * 100
    
    if how_similar > 5:
        return True
    else:
        return False


# Task 4:
def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # filling pixels inside the polygon defined by "vertices" with the fill color
    for vert in vertices:
        cv2.fillPoly(mask, vert, ignore_mask_color)
        cv2.fillPoly(mask, vert, ignore_mask_color)
    
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_parking_spots(img):
    image = cv2.imread(img)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width = int(img.shape[1] / 100)
    height = int(img.shape[0]  / 100)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.add(gray,np.array([75.0]))
    
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(blur_gray, (kernel_size, kernel_size), 0)

    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 18  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    
    # Mask to get Region of Interest    
    vertices1 = np.array([[[(0, 235), 
                           (235, 235), 
                           (235, 275), 
                           (0, 275)
                          ]]], dtype=np.int32)
    vertices2 = np.array([[[(615, 225), 
                           (width, 225), 
                           (width, 260), 
                           (615, 260)
                          ]]], dtype=np.int32)
    vertices = [vertices1, vertices2]
    region = region_of_interest(edges, vertices)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(region, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            theta = math.degrees(math.atan2(y2-y1, x2-x1))
            if theta > 20 and theta < 40:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    return lines_edges


## Task 5:
def analyze_cars(detector, index, start, end, concurrent, speed):
    print("analysing from " + str(start) + " to " + str(end))
    first_seen = []
    last_seen = []
    first_car = 0
    first_filename = ""

    to_do = grab_timestamps(start, end, index)

    # downloading the images concurrently or one by one
    if concurrent:
        if to_do:
            concurent_download(to_do)
    else:
        for img in to_do:
            fetch_and_extract(str(img))

    prev = 0
    speed_const = speed
    curr_speed = speed

    # keeping track of when the car comes into view and when it leaves
    for img in tqdm(range(start, end+1)):
        filename = "./first_frame/" + str(img) + ".jpg"
        if not path.exists(filename):
            continue
        if curr_speed != 0:
            curr_speed -= 1
            continue
        curr_speed = speed_const
        check = has_car(filename, detector)
        if check and first_car == 0:
            first_car = img
            first_filename = "./first_frame/" + str(first_car) + ".jpg"
            first_seen.append(first_car)
        if not check and first_car != 0:
            last_seen.append(prev)
            first_car = 0
            first_filename = ""
        prev = img
    
    last_seen.append(prev)
    
    # used to merge intverals with a false negative
    merged = []
    for first,last in zip(first_seen, last_seen):
        if not merged or merged[-1][1] + 90 < first:
            merged.append([first, last])
        else:
            merged[-1] = [merged[-1][0], max(merged[-1][1],last)]
    
    # output the results
    for first_car,last_car in merged:
        first_filename = "./first_frame/" + str(first_car) + ".jpg"
        elapsed = (last_car - first_car) // 60
        print("found car at " + str(first_car) + ". parked until " + str(last_car) + " (" + str(elapsed) + " minutes).")
        new_file = str(first_car) + "-" + str(elapsed) + "min.jpg"
        shutil.copyfile(first_filename, "./output/" + new_file)
        print("... wrote output/" + new_file)

def main():
    args = sys.argv[1:]
    detector = None
    if args[0] == "has-car" or args[0] == "analyze-cars":
        detector = load_model()
    if args[0] == "fetch-and-extract":
        if len(args) == 2:
            fetch_and_extract(args[1])
        to_do = []
        if len(args) > 2 and args[2]:
            start = args[1]
            end = args[2]
            to_do = grab_timestamps(int(start), int(end))
        if len(args) > 3 and args[3] == "--concurrency":
            if to_do:
                concurent_download(to_do)
        else:
            for img in to_do:
                fetch_and_extract(str(img))

    if args[0] == "has-car":
        filename = "./first_frame/" + str(args[1])
        print("running object detector on " + str(args[1]) + "... ",  end="")
        out = has_car(filename, detector)
        if out:
            print("car detected!")
        else:
            print("no car detected!")

    if args[0] == "is-same-car":
        filename1 = "./first_frame/" + str(args[1])
        filename2 = "./first_frame/" + str(args[2])
        print("comparing " + str(args[1]) + " and " + str(args[2]) + "...", end="")
        out = is_same_car(filename1, filename2)
        if out:
            print(" same car!")
        else:
            print(" different car!")

    if args[0] == "draw-parking-spots":
        filename = "./first_frame/" + str(args[1])
        new_img = draw_parking_spots(filename)
        plt.imshow(new_img)
        plt.show()

    if args[0] == "analyze-cars":
        index = args[1]
        start = args[2]
        end = args[3]
        speed = 0
        if len(args) > 4 and args[4] == "normal":
            speed = 0
        if len(args) > 4 and args[4] == "fast":
            speed = 2
        if len(args) > 4 and args[4] == "superfast":
            speed = 4
        concurrent = False
        if len(args) > 5 and args[5] == "--concurrency":
            concurrent = True
        new_img = analyze_cars(detector, index, int(start), int(end), concurrent, speed)
        print("no more cars found!")
main()





















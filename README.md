# ParkingSpotDetector

Need to run to get the RetinaNet model:
curl -LOk https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

Dependencies:
- brew install ffmpeg
- pip3 install imageai --upgrade
- pip3 install keras==2.2.4
- conda install -c menpo opencv -y
- matplotlib
- numpy
- tensorflow
- PIL
- tqdm

## Tasks
### Task 1:
- I created a simple bash script that takes in the input of the desired timestamp, will download the video and extract the first frame from the image and place them into seperate folders.

- To Run:
    python3 submission.py fetch-and-extract [timestamp1] [timestamp2] [--concurrency]
    
- Argument Notes: 
    {OPTIONAL}[timestamp2]
    {OPTIONAL}[--concurrency] acceptable values are: "", "--concurrency"
    
- For Example:
    python3 submission.py fetch-and-extract 1538076003
    python3 submission.py fetch-and-extract 1538076179 1538076191 --concurrency
    
    
### Task 2:
- I used the ImageAI package and the RetinaNet pretrained model to detect whether or not there is a car in the specified parking spot. Cropped the image and then ran the detection model on the cropped parking spot.

- To Run:
    python3 submission.py has-car [image1]
    
- For Example:
    python3 submission.py has-car 1538076003.jpg


### Task 3:
- Used SIFT as a feature detector to compare two images. Then checks for good matches between the two images to determine if they are the same car.

- To Run:
    python3 submission.py fetch-and-extract [image1] [image2]
    
- For Example:
    python3 submission.py is-same-car 1538076179.jpg 1538076183.jpg
    
    
### Task 4:
- Created a pipeline to find the parking spot lines from the given picture. Does an ok job, but the main steps were:
    - Convert image to grayscale
    - Smooth the image with a Gaussian Blur
    - Use Canny edge detection to determine all of the edges
    - Narrow the region of interest to the parking spots
    - Finally, use a Hough Transform to connect the edges into lines

- To Run:
    python3 submission.py draw-parking-spots [image1]
    
- For Example:
    python3 submission.py draw-parking-spots 1538076179.jpg
    

### Task 5:
- Kept track of when a car came into frame and when it was out of frame by simply checking whether the current spot had a car. Was able to then use a mergeing technique to get rid of the false negatives and the false positive with a reasonable degree of certainty. Simply done to ensure that any missteps with the object detection model would not lead to loss of data. Finally, took the merged intervals and outputted the results into the output folder

- To Run:
    python3 submission.py analyze-cars [index] [timestamp1] [timestamp2] [speed] [--concurrency]

- Argument Notes: 
    [speed] - acceptable values are: "normal", "fast", "superfast"
    {OPTIONAL} [--concurrency] acceptable values are: "", "--concurrency"
    
- For Example:
    python3 submission.py analyze-cars index.txt 1538076003 1538078234 normal --concurrency 
    
    
### Questions:
When do you think this algorithm would work well? when would it not?
- This algorithm works pretty well in the day time and there are not a lot of objects obsuring the car. When a trunk is open or a person is standing in the frame, the detector's certainty in whether or not it is a car drops. The night time also makes it harder to get information about the object from the picture so it would not work as optimally there as well.

What would you suggest for future exploration?
- For future exploration, I would like to take a look at the process of automatically detecting cars in a certain area and how to create a generalizable format for identifying when parking spots are occupied and be able to associate each filled parking spot with a tag for security purposes.


### Bonus:
[X] Download multiple images in parallel (add a --concurrency flag)
[-] Analyze the other nearby parking spot occupancies as well.
[X] Are there other algorithms that work better at identifying cars?

Yes for sure, I personally used the Cascade R-CNN model when trying to detect cars for the Waymo Open Dataset Challenge. It adapts to the problem of requiring a high IOU to be determined as a car by using cascade regression as a resampling mechanism. This cascade learning has three important consequences for detector training. First, the potential for overfitting at large IoU thresholds u is reduced, since positive examples become plentiful at all stages. Second, detectors of deeper stages are optimal for higher IoU thresholds. Third, because some outliers are removed as the IoU threshold increases, the learning effectiveness of bounding box regression increases in the later stages. Resampling progressively improves hypothesis quality, guaranteeing a positive training set of equivalent size for all detectors and minimizing overfitting. The same cascade is applied at inference, to eliminate quality mismatches between hypotheses and detectors. 

[X] Can you detect the color of the car?

Yes, it would require converting to greyscale, filtering for the car object, isolate teh car body and convert to HSV to deterine the color of the car. This would work with varying accuracy as filtering for the car object would be the most difficult part.

[X] Will the method work for both day and night time images?

Yes, but with less success in the night time because there is less information for the detection model to be able to create good educated guesses about whether or not a car exists.

[-] Vehicle counting (parked vs. moving vehicles / thumbnail)
[X] Additional analysis of the scene apart from parking lot occupancy (impress us! :) )

Noticed that there was a of of pedestrain traffic. It might be pertinant, especially as a security company, to start tracking individuals who idle in a location for too long. Also, if there are any figures that are running, a quick detection of that individual will aid in tracking potential thefts. Another important observation is if there is an individual that idles under a camera for too long, if they are trying to determine the FOV of the camera, there could be nefarious purposes involved. Another thing to look out for is aggressive drivers. If cars are fighting for a parking spot or causing other problems, it is pertinant to keep track of the movements of these drivers. 



    

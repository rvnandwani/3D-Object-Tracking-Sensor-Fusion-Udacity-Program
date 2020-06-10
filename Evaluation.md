### Match Bounding Boxes

This is acheived mainly by making a 2d vector of size (number of bounding box in current frame , number of bounding box in previous frame. This matrix is updated based on the fact that how many keypoint matches are there between corresponding bounding box.between 2 frames. So the bounding box with most number of keypoint matches in the bounding box of other frame are inserted in a map, hence making a map of corresponding matches of bounding boxes.

### Compute LIDAR TTC

The basic formula used for TTC calculation is the one that was implemented earlier i.e.

ttc = (curr_d * t) / (prev_d - curr_d)

To avoid disambiguity caused by the ambiguous lidar point, for denominator I calculated the average value of x from previous frame and current frame and used that difference.

For numerator I checked for minimum value of x and had a threhold that it must be atleast greater than 80% of the average of all the x values of that frame.

### Keypoints cluster based on ROI

The main task in this function was to fill the 2 vectors boundingBox.keypoints and boundingBox.kptMatches and in addition to this also remove the disambiguity in matches to have a robust estimation, Earlier I thought of implementing advanced vision concept like epipolar geometry and implement RANSAC to have robust estimation, but that will lead to increase in the processing time and is also not part of this course, so used the concept of thresholding as did in previous step i.e. if the distance between two point correspondences in their respective frame cannot be more than 1.3 times the avg distance between correspondences. 

If a point stasfies the above condition then that along with its match is push_back to the vectors boundingBox.keypoints and boundingBox.kptMatches respectively.

### Compute Camera TTC

Here the implementation is same as that done in the assignment earlier. Stored the distance from outer to inner keypoint by calculating distance ratio of one keypoint with every other keypoint in the ROI.

To have a robust estimation, calculated the TTC using the median of sorted vector of this distance ratio. 
TTC = -t / (1- dist_ratio)

### LIDAR TTC Evaluation

In lidar TTC calculationn I see generally there is a dectrease in TTC except the second last reading, I feel it could be because I am considering 2 frames at a time, so as the car is approaching the car in front, would be braking to avoid collision, that could be a factor inincrease of the TTC. But compared to camera I feel LIDAR calculation for TTC is much accurate.

### Camera TTC Evaluation

The various camera ttc readings, for the detector/descriptor pair that I found most efficient using mid-term project is shown in the attached excel.
I found the camera based evaluation is not as same as that of lidar at any point, maybe we have to normalize the coordinates for the same, but the trend is same, i.e. it is decreasing overall, with certain readings, where it will increase.
out of detector descriptor pairs I tested ORB detector and Brief descriptor performance was very poor as many of the ttc outputs were extremely high sometimes and I found the earlier proposed detector descriptor pair i.e. FAST detector and BRISK descriptor most efficient and relieable.
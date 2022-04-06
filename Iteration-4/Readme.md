# LESCO-Iteration-4

Past iterations (1,2 &3) have something in common; all frames from a video are used to create a 1-dimensional representation of a size [42,1]. In this iteration, we are taking another strategy. Instead of mixing all frames from a video to create a single entry, I am selecting 3-5 frames from each video where hands represent a particular sign. This makes for example, 3 images that are labeled as "Grandfather". For each image, 6 new images are created using augmentation (with OpenCV) by adding rotations, flips and so on to increase the number of training samples.
The central hypothesis here is: that few frames from a video are sufficient to identify a particular sign without considering order. The idea is to create a more flexible classifier that uses lower dimensional structures without the need to mix all frames from a particular video.

## Approach

- Extract important frames manually from each video and save them into the "single-frame" folder. 
- Load each image from the folder and augment it by applying rotation and flipping. 
- Convert each image into a 1-dimensional array and label each array with its corresponding class.
- Create a test-train split to test the Manhattan Distance similarity measure.
- Use Cross-Validation to get a more accurate result.

## Results:

- Cross Validation Accurary on the test set: 80%

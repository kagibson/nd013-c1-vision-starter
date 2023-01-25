# Object Detection in an Urban Environment Project Submission
## Author: Kurtis Gibson

### Project Overview

### Set Up

### Dataset

#### Exploratory Data Analysis

For each of the following analyses, I chose to shuffle the dataset and then take 1000 images from the randomized dataset.
I made the assumption that the 1000 randomized images from the dataset would provide a good representation for the entire dataset.

The first analysis I performed on the data was determining the frequency of classes in the image batch:

![Class Frequency](../images/frequency_of_classes.png)

From this analysis it is apparent that automobiles are most represented and cyclists are least represented.
Depending on the size of the entire dataset, this could mean that we need to gather more images with cyclists in order to best train the model to detect cyclists.

The next analysis I performed was calculating a histogram for the pixel values of each image in the batch per channel (RGB):

![Pixel Distribution By Channel](../images/pixel_distribution_by_channel.png)

One thing I note in this analysis is that there seems to be some clipping on all 3 channels as you can see a spike around 255 for each channel.

Another analysis I performed was calculating the mean and standard deviation of pixel values for each image in the batch and then plotting these values in an scatter plot of mean vs standard deviation:

![Mean vs Std Dev of Pixel values](../images/mean_vs_stddev.png)

From this plot it seems that there is a smaller amount of images with a small mean and standard deviation.
These images are very dark and have very few lighter pixels.
We may need to augment the dataset to include more images like this.

The last analysis I did was calculating the areas of all bounding boxes for the sample dataset. There was a small amount of very large bounding boxes that were biasing the distribution so I removed them (everything above 10000px) and plotted them separately:

![Distribution of bounding box areas](../images/bounding_box_areas.png)

From this plot it seems there is a very large number of small bounding boxes (under 1000px).
Depending on the object detection model type (such as YOLO), these small objects could prove difficult to detect. I also noticed a few bounding boxes that were the same size of the image (409600px).
On inspection these boxes seemed to be erroneous and could be removed from the dataset.

From my qualitative and quantitative analysis, the following seems to be true about the dataset:

- A little less than 80% of the annotated classes are automobiles, about 20% are pedestrians and about 1-2% are cyclists.
- There is a small amount of nighttime images, perhaps 10%.
- A large majority of the bounding boxes have an area of less than 1000px. That is less than 0.25% of the image area (1000/409600)

#### Cross-validation Approach

### Training

#### Reference Experiment

#### Reference Experiment Improvement



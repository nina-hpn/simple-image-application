# About dataset

## Context
Name: Image colorization.\
Author: SHRAVANKUMAR SHETTY.\
Gray scale images are taken as input and a and b components of LAB color space are taken as output.

## Content
The dataset consists of two compressed zip files:
(1) ab.zip : This contains 25 .npy files consisting of a and b dimensions of LAB color space images, 
of the MIRFLICKR25k randomly sized colored image dataset. The LAB color space generally takes up large disk spaces, hence is a lot slower to load. That is the reason, I divided this into 25 files, so that it can be loaded at the time of requirement.

(2) l.zip : This consists of a gray_scale.npy file which is the grayscale version of the MIRFLICKR25k dataset.

The image dataset which I used was taken from the MIRFLICKR25k.
# cv-lab-1
Computer Vision Lab: image filtering (Gauss, Bilateral)

## Task

1. Find a face in the image.
![source](docs/rectangles.png)

<br>

2. We retreat 10% from the borders of the face and get this fragment of the image. Further work only with this fragment.
![source](docs/face.png)

<br>

3. Get a binary image of edges (object boundaries).
![source](docs/contours.png)

<br>

4. Remove small borders with length and width less than 10.
![source](docs/erase-countours.png)

<br>

5. Apply the operation of morphological growth (the size of the structuring element is 5 x 5).
![source](docs/dilation.png)

<br>

6. Smoothen the resulting edge image using a 5 by 5 Gaussian filter. Get a normalized image M where all pixels have values ​​between 0 and 1.
![source](docs/gaussian-normal.png)

<br>

7. Obtain an F1 image of the face using bilateral filtering.
![source](docs/bilateral.png)

<br>

8. Acquire an F2 face image with improved clarity/contrast.
![source](docs/sharp.png)

<br>

9. Perform the final filtering according to the formula:
![source](docs/result.png)

## Source image

![source](docs/image.png)

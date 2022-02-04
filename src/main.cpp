#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void scaleFaces(vector<cv::Rect> &);
cv::Mat filtration(cv::Mat &faceImage, cv::Mat &bilateral, cv::Mat &sharp, cv::Mat &normalised);

void main()
{
    string imgPath = "../assets/lena.png";
    string cascadePath = "../assets/haarcascade_frontalface_alt2.xml";

    cv::Mat srcImage = cv::imread(imgPath);
    cv::imshow("Image", srcImage);

    cv::CascadeClassifier faceCascade;
    faceCascade.load(cascadePath);

    // Detect faces
    vector<cv::Rect> faces;
    faceCascade.detectMultiScale(srcImage, faces, 1.1, 4);
    // Scale
    scaleFaces(faces);

    // Rectangles
    cv::Mat rectangles = srcImage.clone();
    for (auto &face : faces)
    {
        cv::rectangle(rectangles, face.tl(), face.br(), cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("Rectangles", rectangles);

    // Face
    cv::Mat faceImg = srcImage(faces[0]);
    cv::imshow("Face", faceImg);

    // Find contours
    vector<vector<cv::Point>> contPoints;
    cv::Mat contours = cv::Mat::zeros(faceImg.size(), CV_8UC3);
    cv::Mat canny;

    cv::Canny(faceImg, canny, 100, 200);
    cv::findContours(canny, contPoints, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(contours, contPoints, -1, cv::Scalar(255, 255, 255), 1);

    cv::imshow("Canny", canny);
    cv::imshow("Contours", contours);

    // Erase contours
    cv::Mat eraseContours = cv::Mat::zeros(faceImg.size(), CV_8UC3);

    contPoints.erase(remove_if(contPoints.begin(), contPoints.end(),
                               [](vector<cv::Point> const &x)
                               {
                                   return cv::arcLength(x, false) <= 10;
                               }),
                     contPoints.end());
    cv::drawContours(eraseContours, contPoints, -1, cv::Scalar(255, 255, 255), 1);

    cv::imshow("Erase contours", eraseContours);

    // Dilation
    cv::Mat dilated;

    cv::dilate(eraseContours, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::imshow("Dilation", dilated);

    // Gauss, normalised
    cv::Mat gaussian, normalised;

    cv::GaussianBlur(dilated, gaussian, cv::Size(5, 5), 3);
    cv::normalize(gaussian, normalised, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
    cv::imshow("Gaussian blur", gaussian);
    cv::imshow("Gaussian, normalised", normalised);

    // Bilateral
    cv::Mat bilateral;
    cv::bilateralFilter(faceImg, bilateral, 15, 80, 80);
    cv::imshow("Bilateral", bilateral);

    // Sharp
    double sigma = 1, amount = 3;
    cv::Mat sharp, blur;
    cv::GaussianBlur(faceImg, blur, cv::Size(), sigma);
    cv::addWeighted(faceImg, 1 + amount, blur, -amount, 0, sharp);
    cv::imshow("Sharp", sharp);

    // Filtration
    cv::Mat res = filtration(faceImg, bilateral, sharp, normalised);
    cv::imshow("Result", res);

    cv::waitKey(0);
    system("pause");
};

void scaleFaces(vector<cv::Rect> &faces)
{
    for (auto &face : faces)
    {
        float percent = 0.2f;
        cv::Size deltaSize(face.width * percent, face.height * percent);
        cv::Point offset(deltaSize.width / 2, deltaSize.height / 2);
        face += deltaSize;
        face -= offset;
    }
}

cv::Mat filtration(cv::Mat &faceImage, cv::Mat &bilateral, cv::Mat &sharp, cv::Mat &normalised)
{
    cv::Mat res = cv::Mat::zeros(faceImage.size(), CV_8UC3);

    for (int x = 0; x < faceImage.cols; x++)
    {
        for (int y = 0; y < faceImage.rows; y++)
        {
            cv::Vec3b resPixel;
            cv::Vec3b bilateralPixel = bilateral.at<cv::Vec3b>(x, y);
            cv::Vec3b sharpPixel = sharp.at<cv::Vec3b>(x, y);
            float normalisedPixel = normalised.at<float>(x, y);

            for (int c = 0; c < 3; c++)
                resPixel[c] = normalisedPixel * sharpPixel[c] + (1.0 - normalisedPixel) * bilateralPixel[c];

            res.at<cv::Vec3b>(cv::Point(y, x)) = resPixel;
        }
    }

    return res;
}

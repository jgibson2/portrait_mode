#include <iostream>
#include "FacialRecognitionActor.h"
#include "FeatureMatchingActor.h"
#include "FastBILDepthMapImpl.h"
#include "DepthMapActor.h"
#include "GaussianBlurImpl.h"
#include "BlurActor.h"
#include "BoxBlurImpl.h"
#include "ParGaussianBlurImpl.h"
#include "MonoDepth2Impl.h"
#include "StereoSGBMImpl.h"
#include "ParDiscBlurImpl.h"
#include <unistd.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>

double getFaceDepth(cv::Mat input, int deltaDirection = 100) {
    auto val = (double)input.at<uchar>(input.rows / 2, input.cols / 2)
            + (double)input.at<uchar>(input.rows / 2 - deltaDirection, input.cols / 2)
            + (double)input.at<uchar>(input.rows / 2, input.cols / 2 - deltaDirection)
            + (double)input.at<uchar>(input.rows / 2 + deltaDirection, input.cols / 2)
            + (double)input.at<uchar>(input.rows / 2, input.cols / 2 + deltaDirection)
            + (double)input.at<uchar>(input.rows / 2 - deltaDirection, input.cols / 2 + deltaDirection)
            + (double)input.at<uchar>(input.rows / 2 - deltaDirection, input.cols / 2 - deltaDirection)
            + (double)input.at<uchar>(input.rows / 2 + deltaDirection, input.cols / 2 + deltaDirection)
            + (double)input.at<uchar>(input.rows / 2 + deltaDirection, input.cols / 2 - deltaDirection);
    return val / 9.0;
}

int main(int argc, char** argv) {

    if(argc != 5) {
        std::cerr << "Usage: 1. image 1   2. image 2   3. blur strength   4. dead zone" << std::endl;
        return 1;
    }

    float blurStrength = atof(argv[3]);
    int deadZone = atoi(argv[4]);

    FacialRecognitionActor faceRecognizer("../resources/haarcascade_frontalface_alt.xml");
    FeatureMatchingActor featureMatcher;

    DepthMapActor depthMapActor1(std::make_shared<FastBILDepthMapImpl>());
    DepthMapActor depthMapActor2(std::make_shared<MonoDepth2Impl>(8123));
    DepthMapActor depthMapActor3(std::make_shared<StereoSGBMImpl>());

    BlurActor blurActor1(std::make_shared<ParDiscBlurImpl>(blurStrength));
    BlurActor blurActor2(std::make_shared<ParGaussianBlurImpl>(blurStrength));
    BlurActor blurActor3(std::make_shared<BoxBlurImpl>(blurStrength));

    auto img1 = std::make_shared<cv::Mat>(cv::imread(argv[1]));
    auto img2 = std::make_shared<cv::Mat>(cv::imread(argv[2]));
    auto face1 = faceRecognizer.detectFace(img1);
    auto face2 = faceRecognizer.detectFace(img2);
    if(face1.has_value() && face2.has_value()) {
        std::cout << "Found faces." << std::endl;
        auto matches = featureMatcher.getMatches(img1, img2);
        std::cout << "Found " << matches.size() << " matches." << std::endl;
        auto transformation = featureMatcher.getTransformation(matches);
        if(transformation.has_value()){
            std::cout << "Found transformation." << std::endl;
            std::cout << "Transformation x: " << transformation.value().x << " | Transformation y: " << transformation.value().y << std::endl;
            FeatureMatchingActor::shiftImage(*img2, 0, -1*transformation.value().y);
        }
        featureMatcher.viewFilteredMatches(img1, img2);
        auto depthMap = depthMapActor1.getDepthMap(img1, img2);
        if(!depthMap.has_value()) {
            return 1;
        }
        cv::namedWindow( "Depth map", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Depth map", *depthMap.value() );                   // Show our image inside it.
        cv::waitKey(0);
        cv::imwrite("depth_map.jpg", *depthMap.value());

        auto faceRect = face1.value();
        auto faceData = depthMap.value()->operator()(faceRect);
        auto faceDepth = getFaceDepth(faceData);

        auto img1_blur1 = std::make_shared<cv::Mat>(cv::imread(argv[1]));

        std::cout << "Face depth: " << faceDepth << std::endl;
        blurActor1.blur(img1_blur1, depthMap.value(), (int)std::round(faceDepth), deadZone);
        std::cout << "Blurred with strength " << blurStrength << " and dead zone " << deadZone << std::endl;

        cv::namedWindow( "Blurred", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Blurred", *img1_blur1 );                   // Show our image inside it.
        cv::waitKey(0);
        cv::imwrite("output.jpg", *img1_blur1);

    }
    return 0;

}

/*
auto img1_blur1 = std::make_shared<cv::Mat>(cv::imread(argv[1]));
auto img1_blur2 = std::make_shared<cv::Mat>(cv::imread(argv[1]));
auto img1_blur3 = std::make_shared<cv::Mat>(cv::imread(argv[1]));


std::cout << "Face depth: " << faceDepth << std::endl;
blurActor1.blur(img1_blur1, depthMap.value(), (int)std::round(faceDepth), deadZone);
std::cout << "Blurred with strength " << blurStrength << " and dead zone " << deadZone << std::endl;

cv::namedWindow( "Blurred", cv::WINDOW_AUTOSIZE );// Create a window for display.
cv::imshow( "Blurred", *img1_blur1 );                   // Show our image inside it.
cv::waitKey(0);
cv::imwrite("output_discblur.jpg", *img1_blur1);

std::cout << "Face depth: " << faceDepth << std::endl;
blurActor2.blur(img1_blur2, depthMap.value(), (int)std::round(faceDepth), deadZone);
std::cout << "Blurred with strength " << blurStrength << " and dead zone " << deadZone << std::endl;

cv::namedWindow( "Blurred", cv::WINDOW_AUTOSIZE );// Create a window for display.
cv::imshow( "Blurred", *img1_blur2 );                   // Show our image inside it.
cv::waitKey(0);
cv::imwrite("output_gaussianblur.jpg", *img1_blur2);

std::cout << "Face depth: " << faceDepth << std::endl;
blurActor3.blur(img1_blur3, depthMap.value(), (int)std::round(faceDepth), deadZone);
std::cout << "Blurred with strength " << blurStrength << " and dead zone " << deadZone << std::endl;

cv::namedWindow( "Blurred", cv::WINDOW_AUTOSIZE );// Create a window for display.
cv::imshow( "Blurred", *img1_blur3 );                   // Show our image inside it.
cv::waitKey(0);
cv::imwrite("output_boxblur.jpg", *img1_blur3);
 */
//
// Created by john on 11/24/19.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "BoxBlurImpl.h"

void BoxBlurImpl::operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) {
    //create mask based on depth (1-0 mask)
    cv::Mat fimg;
    img->convertTo(fimg, CV_32FC3, 1.0/255.0);
    auto output = cv::Mat(cv::Mat::zeros(fimg.rows, fimg.cols, fimg.type()));
    auto maskOut = cv::Mat(cv::Mat::zeros(depthMap->rows, depthMap->cols, CV_32F));
    auto disparities = cv::Mat(cv::abs(*depthMap - targetDepth));
    double min, max;
    cv::minMaxIdx(disparities, &min, &max);

    for(int d = (int)min; d <= max; d += 1) {
        std::cout << "Blurring at depth " << d << std::endl << std::flush;
        cv::Mat mask;
        cv::Mat blurredImg;
        cv::compare(disparities, d * cv::Mat::ones(disparities.rows, disparities.cols, disparities.type()), mask, cv::CMP_EQ);

        mask.convertTo(mask, CV_32F, 1.0/255.0);
        cv::Mat mask3Channel;
        cv::Mat t[] = {mask, mask, mask};
        merge(t, 3, mask3Channel);
        cv::multiply(fimg, mask3Channel, blurredImg);
        std::cout << "Found " << cv::countNonZero(mask) << " elements" << std::endl << std::flush;

        //define disk radius = |depth - targetDepth|
        int radius = d - deadZone;
        if(d <= deadZone) {
            radius = 1;
        }
        radius += 1 - (radius % 2);

        //blur mask and all channels of image
        cv::blur(blurredImg, blurredImg, cv::Size(radius, radius));
        cv::blur(mask, mask, cv::Size(radius, radius));

        //add (blurred mask * blurred image) to output image

        cv::Mat oneMinusBlurredMask3Channel;
        cv::Mat omt[] = {1 - mask, 1 - mask, 1 - mask};
        merge(omt, 3, oneMinusBlurredMask3Channel);

        cv::multiply(output, oneMinusBlurredMask3Channel, output);
        cv::add(blurredImg, output, output);

        cv::multiply(maskOut, 1 - mask, maskOut);
        cv::add(maskOut, mask, maskOut);


//        cv::namedWindow( "Test", cv::WINDOW_AUTOSIZE );// Create a window for display.
//        cv::imshow( "Test", output );                   // Show our image inside it.
//        cv::waitKey(0);
    }
    cv::Mat maskOut3Channel;
    cv::Mat t[] = {maskOut, maskOut, maskOut};
    merge(t, 3, maskOut3Channel);
    cv::divide(output, maskOut3Channel, output);
    output.convertTo(output, img->type(), 255);
    *img = output;
}

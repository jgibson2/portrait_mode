//
// Created by john on 11/26/19.
//

#ifndef PORTRAIT_MODE_STEREOSGBMIMPL_H
#define PORTRAIT_MODE_STEREOSGBMIMPL_H


#include <opencv2/opencv.hpp>
#include "DepthMapImpl.h"

class StereoSGBMImpl : public DepthMapImpl {
public:
    explicit StereoSGBMImpl(int minDisparity = 0, int numDisparities = 64, int blockSize = 9, int P1 = 1, int P2 = 32);

    std::shared_ptr<cv::Mat> operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) override;
private:
    int _minDisparity, _numDisparities, _blockSize, _P1, _P2;
    cv::Ptr<cv::StereoMatcher> _matcher;
};


#endif //PORTRAIT_MODE_STEREOSGBMIMPL_H

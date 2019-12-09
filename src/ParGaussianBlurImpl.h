//
// Created by john on 11/25/19.
//

#ifndef PORTRAIT_MODE_PARGAUSSIANBLURIMPL_H
#define PORTRAIT_MODE_PARGAUSSIANBLURIMPL_H


#include "BlurImpl.h"

class ParGaussianBlurImpl : public BlurImpl {
public:
    explicit ParGaussianBlurImpl(float blurStrength, float sigma = 10.0);
    void operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) override;
public:
    float _sigma;
};

#endif //PORTRAIT_MODE_PARGAUSSIANBLURIMPL_H

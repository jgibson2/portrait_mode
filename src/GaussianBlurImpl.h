//
// Created by john on 11/24/19.
//

#ifndef PORTRAIT_MODE_GAUSSIANBLURIMPL_H
#define PORTRAIT_MODE_GAUSSIANBLURIMPL_H


#include "BlurImpl.h"

class GaussianBlurImpl : public BlurImpl {
public:
    explicit GaussianBlurImpl(float sigma);
    void operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) override;
public:
    float _sigma;
};


#endif //PORTRAIT_MODE_GAUSSIANBLURIMPL_H

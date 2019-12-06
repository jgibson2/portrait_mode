//
// Created by john on 11/24/19.
//

#ifndef PORTRAIT_MODE_BOXBLURIMPL_H
#define PORTRAIT_MODE_BOXBLURIMPL_H


#include "BlurImpl.h"

class BoxBlurImpl : public BlurImpl {
public:
    BoxBlurImpl()= default;
    void operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) override;
};


#endif //PORTRAIT_MODE_BOXBLURIMPL_H

//
// Created by john on 11/24/19.
//

#ifndef PORTRAIT_MODE_BOXBLURIMPL_H
#define PORTRAIT_MODE_BOXBLURIMPL_H


#include "BlurImpl.h"

class BoxBlurImpl : public BlurImpl {
public:
    explicit BoxBlurImpl(float m) : _m(m) {}
    void operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) override;
private:
    float _m;
};


#endif //PORTRAIT_MODE_BOXBLURIMPL_H

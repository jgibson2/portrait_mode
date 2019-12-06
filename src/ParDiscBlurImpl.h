//
// Created by john on 11/27/19.
//

#ifndef PORTRAIT_MODE_PARDISCBLURIMPL_H
#define PORTRAIT_MODE_PARDISCBLURIMPL_H


#include "BlurImpl.h"

class ParDiscBlurImpl : public BlurImpl {
public:
    explicit ParDiscBlurImpl(float m = 1.0);
    void operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) override;
private:
    float _m;
};


#endif //PORTRAIT_MODE_PARDISCBLURIMPL_H

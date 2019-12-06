//
// Created by john on 11/18/19.
//

#ifndef PORTRAIT_MODE_BLURACTOR_H
#define PORTRAIT_MODE_BLURACTOR_H


#include "BlurImpl.h"

class BlurActor {
public:
    explicit BlurActor(const std::shared_ptr<BlurImpl> &impl);
    void blur(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone);
private:
    std::shared_ptr<BlurImpl> _impl;
};


#endif //PORTRAIT_MODE_BLURACTOR_H

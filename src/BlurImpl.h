//
// Created by john on 11/18/19.
//

#ifndef PORTRAIT_MODE_BLURIMPL_H
#define PORTRAIT_MODE_BLURIMPL_H


#include <cstddef>
#include <opencv2/core/mat.hpp>

class BlurImpl {
public:
    virtual void operator()(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone) = 0;
};


#endif //PORTRAIT_MODE_BLURIMPL_H

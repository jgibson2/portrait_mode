//
// Created by john on 11/18/19.
//

#ifndef PORTRAIT_MODE_DEPTHMAPIMPL_H
#define PORTRAIT_MODE_DEPTHMAPIMPL_H


#include <opencv2/core/mat.hpp>
#include <optional>

class DepthMapImpl {
public:
    virtual std::shared_ptr<cv::Mat> operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) = 0;

};


#endif //PORTRAIT_MODE_DEPTHMAPIMPL_H

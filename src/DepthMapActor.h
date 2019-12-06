//
// Created by john on 11/18/19.
//

#ifndef PORTRAIT_MODE_DEPTHMAPACTOR_H
#define PORTRAIT_MODE_DEPTHMAPACTOR_H


#include <optional>
#include <opencv2/core/mat.hpp>
#include "DepthMapImpl.h"

class DepthMapActor {
public:
    explicit DepthMapActor(const std::shared_ptr<DepthMapImpl>& impl);
    std::optional<std::shared_ptr<cv::Mat>> getDepthMap(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2);

private:
    std::shared_ptr<DepthMapImpl> _impl;
};


#endif //PORTRAIT_MODE_DEPTHMAPACTOR_H

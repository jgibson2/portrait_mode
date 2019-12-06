//
// Created by john on 11/26/19.
//

#ifndef PORTRAIT_MODE_MONODEPTH2IMPL_H
#define PORTRAIT_MODE_MONODEPTH2IMPL_H

#include "DepthMapImpl.h"

class MonoDepth2Impl : public DepthMapImpl {
public:
    MonoDepth2Impl(int port);

    std::shared_ptr<cv::Mat>
    operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) override;
private:
    int _port;
};


#endif //PORTRAIT_MODE_MONODEPTH2IMPL_H

//
// Created by john on 11/18/19.
//

#include "BlurActor.h"

BlurActor::BlurActor(const std::shared_ptr<BlurImpl>& impl) : _impl(impl) {}

void BlurActor::blur(std::shared_ptr<cv::Mat> img, std::shared_ptr<cv::Mat> depthMap, int targetDepth, unsigned int deadZone){
    _impl->operator()(img, depthMap, targetDepth, deadZone);
}
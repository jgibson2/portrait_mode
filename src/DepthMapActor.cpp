//
// Created by john on 11/18/19.
//

#include "DepthMapActor.h"

DepthMapActor::DepthMapActor(const std::shared_ptr<DepthMapImpl>& impl) : _impl(impl) {}

std::optional<std::shared_ptr<cv::Mat>>
DepthMapActor::getDepthMap(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) {
    for(uint8_t i = 0; i < 10; ++i) {
        auto imgptr = _impl->operator()(img1, img2);
        if(imgptr != nullptr) {
            return std::optional<std::shared_ptr<cv::Mat>>(imgptr);
        }
    }
    return std::nullopt;
}

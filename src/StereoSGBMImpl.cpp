//
// Created by john on 11/26/19.
//

#include "StereoSGBMImpl.h"

std::shared_ptr<cv::Mat> StereoSGBMImpl::operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) {
    static cv::Mat img1_8bit;
    static cv::Mat img2_8bit;
    cv::cvtColor( *img1, img1_8bit, cv::COLOR_BGR2GRAY ); // Convert to Gray Scale
    cv::cvtColor( *img2, img2_8bit, cv::COLOR_BGR2GRAY ); // Convert to Gray Scale
    auto map = std::make_shared<cv::Mat>();

    img1_8bit.convertTo(img1_8bit, CV_8UC1);
    img2_8bit.convertTo(img2_8bit, CV_8UC1);

    _matcher->compute(img1_8bit, img2_8bit, *map);
    map->convertTo(*map, CV_8UC1,
               255.0 / _numDisparities,
               -_minDisparity * 255.0f / _numDisparities);
    return map;
}

StereoSGBMImpl::StereoSGBMImpl(int minDisparity, int numDisparities, int blockSize, int P1, int P2) : _minDisparity(minDisparity),
                                                                                      _numDisparities(numDisparities),
                                                                                      _blockSize(blockSize),
                                                                                                      _P1(P1), _P2(P2){
    _matcher = cv::StereoSGBM::create(_minDisparity, _numDisparities, _blockSize, _P1, _P2);

}

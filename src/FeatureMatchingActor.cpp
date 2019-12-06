//
// Created by john on 11/18/19.
//

#include "FeatureMatchingActor.h"
#include <cmath>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

FeatureMatchingActor::FeatureMatchingActor(float middle_percentile_range, float nn_match_ratio) :
    _middle_percentile_range(middle_percentile_range), _nn_match_ratio(nn_match_ratio) {

}

std::vector<FeatureMatchingActor::Pair2f>
FeatureMatchingActor::getMatches(const std::shared_ptr<cv::Mat> &img1, const std::shared_ptr<cv::Mat> &img2) {
    static std::vector<cv::KeyPoint> _kpts1, _kpts2;
    static cv::Mat _desc1, _desc2;
    static std::vector< std::vector<cv::DMatch> > _nn_matches;
    static std::vector< std::vector<cv::DMatch> > _filtered_matches;

    _akaze->detectAndCompute(*img1, cv::noArray(), _kpts1, _desc1);
    _akaze->detectAndCompute(*img2, cv::noArray(), _kpts2, _desc2);
    _matcher.knnMatch(_desc1, _desc2, _nn_matches, 2);

    std::vector<FeatureMatchingActor::Pair2f> matches;

    std::copy_if(_nn_matches.cbegin(), _nn_matches.cend(), std::back_inserter(_filtered_matches), [this](auto match) -> bool {
        return match[0].distance < _nn_match_ratio * match[1].distance;
    });

    std::transform(_filtered_matches.begin(), _filtered_matches.end(), std::back_inserter(matches), [&](auto match) -> Pair2f {
        return FeatureMatchingActor::Pair2f{_kpts1[match[0].queryIdx].pt, _kpts2[match[0].trainIdx].pt};
    });

    _kpts1.clear(); _kpts2.clear(); _nn_matches.clear(); _filtered_matches.clear();
    return matches;
}

std::optional<cv::Point2i>
FeatureMatchingActor::getTransformation(std::vector<FeatureMatchingActor::Pair2f> matches) {
    std::sort(matches.begin(), matches.end(), [](auto a, auto b) -> bool {
        return tan(a.second.y - a.first.y / (a.second.x - a.first.x + 0.0001)) < tan(b.second.y - b.first.y / (b.second.x - b.first.x + 0.0001));
    });
    auto lower = static_cast<size_t>(round((1.0 -_middle_percentile_range) / 2.0 * matches.size()));
    auto upper = matches.size() - lower;
    if (matches.empty() || lower >= upper) {
        return std::nullopt;
    }
    auto middleAvgTransformation = std::accumulate(matches.cbegin() + lower, matches.cbegin() + upper, cv::Point2f(0.0, 0.0), [](const cv::Point2f& accum, const Pair2f& pair){
        return accum + cv::Point2f(pair.second.x - pair.first.x, pair.second.y - pair.first.y);
    });
    middleAvgTransformation.x /= (upper - lower);
    middleAvgTransformation.y /= (upper - lower);
    return cv::Point2i(round(middleAvgTransformation.x), round(middleAvgTransformation.y));
}

#ifdef DEBUG
void FeatureMatchingActor::viewFilteredMatches(const std::shared_ptr<cv::Mat> &img1, const std::shared_ptr<cv::Mat> &img2) {
    static std::vector<cv::KeyPoint> _kpts1, _kpts2;
    static cv::Mat _desc1, _desc2;
    static std::vector< std::vector<cv::DMatch> > _nn_matches;
    static std::vector< std::vector<cv::DMatch> > _filtered_matches;

    _akaze->detectAndCompute(*img1, cv::noArray(), _kpts1, _desc1);
    _akaze->detectAndCompute(*img2, cv::noArray(), _kpts2, _desc2);
    _matcher.knnMatch(_desc1, _desc2, _nn_matches, 2);

    std::vector<std::vector<cv::DMatch>> matches;

    std::copy_if(_nn_matches.cbegin(), _nn_matches.cend(), std::back_inserter(_filtered_matches), [this](auto match) -> bool {
        return match[0].distance < _nn_match_ratio * match[1].distance;
    });

    std::sort(_filtered_matches.begin(), _filtered_matches.end(), [&](const auto& a, const auto& b) -> bool {
        auto a_first = _kpts1[a[0].queryIdx].pt;
        auto a_second = _kpts2[a[0].trainIdx].pt;;
        auto b_first = _kpts1[b[0].queryIdx].pt;
        auto b_second = _kpts2[b[0].trainIdx].pt;
        return tan(a_second.y - a_first.y / (a_second.x - a_first.x + 0.0001)) < tan(b_second.y - b_first.y / (b_second.x - b_first.x + 0.0001));
    });
    auto lower = static_cast<size_t>(round((1.0 - _middle_percentile_range) / 2.0 * _filtered_matches.size()));
    auto upper = _filtered_matches.size() - lower;
    if (_filtered_matches.empty() || lower >= upper) {
        std::cout << "Matches empty!" << std::endl;
        return;
    }

    std::for_each(_filtered_matches.cbegin() + lower, _filtered_matches.cbegin() + upper, [&](auto& match) {
        matches.push_back(match);
    });

    cv::Mat res;
    cv::drawMatches(*img1, _kpts1, *img2, _kpts2, matches, res);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", res );                   // Show our image inside it.
    cv::waitKey(0);

    _kpts1.clear(); _kpts2.clear(); _nn_matches.clear(); _filtered_matches.clear();
}

void FeatureMatchingActor::shiftImage(cv::Mat &image, int xShift, int yShift) {
    float mData[] = {1, 0, (float)xShift, 0, 1, (float)yShift};
    cv::Mat M(2, 3, CV_32F, mData);
    cv::warpAffine(image, image, M, cv::Size2l(image.cols, image.rows));
}

#endif
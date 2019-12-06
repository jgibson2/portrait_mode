//
// Created by john on 11/18/19.
//

#ifndef PORTRAIT_MODE_FEATUREMATCHINGACTOR_H
#define PORTRAIT_MODE_FEATUREMATCHINGACTOR_H

#define DEBUG 1

#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

class FeatureMatchingActor {
public:
    using Pair2f = std::pair<cv::Point2f, cv::Point2f>;
    explicit FeatureMatchingActor(float middle_percentile_range = 0.75f, float nn_match_ratio = 0.8f);

    std::vector<Pair2f> getMatches(const std::shared_ptr<cv::Mat>& img1, const std::shared_ptr<cv::Mat>& img2);

    std::optional<cv::Point2i> getTransformation(std::vector<FeatureMatchingActor::Pair2f> matches);

    static void shiftImage(cv::Mat& image, int xShift, int yShift);

#ifdef DEBUG
    void viewFilteredMatches(const std::shared_ptr<cv::Mat> &img1, const std::shared_ptr<cv::Mat> &img2);
#endif

private:
    float _middle_percentile_range, _nn_match_ratio;
    cv::Ptr<cv::AKAZE> _akaze = cv::AKAZE::create();
    cv::BFMatcher _matcher{cv::NormTypes::NORM_HAMMING};

};


#endif //PORTRAIT_MODE_FEATUREMATCHINGACTOR_H

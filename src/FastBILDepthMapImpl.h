//
// Created by john on 11/22/19.
//

#ifndef PORTRAIT_MODE_FASTBILDEPTHMAPIMPL_H
#define PORTRAIT_MODE_FASTBILDEPTHMAPIMPL_H

#include "stereo_matcher_birchfield_tomasi.h"
#include "DepthMapImpl.h"


class FastBILDepthMapImpl : public DepthMapImpl {
public:
    std::shared_ptr<cv::Mat>
    operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) override;

private:
    // bilateral grid properties
    const int property_grid_sigma_spatial = 16;
    const int property_grid_sigma_luma = 16;
    const int property_grid_sigma_chroma = 16;

    // stereo matching properties
    const int property_disparity_min = -50;
    const int property_disparity_max = 50;
    const stereo_matcher_birchfield_tomasi::block_filter_size property_stereo_block_filter = stereo_matcher_birchfield_tomasi::block_filter_size::size_5x5;

    // solver properties
    const int property_solver_nb_iterations = 500;
    const float property_solver_lambda = 0.2f;
    const int property_solver_keep_nb_of_intermediate_images = 0; // you can ignore this one, for debugging
    // post-process domain transform properties
    const float property_dt_sigmaSpatial = 40.0f;
    const float property_dt_sigmaColor = 220.0f;
    const int property_dt_numIters = 3;
};


#endif //PORTRAIT_MODE_FASTBILDEPTHMAPIMPL_H

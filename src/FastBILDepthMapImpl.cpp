//
// Created by john on 11/22/19.
//

#include "fast_bilateral_solver.h"
#include "FastBILDepthMapImpl.h"
#include <opencv2/ximgproc/edge_filter.hpp>

std::shared_ptr<cv::Mat>
FastBILDepthMapImpl::operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat> img2) {
    // convert to gray scale
    cv::Mat stereo_images_gray[2];
    cv::cvtColor(*img1, stereo_images_gray[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(*img2, stereo_images_gray[1], cv::COLOR_BGR2GRAY);

    // grid
    bilateral_grid_simplified grid;
    grid.init(*img1, property_grid_sigma_spatial, property_grid_sigma_luma, property_grid_sigma_chroma);

    // stereo matching
    stereo_matcher_birchfield_tomasi stereo_matcher;
    stereo_matcher.get_parameters().disparity_min = property_disparity_min;
    stereo_matcher.get_parameters().disparity_max = property_disparity_max;
    stereo_matcher.get_parameters().filter_size = property_stereo_block_filter;
    stereo_matcher.stereo_match(stereo_images_gray);

    // loss function
    std::vector<int> lookup;
    stereo_matcher.generate_data_loss_table(grid, lookup);

    ///// bilateral solver
    fast_bilateral_solver solver;

    // let's work from "0 --> disparity range" instead of "disparity min --> disparty max"
    // and let's use the minimum disparity image as a starting point.
    cv::Mat input_x = stereo_matcher.get_output().min_disp_image - stereo_matcher.get_parameters().disparity_min;
    cv::Mat input_x_fl;
    input_x.convertTo(input_x_fl, CV_32FC1);
    cv::Mat input_confidence_fl;
    stereo_matcher.get_output().conf_disp_image.convertTo(input_confidence_fl, CV_32FC1, 1.0f / 255.0f);

    // for initialization, let's apply a weighted bilateral filter!
    //   filtered image = blur(image x confidence) / blur(confidence)
    //   the confidence image is an image where a 1 means we have a match with the stereo matcher. 0 if there was no match.
    cv::Mat tc_im;
    cv::multiply(input_x_fl, input_confidence_fl, tc_im);
    cv::Mat tc = grid.filter(tc_im);
    cv::Mat c = grid.filter(input_confidence_fl);
    cv::Mat start_point_image;
    cv::divide(tc, c, start_point_image);

    //// decomment if you want to start with 0...
    ////start_point_image = cv::Scalar(0.f);

    std::optional<cv::Mat> final_disparty_image_opt = solver.solve(start_point_image, grid, lookup,
                                                (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min) + 1, property_solver_lambda, property_solver_nb_iterations, property_solver_keep_nb_of_intermediate_images);
    if(!final_disparty_image_opt.has_value()){
        return nullptr;
    }
    auto final_disparty_image = final_disparty_image_opt.value();
    final_disparty_image += (float)stereo_matcher.get_parameters().disparity_min;

    // display disparity image
    auto adjmap_final = std::make_shared<cv::Mat>();

    cv::ximgproc::dtFilter(*img1,
                           final_disparty_image, final_disparty_image,
                           property_dt_sigmaSpatial, property_dt_sigmaColor,
                           cv::ximgproc::DTF_RF,
                           property_dt_numIters);

    final_disparty_image.convertTo(*adjmap_final, CV_8UC1,
                                   255.0 / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min),
                                   -stereo_matcher.get_parameters().disparity_min * 255.0f / (stereo_matcher.get_parameters().disparity_max - stereo_matcher.get_parameters().disparity_min));
    return std::shared_ptr<cv::Mat>(adjmap_final);
}

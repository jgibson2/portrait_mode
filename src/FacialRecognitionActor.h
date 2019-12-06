//
// Created by john on 11/18/19.
//

#ifndef PORTRAIT_MODE_FACIALRECOGNITIONACTOR_H
#define PORTRAIT_MODE_FACIALRECOGNITIONACTOR_H

#include <string>
#include <memory>
#include <optional>
#include <vector>
#include <exception>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

class FacialRecognitionActor {
public:
    FacialRecognitionActor(std::string classifierFilename);

    std::optional<cv::Rect> detectFace(const std::shared_ptr<cv::Mat>& img);

private:

    cv::CascadeClassifier _classifier{};
    std::string _classFname;
    cv::Mat gray;

};


#endif //PORTRAIT_MODE_FACIALRECOGNITIONACTOR_H

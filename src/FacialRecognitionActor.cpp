//
// Created by john on 11/18/19.
//

#include "FacialRecognitionActor.h"

FacialRecognitionActor::FacialRecognitionActor(std::string classifierFilename) : _classFname(classifierFilename) {
    if(!_classifier.load(classifierFilename)){ throw std::invalid_argument("classifier file not found");}
}

std::optional<cv::Rect> FacialRecognitionActor::detectFace(const std::shared_ptr<cv::Mat> &img) {
    std::vector<cv::Rect> faces;

    cv::cvtColor( *img, gray, cv::COLOR_BGR2GRAY ); // Convert to Gray Scale
    equalizeHist( gray, gray );

    // Detect faces of different sizes using cascade classifier
    _classifier.detectMultiScale( gray, faces );
    if(faces.empty()) {
        return std::nullopt;
    }

//    cv::Point center( faces[0].x + faces[0].width/2, faces[0].y + faces[0].height/2 );
//    ellipse( *img, center, cv::Size( faces[0].width/2, faces[0].height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4 );
//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::imshow( "Display window", *img );                   // Show our image inside it.
//    cv::waitKey(0);

    return std::optional<cv::Rect>{faces[0]};
}

//
// Created by john on 11/26/19.
//

#include "MonoDepth2Impl.h"
#include <nngpp/protocol/req0.h>
#include <nngpp/nngpp.h>
#include <opencv2/opencv.hpp>


std::shared_ptr<cv::Mat> MonoDepth2Impl::operator()(std::shared_ptr<cv::Mat> img1, std::shared_ptr<cv::Mat>) {
    try {
        // create a socket for the req protocol
        nng::socket req_sock = nng::req::open();

        // req dials and establishes a connection
        req_sock.dial( std::string("tcp://127.0.0.1:" + std::to_string(_port)).c_str() );

        {
            std::vector<uchar> data;
            cv::imencode(".jpg", *img1, data);
            std::cout << "Encoded " << data.size() << " bytes of data" << std::endl << std::flush;
            int r = nng_send(req_sock.get(),data.data(),data.size(), 0);
            if( r != 0 ) {
                throw nng::exception(r,"nng_send");
            }
            std::cout << "Sent " << data.size() << " bytes of data" << std::endl << std::flush;
        }

        nng::buffer req_buf = req_sock.recv();
        std::cout << "Received " << req_buf.size() << " bytes of data" << std::endl << std::flush;
        std::vector<uchar> data2(req_buf.data<uchar>(), req_buf.data<uchar>() + req_buf.size());
        auto map = std::make_shared<cv::Mat>(cv::imdecode(data2, cv::IMREAD_COLOR));
        cv::cvtColor(*map, *map, cv::COLOR_BGR2GRAY); // Convert to Gray Scale
        return map;


    }
    catch( const nng::exception& e ) {
        // who() is the name of the nng function that produced the error
        // what() is a description of the error code
        printf( "%s: %s\n", e.who(), e.what() );
        return nullptr;
    }
}

MonoDepth2Impl::MonoDepth2Impl(int port) : _port(port) {}

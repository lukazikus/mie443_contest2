#ifndef FUNC_HEADER_H
#define FUNC_HEADER_H

#include <nav_header.h>
#include <imageTransporter.hpp>

using namespace std;

bool init(vector<vector<float> >& coord, std::vector<cv::Mat>& imgs_track);

int findPic(imageTransporter imgTransport, vector<cv::Mat> imgs_track);
int feature2D_homography (const char *image1, const char *image2 );

#endif

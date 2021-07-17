//
// Created by celine on 27/03/19.
//

#ifndef MESHHOMO_PRINTMAT_H
#define MESHHOMO_PRINTMAT_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

void printmat3D(Mat_<float>, int, int, int);
void printmat3D(vector<Mat>, int, int, int);
void printmat3D(Mat, int, int, int);
void printmat2D(Mat, int, int);

#endif //MESHHOMO_PRINTMAT_H

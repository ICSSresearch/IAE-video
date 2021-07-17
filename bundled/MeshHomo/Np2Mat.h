//
// Created by celine on 27/03/19.
//

#ifndef MESHHOMO_NP2MAT_H
#define MESHHOMO_NP2MAT_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "PrintMat.h"
using namespace std;
using namespace cv;
namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace p;
using namespace np;

enum class DTYPE{
    MAT, MAT_
};

// convert numpy array to vector<Mat> or vector<Mat_<float>> channels
// TODO generalize to vector<Mat_<T>>
vector<Mat> np2ch(ndarray&, ndarray&, ndarray&);

// convert numpy array to vector<Mat> but this time assuming one ndarray which is contiguous
// and assume the array is channel FIRST
// ndarray -> vector<Mat>, vector<Mat_<>>
vector<Mat> np2ch(ndarray&);

// convert numpy array to Mat
Mat_<float> np2mat(ndarray&);

void mat2np(vector<Mat_<float>>&, Mat_<float>&, double, double, double, double);

#endif //MESHHOMO_NP2MAT_H

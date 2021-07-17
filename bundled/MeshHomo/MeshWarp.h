//
// Created by celine on 31/03/19.
//

#ifndef MESHHOMO_WARP_H
#define MESHHOMO_WARP_H

#include "Asap.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "math.h"
#include "Np2Mat.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace p;
using namespace np;

class MeshWarp {
public:
    MeshWarp(int imageHeight, int imageWidth, double quadHeight, double quadWidth, double alpha, double gap);
    MeshWarp();
    ~MeshWarp();
    bool warp(ndarray& image, ndarray& warpImg);
    void warpch(ndarray& ch1, ndarray& ch2, ndarray& ch3, ndarray& warpImg);
    void setVertex(int vertex_x, int vertex_y, double point_x, double point_y);
    void printVertex(int vertex_x, int vertex_y);
    void set(Asap& _asap);

private:
    Asap asap;
    bool quadWarp(vector<Mat>& image_mat, vector<Mat_<float>>& warpImg_mat, Quad q1, Quad q2);
    void myWarp(double  minx, double  maxx, double  miny, double  maxy,
                vector<Mat>& image_mat, vector<Mat_<float>>& warpImg_mat, Homography invH);
    bool warpImage(vector<Mat>& image_mat, vector<Mat_<float>>& warpImg_mat);
};


#endif //MESHHOMO_WARP_H

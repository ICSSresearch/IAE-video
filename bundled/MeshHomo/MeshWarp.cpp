//
// Created by celine on 31/03/19.
//

#include "MeshWarp.h"

MeshWarp::MeshWarp(int imageHeight, int imageWidth,
                                         double quadHeight, double quadWidth, double alpha, double gap){
    asap = Asap(imageHeight, imageWidth, quadHeight, quadWidth, alpha, gap);
}

MeshWarp::MeshWarp(){

}

void MeshWarp::set(Asap& _asap){
    asap = _asap;
}

MeshWarp::~MeshWarp(){

}

bool MeshWarp::warpImage(vector<Mat>& image_mat, vector<Mat_<float>>& warpImg_mat){
    // warp each mesh
    int meshHeight = asap.imgHeight / asap.quadHeight;
    int meshWidth = asap.imgWidth / asap.quadWidth;
    bool success = true;

#pragma omp parallel num_threads(100)
    {
        int i, j;
#pragma omp for collapse(2)
        for (i = 0; i < meshHeight; ++i) {
            for (j = 0; j < meshWidth; ++j) {
                POINT p0;
                p0 = asap.source.getVertex(i, j);
                POINT p1;
                p1 = asap.source.getVertex(i, j + 1);
                POINT p2;
                p2 = asap.source.getVertex(i + 1, j);
                POINT p3;
                p3 = asap.source.getVertex(i + 1, j + 1);
                POINT q0;
                q0 = asap.destin.getVertex(i, j);
                POINT q1;
                q1 = asap.destin.getVertex(i, j + 1);
                POINT q2;
                q2 = asap.destin.getVertex(i + 1, j);
                POINT q3;
                q3 = asap.destin.getVertex(i + 1, j + 1);
                Quad qd1(p0, p1, p2, p3);
                Quad qd2(q0, q1, q2, q3);
                bool ss = quadWarp(image_mat, warpImg_mat, qd1, qd2);
                if (!ss) success = false;
            }
        }
    }
//    if (!success) cout<<"Failed: grid goes over the boundary. Try increasing padding."<<endl;
    return success;
}

//TODO: The following two should be overloaded, but need to figure out how boost works with that
void MeshWarp::warpch(ndarray& ch1, ndarray& ch2, ndarray& ch3, ndarray& warpImg){
    // get image height and width
    double img_height = ch1.shape(0);
    double img_width = ch1.shape(1);

    // parse channels to vector<Mat>
    vector<Mat> channels = np2ch(ch1, ch2, ch3);

    // parse warpImg to Mat_<float> for final copying
    Mat_<float> warpImg_mat = np2mat(warpImg);

    // create temporary vector<Mat> warpImg to store result of warp
    vector<Mat_<float>> warpImg_temp(3);

    for(int k=0; k<3; ++k){
        warpImg_temp[k] = Mat::zeros(Size(img_width + asap.gap*2, img_height + asap.gap*2), CV_32FC1);
    }

    // warp and store restul in warpImg_temp
    warpImage(channels,warpImg_temp);

    // copy warpImg_temp to warpImg_mat
    mat2np(warpImg_temp, warpImg_mat, img_height, img_width, 3.0, asap.gap);

}

bool MeshWarp::warp(ndarray& image, ndarray& warpImg){
    // get image height and width: channel first image
    double img_height = image.shape(1);
    double img_width = image.shape(2);

    // parse image into channels
    vector<Mat> channels = np2ch(image);

    // parse warpImg to Mat_<float> for final copying
    Mat_<float> warpImg_mat = np2mat(warpImg);

    // create temporary vector<Mat> warpImg to store result of warp
    vector<Mat_<float>> warpImg_temp(3);
    for(int k=0; k<3; ++k){
        warpImg_temp[k] = Mat::zeros(Size(img_width + asap.gap*2, img_height + asap.gap*2), CV_32FC1);
    }

    // warp and store restul in warpImg_temp
    bool success = warpImage(channels,warpImg_temp);
        // copy warpImg_temp to warpImg_mat
//        cout <<"copying"<<endl;
    mat2np(warpImg_temp, warpImg_mat, img_height, img_width, 3.0, asap.gap);
    return success;
}

bool MeshWarp::quadWarp(vector<Mat>& image_mat, vector<Mat_<float>>& warpImg_mat, Quad q1, Quad q2){
//    cout<<"quadWarp"<<endl;
    double minx = q2.getMinX();
    double maxx = q2.getMaxX();
    double miny = q2.getMinY();
    double maxy = q2.getMaxY();
    double x_rb = asap.gap + asap.imgWidth; // index of right edge of image with padding
    double y_bb = asap.gap + asap.imgHeight; // index of bottom edge of image with padding

    // cap min at -gap, and cap max w + gap
    minx = max(floor(minx), -asap.gap+1);
    miny = max(floor(miny), -asap.gap+1);
    maxx = min(ceil(maxx), x_rb);
    maxy = min(ceil(maxy), y_bb);

    // don't do anything if either of the minimum coordinate goes beyond the image paddings - it shouldn't
    // if so, check what's wrong
//    cout << asap.quadHeight<<" "<<asap.quadWidth<<endl;
//    cout << maxx - minx << " "<<maxy - miny<<endl;
    if (minx > x_rb || miny > y_bb || maxx <= minx || maxy <= miny ||
    ((maxy - miny) > (asap.quadHeight * 2) && (maxx - minx) > (asap.quadWidth * 2))){
        return false; // fail: don't change this portion of warpImage
    }

    Homography h;
    vector<POINT> p1, p2;
    p1.push_back(q2.v00);
    p1.push_back(q2.v01);
    p1.push_back(q2.v10);
    p1.push_back(q2.v11);
    p2.push_back(q1.v00);
    p2.push_back(q1.v01);
    p2.push_back(q1.v10);
    p2.push_back(q1.v11);
//    cout << "computing H"<<endl;
    h.computeFromPoints(p1, p2);
    h.normalize();
//    cout<<"after H"<<endl;
//    cout<<"warp"<<endl;
    myWarp(minx, maxx, miny, maxy, image_mat, warpImg_mat, h);
//    cout<<"afterwarp"<<endl;
    return true;
}

void MeshWarp::myWarp(double  minx, double  maxx, double  miny, double  maxy,
                                 vector<Mat>& image_mat, vector<Mat_<float>>& warpImg_mat, Homography invH){
    double gap = asap.gap;
    double szImx = maxy - miny;
    double szImy = maxx - minx;

    // meshgrid(minx:maxx-1, miny:maxy-1)
    MatrixXd x = VectorXd::LinSpaced(szImy, minx-1, maxx-2).transpose().replicate(szImx , 1);
    x.resize(1, szImx * szImy);
    MatrixXd y = VectorXd::LinSpaced(szImx, miny-1, maxy-2).replicate(1, szImy);
    y.resize(1, szImx * szImy);
    MatrixXd ones = MatrixXd::Ones(1, szImx * szImy);

    // stack x and y
//    cout <<"copy to hpixels"<<endl;
    MatrixXd hPixels(x.rows() + y.rows() + ones.rows(), x.cols());
    hPixels << x, y, ones;

//    cout << "h*hpixels"<<endl;
    MatrixXd _hScene  = invH.getMat() * hPixels; //
    MatrixXf hScene = _hScene.cast<float>();

//    cout << "xprime yprime"<<endl;
    MatrixXf xprime = round(hScene.row(0).array() / hScene.row(2).array());
    MatrixXf yprime = round(hScene.row(1).array() / hScene.row(2).array());

//    cout << "resize "<< szImx<<" "<<szImy<<endl;
    xprime.resize(szImx, szImy);
    yprime.resize(szImx, szImy);
//    cout << "conversion"<<endl;
    Mat_<float> xprime_mat(szImx, szImy);
    Mat_<float> yprime_mat(szImx, szImy);
    eigen2cv(xprime, xprime_mat);
    eigen2cv(yprime, yprime_mat);

    Mat_<float> results0(xprime_mat.size(), xprime_mat.type());
    Mat_<float> results1(xprime_mat.size(), xprime_mat.type());
    Mat_<float> results2(xprime_mat.size(), xprime_mat.type());
//    cout << "interp"<<endl;
    Mat xprimemap, yprimemap;
    convertMaps(xprime_mat, yprime_mat, xprimemap, yprimemap, false);
    remap(image_mat[0], results0, yprimemap, xprimemap, CV_INTER_CUBIC);
    remap(image_mat[1], results1, yprimemap, xprimemap, CV_INTER_CUBIC);
    remap(image_mat[2], results2, yprimemap, xprimemap, CV_INTER_CUBIC);
//    cout<<"after interp"<<endl;
    results0.copyTo(warpImg_mat[0](Range(miny+gap-1,maxy+gap-1),Range(minx+gap-1,maxx+gap-1)));
    results1.copyTo(warpImg_mat[1](Range(miny+gap-1,maxy+gap-1),Range(minx+gap-1,maxx+gap-1)));
    results2.copyTo(warpImg_mat[2](Range(miny+gap-1,maxy+gap-1),Range(minx+gap-1,maxx+gap-1)));
}

void MeshWarp::setVertex(int vertex_x, int vertex_y, double point_x, double point_y){
    POINT point;
    point.x = point_x; point.y = point_y;
    asap.destin.setVertex(vertex_x, vertex_y, point);
}

void MeshWarp::printVertex(int vertex_x, int vertex_y){
    cout<<asap.destin.getVertex(vertex_x, vertex_y).x<<" "<<asap.destin.getVertex(vertex_x, vertex_y).y<<endl;
    cout<<asap.source.getVertex(vertex_x, vertex_y).x<<" "<<asap.source.getVertex(vertex_x, vertex_y).y<<endl;
}

/* boost python wrapper */
BOOST_PYTHON_MODULE(meshwarp)
{
    Py_Initialize();
    np::initialize();

    class_<MeshWarp>("MeshWarp", init<>())
            .def("warp", &MeshWarp::warp)
            .def("warpch", &MeshWarp::warpch)
            .def("setVertex", &MeshWarp::setVertex)
            .def("printVertex", &MeshWarp::printVertex)
            .def("set", &MeshWarp::set);
}

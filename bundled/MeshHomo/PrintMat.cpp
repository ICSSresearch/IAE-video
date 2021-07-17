//
// Created by celine on 27/03/19.
//

#include "PrintMat.h"

void printmat3D(Mat_<float> image, int row, int col, int depth){
    for (int k=0; k<depth; ++k) {
        cout << "channel "<<k<<endl;
        for (int i = 0; i<row; ++i){
            for(int j=0; j<col; ++j){
                cout<<image(i,j, k)<<" ";
            }
            cout << endl;
        }
    }
}

void printmat3D(vector<Mat> image, int row, int col, int depth){
    for (int k=0; k<depth; ++k) {
        cout << "channel "<<k<<endl;
        for (int i = 0; i<row; ++i){
            for(int j=0; j<col; ++j){
                cout<<image[k].at<float>(i,j)<<" ";
            }
            cout << endl;
        }
    }
}

void printmat3D(Mat image_mat, int row, int col, int depth){
    for (int k=0; k<depth; ++k) {
        cout << "channel "<<k<<endl;
        for (int i = 0; i<row; ++i){
            for(int j=0; j<col; ++j){
                cout<<image_mat.at<Vec3f>(i, j)[k]<<" ";
            }
            cout << endl;
        }
    }
}

void printmat2D(Mat image_mat, int row, int col){
    for (int i = 0; i<row; ++i) {
        for (int j = 0; j < col; ++j) {
            cout << image_mat.at<float>(i, j) << " ";
        }
        cout << endl;
    }
}
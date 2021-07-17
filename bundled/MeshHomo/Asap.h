#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Geometry"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#include "Mesh.h"
#include "Quad.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace p;
using namespace np;
using namespace Eigen;
using namespace std;

#define NUM_HOMO_ELEMENTS 9
#define SHAPE_HOMO_MATRIX 3
typedef Eigen::Triplet<double> T;

class Homography
{
	public:
		Homography();
		~Homography();
		Homography(Matrix3d&);
		Homography(vector<double>&);
		Homography(double data[9]);
		Matrix3d getMat();
		Transform<double, 2, Projective> getTransfrom();
		vector<double> getVectorN9();
		double* getDoubleN9();
		int computeFromPoints(vector<Point> p1, vector<Point> p2);
		void normalize();
	private:
		Matrix3d mat;
		Transform<double, 2, Projective> trans;
};

class Asap
{
	public:
		double imgHeight;
		double imgWidth;
		double quadHeight;
		double quadWidth;
		double alpha;
		double gap;
		int height, width;

		Mesh source;
		Mesh destin;

		Asap();
		Asap(int, int, double, double, double, double);
		~Asap();

		int addControlPoints(int ncp, double *p1x, double *p1y, double *p2x, double *p2y);
		string printInfo();
		int solve();
		vector<vector<Homography>> calcHomos(); //Homography h = homos[row][col];
        void calcDoubleHomos(double* homos);

        double CalcError();

		int createDataCons(VectorXd & b);

		int checkBadCPs();

		int createSmoothCons(double weight);

		void addCoefficient_LU(int i, int j);
		void addCoefficient_UL(int i, int j);
		void addCoefficient_UR(int i, int j);
		void addCoefficient_RU(int i, int j);
		void addCoefficient_RD(int i, int j);
		void addCoefficient_DR(int i, int j);
		void addCoefficient_DL(int i, int j);
		void addCoefficient_LD(int i, int j);

		void assignUV(Point v1, Point v2, Point v3, double &u, double &v);
		void dataElement_reset();

	private:
		// smooth constraints
		int nSmoothConstraints;
		vector<Triplet<double>> SmoothConstraints;

		// data constraints
		vector<int> dataElement_i;
		vector<int> dataElement_j;
		vector<double> dataElement_V00;
		vector<double> dataElement_V01;
		vector<double> dataElement_V10;
		vector<double> dataElement_V11;
		vector<Point> dataElement_orgPt;
		vector<Point> dataElement_desPt;

		vector<Triplet<double>> DataConstraints;
		int nDataConstraints;

		int rowCount;
		int columns;
		vector<int> x_index;
		vector<int> y_index;
	};


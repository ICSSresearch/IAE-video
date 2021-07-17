#include "MeshHomo.h"

MeshHomo::MeshHomo(int imageHeight, int imageWidth,
                                         double quadHeight, double quadWidth, double alpha, double gap){
    asap = Asap(imageHeight, imageWidth, quadHeight, quadWidth, alpha, gap);
}

MeshHomo::MeshHomo(){
}

MeshHomo::~MeshHomo(){

}

void MeshHomo::set(Asap& _asap){
    asap = _asap;
}

// cp1 and cp2 are feature list: N x 2
// cp1 and cp2 must be list in python. Use tolist() with numpy array before passing to this function
void MeshHomo::computeHomos(ndarray& _p1x, ndarray& _p1y, ndarray& _p2x, ndarray& _p2y, ndarray& homos)
{
    asap.dataElement_reset();

    //get list length: should be N
    int ncp = _p1x.shape(0);
    int dims = _p2x.shape(0);
    assert(ncp == dims);

    // cast char* data into double *
    // Note: if change double* p1x, the original content in _p1x will change as well
    double* p1x = reinterpret_cast<double*>(_p1x.get_data());
    double* p2x = reinterpret_cast<double*>(_p2x.get_data());
    double* p1y = reinterpret_cast<double*>(_p1y.get_data());
    double* p2y = reinterpret_cast<double*>(_p2y.get_data());

    asap.addControlPoints(ncp, p1x, p1y, p2x, p2y);
    asap.solve();

    // pass ndarray to calcDoubleHomos to store homo result
    double* homoPts = reinterpret_cast<double*>(homos.get_data());
    asap.calcDoubleHomos(homoPts);
}

/* boost python wrapper */
BOOST_PYTHON_MODULE(meshhomo)
{
    Py_Initialize();
    np::initialize();

    class_<MeshHomo>("MeshHomo", init<>())
            .def("computeHomos", &MeshHomo::computeHomos)
            .def("set", &MeshHomo::set);
}

#include "Asap.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "math.h"

using namespace std;
using namespace Eigen;
namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace p;
using namespace np;


class MeshHomo{
    public:
        MeshHomo(int imageHeight, int imageWidth, double quadHeight, double quadWidth, double alpha, double gap);
        MeshHomo();
        ~MeshHomo();
        void computeHomos(ndarray& _p1x, ndarray& _p1y, ndarray& _p2x, ndarray& _p2y, ndarray& homos);
        void set(Asap& _asap);

    private:
        Asap asap;
};

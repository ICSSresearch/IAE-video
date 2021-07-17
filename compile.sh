#!/usr/bin/env bash

# compile bundled
echo "Compile bundled"
ENVIRONMENT_ROOT=$1
OPENCV_LIBRARY_DIR=$2
PYTHON_VERSION=$3
PYTHON_HEADER_DIR=$4
EIGEN_HEADER_DIR=$5
BOOST_HEADER_DIR=$6
BOOST_LIBRARY_DIR=$7

# get objective files
cd bundled/MeshHomo/
rm -rf build
mkdir -p build
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -I$EIGEN_HEADER_DIR -I$BOOST_HEADER_DIR -I$PYTHON_HEADER_DIR -g3 -c Asap.cpp -o build/Asap.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -I$EIGEN_HEADER_DIR -I$BOOST_HEADER_DIR -I$PYTHON_HEADER_DIR -g3 -c Mesh.cpp -o build/Mesh.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -I$EIGEN_HEADER_DIR -I$BOOST_HEADER_DIR -I$PYTHON_HEADER_DIR -g3 -c Np2Mat.cpp -o build/Np2Mat.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -I$EIGEN_HEADER_DIR -I$BOOST_HEADER_DIR -I$PYTHON_HEADER_DIR -g3 -c Quad.cpp -o build/Quad.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -I$EIGEN_HEADER_DIR -I$BOOST_HEADER_DIR -I$PYTHON_HEADER_DIR -g3 -c MeshHomo.cpp -o build/MeshHomo.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto -fopenmp \
    -I$EIGEN_HEADER_DIR -I$BOOST_HEADER_DIR -I$PYTHON_HEADER_DIR -g3 -c MeshWarp.cpp -o build/MeshWarp.o

# link objective files
cd build/
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -o ../../meshhomo.so Asap.o Mesh.o Quad.o MeshHomo.o \
    -L$BOOST_LIBRARY_DIR \
    -lboost_python$PYTHON_VERSION -lboost_numpy$PYTHON_VERSION

g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto -fopenmp \
    -o ../../meshwarp.so MeshWarp.o Asap.o Mesh.o Quad.o Np2Mat.o \
    -L$BOOST_LIBRARY_DIR -L$OPENCV_LIBRARY_DIR \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_python$PYTHON_VERSION -lboost_numpy$PYTHON_VERSION

g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=x86-64 -mtune=generic -fPIC -shared -flto \
    -o ../../asap.so Asap.o Mesh.o Quad.o \
    -L$BOOST_LIBRARY_DIR \
    -lboost_python$PYTHON_VERSION -lboost_numpy$PYTHON_VERSION

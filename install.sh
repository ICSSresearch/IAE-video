#!/usr/bin/env bash

# Default environment
ENVIRONMENT_ROOT=$CONDA_PREFIX
OPENCV_LIBRARY_DIR="$ENVIRONMENT_ROOT/lib"
PYTHON_VERSION="37"
PYTHON_HEADER_DIR="$ENVIRONMENT_ROOT/include/python${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1:2}m"
EIGEN_HEADER_DIR="$ENVIRONMENT_ROOT/include"
BOOST_HEADER_DIR="$ENVIRONMENT_ROOT/include/boost"  # boost/boost contain header
BOOST_LIBRARY_DIR="$ENVIRONMENT_ROOT/include/boost/stage/lib"   # boost/stage/lib contain .so files

# Parse arguments
for option
do
    case $option in

    -help | --help | -h)
      want_help=yes ;;

    -disable-download | --disable-download | -d)
      disable_download=yes ;;

    -disable-bundled-compile | --disable-bundled-compile | -c)
      disable_compile=yes ;;

    -python-version=* | --python-version=*)
      PYTHON_VERSION=`expr "x$option" : "x-*python-version=\(.*\)"`
      PYTHON_HEADER_DIR="$ENVIRONMENT_ROOT/include/python${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1:2}m"
      ;;

    -envdir=* | --envdir=*)
      ENVIRONMENT_ROOT=`expr "x$option" : "x-*envdir=\(.*\)"`
      OPENCV_LIBRARY_DIR="$ENVIRONMENT_ROOT/lib"
      PYTHON_HEADER_DIR="$ENVIRONMENT_ROOT/include/python${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1:2}m"
      EIGEN_HEADER_DIR="$ENVIRONMENT_ROOT/include"
      BOOST_HEADER_DIR="$ENVIRONMENT_ROOT/include/boost"  # boost/boost contain header
      BOOST_LIBRARY_DIR="$ENVIRONMENT_ROOT/include/boost/stage/lib"   # boost/stage/lib contain .so files
      ;;

    -libopencvdir=* | --libopencvdir=*)
      OPENCV_LIBRARY_DIR=`expr "x$option" : "x-*libopencvdir=\(.*\)"`
      ;;

    -python-header-dir=* | --python-header-dir=*)
      PYTHON_HEADER_DIR=`expr "x$option" : "x-*python-header-dir=\(.*\)"`
      ;;

    -eigen-header-dir=* | --eigen-header-dir=*)
      EIGEN_HEADER_DIR=`expr "x$option" : "x-*eigen-header-dir=\(.*\)"`
      ;;

    -boost-header-dir=* | --boost-header-dir=*)
      BOOST_HEADER_DIR=`expr "x$option" : "x-*boost-header-dir=\(.*\)"`
      ;;

    -libboostdir=* | --libboostdir=*)
      BOOST_LIBRARY_DIR=`expr "x$option" : "x-*libboostdir=\(.*\)"`
      ;;
    esac
done

if test "x$want_help" = xyes; then
    cat <<EOF
./install.sh install bundled on your machine for you

Usage: $0 [OPTION]...

Defaults for the options are specified in brackets.

Configuration:
  -h, --help                    display this help and exit
  --disable-download            download boost and Eigen library to ./libs
  --disable-bundled-compile     compile MeshHomo into asap.so, meshhomo.so, and meshwarp.so on your machine
  --python-header-dir           specify the root of Python header files
                                [default: $ENVIRONMENT_ROOT/include/python${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1:2}m]
  --python-version=XY           specify the Python version as XY
                                [default: 37]
  --envdir                      specify virtual environment root directory
                                [default: $HOME/.conda]
  --libopencvdir                specify the directory that contains opencv library directory
                                [default: $ENVIRONMENT_ROOT/lib]
  --eigen-header-dir            specify the directory that contains Eigen header directory
                                [default: $ENVIRONMENT_ROOT/include]
  --boost-header-dir            specify the directory that contains boost header directory
                                [default: $ENVIRONMENT_ROOT/include/boost]
  --libboostdir                 specify the directory that contains all the liboost_.so libraries
                                [default: $ENVIRONMENT_ROOT/include/boost/stage/lib]
EOF
exit
fi

if test "x$disable_download" = xyes; then
    echo "Disabled Download"
else
    conda install -c conda-forge opencv==3.4.9 numpy -y
    echo "downloading"
    # Download Eigen and boost
    if [[ ! -f boost_1_72_0.tar.bz2 ]] ; then
        wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.bz2 || exit 1
    fi
    if [[ ! -f eigen-3.3.7.tar.bz2 ]] ; then
        wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 || exit 1
    fi
    mkdir -p libs
    tar -xf eigen-3.3.7.tar.bz2 -C libs/
    tar -xf boost_1_72_0.tar.bz2 -C libs/
    mv libs/eigen-3.3.7/Eigen libs/
    rm -r libs/eigen-3.3.7
    mv libs/boost_1_72_0 libs/boost
    rm eigen-3.3.7.tar.bz2
    rm boost_1_72_0.tar.bz2

    # move library to their place
    echo "copying Eigen to $EIGEN_HEADER_DIR/Eigen"
    mkir -p $EIGEN_HEADER_DIR/Eigen
    rsync -azP libs/Eigen/ $EIGEN_HEADER_DIR/Eigen
    echo "copying boost to $BOOST_HEADER_DIR"
    mkdir -p $BOOST_HEADER_DIR
    rsync -azP libs/boost/ $BOOST_HEADER_DIR

    # compile boost
    echo "Compile boost"
    PROGRAMDIR=`pwd`
    cd $BOOST_HEADER_DIR
    ./bootstrap.sh --with-python="$(which python)" --with-libraries=all
    export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PYTHON_HEADER_DIR"    # boost can't find pyconfig.h, this will help
    ./b2 -a
    export CPLUS_INCLUDE_PATH=""
    cd $PROGRAMDIR
fi

if test "x$disable_compile" = xyes; then
    echo "Disabled Compilation of Bundled shared libraries (.so)"
else
    export LD_LIBRARY_PATH=$BOOST_LIBRARY_DIR:$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PYTHON_HEADER_DIR"
    ./compile.sh $ENVIRONMENT_ROOT $OPENCV_LIBRARY_DIR $PYTHON_VERSION $PYTHON_HEADER_DIR $EIGEN_HEADER_DIR $BOOST_HEADER_DIR $BOOST_LIBRARY_DIR
    echo "compiling"
fi

rm -rf libs

python3 setup.py install

cat <<EOF
    Please add BOOST_LIBRARY_DIR, and OPENCV_LIBRARY_DIR to your LD_LIBRARY_PATH.

    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$OPENCV_LIBRARY_DIR:$BOOST_LIBRARY_DIR
EOF


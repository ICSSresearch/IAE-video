# compile the hello program with spaces instead of Tabs
# the compiler to use
CC = g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings

CFLAGS  = -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto
CFLAGS_WITH_PYTHON = -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -I/usr/include/python3.5m

#files to link:
LFLAGS = -lboost_python35 -lboost_numpy35

#build
BUILD_DIR = build
DLL_DIR = ../..

meshhomo: Asap Quad Mesh MeshHomo meshfinal
Asap: Asap.cpp ; $(CC) $(CFLAGS_WITH_PYTHON) -c Asap.cpp -o $(BUILD_DIR)/Asap.o
Quad: Quad.cpp ; $(CC) $(CFLAGS) -c Quad.cpp -o $(BUILD_DIR)/Quad.o
Mesh: Mesh.cpp ; $(CC) $(CFLAGS) -c Mesh.cpp -o $(BUILD_DIR)/Mesh.o
MeshHomo: MeshHomo.cpp ; $(CC) $(CFLAGS_WITH_PYTHON) -c MeshHomo.cpp -o $(BUILD_DIR)/MeshHomo.o
meshfinal: $(BUILD_DIR)/Asap.o $(BUILD_DIR)/Quad.o $(BUILD_DIR)/Mesh.o $(BUILD_DIR)/MeshHomo.o; $(CC) $(CFLAGS) -o $(DLL_DIR)/meshhomo.so $(BUILD_DIR)/Mesh.o $(BUILD_DIR)/Asap.o $(BUILD_DIR)/Quad.o $(LFLAGS)


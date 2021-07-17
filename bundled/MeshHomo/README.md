# To compile -O2 to be safer
# compile one by one and link at the end - faster for two executable
cd MeshHomo/
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -I/usr/include/python3.5m -c Asap.cpp -o build/Asap.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -c Mesh.cpp -o build/Mesh.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -I/usr/include/python3.5m -c Np2Mat.cpp -o build/Np2Mat.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -c Quad.cpp -o build/Quad.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -I/usr/include/python3.5m -c MeshHomo.cpp -o build/MeshHomo.o
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -fopenmp -I/usr/include/python3.5m -c MeshWarp.cpp -o build/MeshWarp.o
cd build/
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -o ../../meshhomo.so Asap.o Mesh.o Quad.o MeshHomo.o -lboost_python35 -lboost_numpy35
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -fopenmp -o ../../meshwarp.so MeshWarp.o Asap.o Mesh.o Quad.o Np2Mat.o -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_python35 -lboost_numpy35
g++ -std=c++11 -O3 -ffast-math -DNDEBUG -march=native -fPIC -shared -flto -o ../../asap.so Asap.o Mesh.o Quad.o -lboost_python35 -lboost_numpy35
cd ../../

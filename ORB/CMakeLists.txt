cmake_minimum_required(VERSION 2.8)
project( mySIFT )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
ADD_COMPILE_OPTIONS(-g)
add_executable(ORB_TEST oFAST.cpp rBRIEF.cpp ORBclass.cpp ORB.cpp)
target_link_libraries(ORB_TEST ${OpenCV_LIBS} )


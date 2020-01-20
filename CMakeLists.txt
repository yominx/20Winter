cmake_minimum_required(VERSION 2.8)
project( mySIFT )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
ADD_COMPILE_OPTIONS(-g -Wall)
add_executable( mySIFT mySIFT.cpp )
target_link_libraries( mySIFT ${OpenCV_LIBS} )


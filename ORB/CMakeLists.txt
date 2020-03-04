

#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(ORB_TEST)

add_executable(ORB_TEST matcher.cpp oFAST.cpp rBRIEF.cpp ORBclass.cpp ORB.cpp)
ADD_COMPILE_OPTIONS(-g)
set_property(TARGET ORB_TEST PROPERTY CXX_STANDARD 11)
target_link_libraries(ORB_TEST ${DEPENDENCIES})
set_target_properties (ORB_TEST PROPERTIES
	FOLDER "Examples/OpenCV"
)
set(CMAKE_BUILD_TYPE=Release)

install(
	TARGETS
	ORB_TEST
	RUNTIME DESTINATION
	${CMAKE_INSTALL_PREFIX}/bin
)

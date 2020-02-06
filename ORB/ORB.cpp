#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <assert.h> 
#include "oFAST.h"
#include "ORBclass.h"
#include "rBRIEF.h"

#define N_LEVELS 8
#define SCALE_FACTOR 1.2f
#define BORDER 15
using namespace cv;
using namespace std;


int main(int argc, char * argv[]) try
{
    cv::Mat *imgPyr[N_LEVELS];
    for(int i=0;i<N_LEVELS;i++)
        imgPyr[i] = new Mat(Size(2000,1080),CV_32FC1); 

    std::vector<Feature> allFeatureList;
    std::vector<FingerPrint> FPlist;
    int fastThres = 20; //Default threshold for FAST
    if(argv[1]!=NULL)
        fastThres = atoi(argv[1]);
    float factorList[N_LEVELS];
    factorList[0] = 1;
    for(int i=1; i<N_LEVELS;i++)
        factorList[i] = factorList[i-1] * SCALE_FACTOR;
    

    char window_name[] = "ORB test";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    rs2::colorizer color_map;
    rs2::pipeline pipe;
    pipe.start();
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::frame  rgbImg = data.get_color_frame();
        const int w = rgbImg.as<rs2::video_frame>().get_width();
        const int h = rgbImg.as<rs2::video_frame>().get_height();
        Mat image(Size(w, h), CV_8UC3, (void*)rgbImg.get_data(), Mat::AUTO_STEP);
        cv::cvtColor(image, image, CV_BGR2GRAY);

        //Make Img Pyramid.
        for(int i=0;i<N_LEVELS;i++){
            cv::Mat outImgi;
            cv::Size newSize = cv::Size((int)((image.cols)/factorList[i]),
                                        (int)((image.rows)/factorList[i]));
            cv::resize(image,*(imgPyr[i]),newSize,0,0,INTER_LINEAR_EXACT);
            copyMakeBorder(*(imgPyr[i]), *(imgPyr[i]), BORDER, BORDER, BORDER, BORDER, BORDER_REFLECT_101+BORDER_ISOLATED);
        }

        // oFAST operation.
        allFeatureList.clear();
        for(int level=0;level<N_LEVELS;level++){
    	    oFAST fast = oFAST();
            int x = imgPyr[level]->cols, y=imgPyr[level]->rows;
            fast.findFeature(imgPyr[level], factorList[level], level, BORDER,fastThres);
            if(!level) imshow(window_name,fast.featureImg());
        	std::copy(fast.Featurelist.begin(), fast.Featurelist.end(), std::back_inserter(allFeatureList));
        }

        // rBRIEF operation.
        FPlist.clear();
        rBRIEF(imgPyr,allFeatureList, FPlist);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

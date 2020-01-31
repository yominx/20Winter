// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <math.h>
#include <stdio.h>
#include <unistd.h>
//void salt(Mat& mat, int count);
using namespace cv;
using namespace std;

#define OCT_NUM  3
#define BLUR_NUM 3
void makeOctave(Mat& mat, Size size);
void click_blur(Mat mat);

const auto window_name = "SIFT test";


int main(int argc, char * argv[]) try
{
    rs2::colorizer color_map;
    rs2::pipeline pipe;
    pipe.start();

    int frame = 1;
    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0 && frame)
    {
        frame++;
        //if(frame<99) continue;

        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        //rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame  rgbImg = data.get_color_frame();
        // Query frame size (width and height)
        //const int w = depth.as<rs2::video_frame>().get_width();
        //const int h = depth.as<rs2::video_frame>().get_height();
        const int w = rgbImg.as<rs2::video_frame>().get_width();
        const int h = rgbImg.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        Mat image(Size(w, h), CV_8UC3, (void*)rgbImg.get_data(), Mat::AUTO_STEP);
    	makeOctave(image, Size(w,h));

        //click_blur(image);
        //imshow(window_name, image);
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



void makeOctave(Mat& mat, Size size){
    cv::Mat imgGray, temp, bigImg;
    cv::Mat octave[OCT_NUM][BLUR_NUM], dogList[OCT_NUM][BLUR_NUM-1]; // octave[number][blur_number]
    double initSigma = 1.0/sqrt(2.0f);
    double sigmaF = sqrt(2.0f);
    double curSigma;
 
    std::cout << "!! Generating scale space. !!" << endl;

    cv::cvtColor(mat, imgGray, CV_BGR2GRAY);
    imshow(window_name, imgGray);
    cv::pyrUp(imgGray, bigImg); 
    GaussianBlur(bigImg, octave[0][0], Size(0,0), initSigma, initSigma);

 
    std::cout << "Allocating memory." << endl;
    for(int i = 0; i < OCT_NUM; i++){
        double factor = 2/pow(2,i);
        int newH = (int)(size.height * factor);
        int newW = (int)(size.width  * factor);

        for(int j = 0; j < BLUR_NUM; j++){
            Mat sizeMat(newH, newW, CV_32FC1,Scalar(0));
            octave[i][j] = sizeMat;
            }
        }

    std::cout << "Making image pyramid." << endl;
    for(int i = 0; i < OCT_NUM - 1; i++){
        cv::pyrDown(octave[i][0], octave[i+1][0]); 
        }

    std::cout << "Take gaussian blurring." << endl;
    for(int i = 0; i < OCT_NUM; i++){
        curSigma = initSigma;
        for(int j = 1; j < BLUR_NUM; j++){
            GaussianBlur(octave[i][0], octave[i][j],Size(0,0),curSigma,curSigma);
            curSigma *=sigmaF;
            }
        }

    std::cout << "Computing DoG." << endl;
    for(int i = 0; i < OCT_NUM; i++){
        for(int j = 0; j < BLUR_NUM-1; j++){
            cv::subtract(octave[i][j], octave[i][j+1], dogList[i][j]);
            }
        }

    sleep(1);
    }

void click_blur(Mat mat){
    cv::Mat copyImg = mat.clone();
    int time = 0;
    imshow(window_name, copyImg);
    sleep(1);
    GaussianBlur(copyImg, copyImg, Size(3,3), 3, 3);
    imshow(window_name, copyImg);
    sleep(1);
    GaussianBlur(copyImg, copyImg, Size(3,3), 3, 3);
    imshow(window_name, copyImg);
    while(true){
        sleep(5);
        GaussianBlur(copyImg, copyImg, Size(3,3), 3, 3);
        imshow(window_name, copyImg);
        sleep(5);
        //getchar();
        std::cout << time++ << endl << endl;
    }
}

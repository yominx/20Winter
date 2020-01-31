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

using namespace cv;
using namespace std;

int main(int argc, char * argv[]){
	char* window_name = "ORB test";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    cv::Mat image = imread("./images/myimage.jpg", IMREAD_COLOR);
    if (image.empty()){
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    oFAST fast = oFAST();
    fast.findFeature(&image);
    imshow(window_name, image);
    cv::Mat temp;
    cv::cvtColor(image, temp, CV_BGR2GRAY);
    imshow(window_name, fast.featureImg());
    waitKey(0);

}
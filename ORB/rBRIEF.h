#ifndef RBRIEF_H
#define RBRIEF_H
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <assert.h> 
#include <chrono> 
#include "ORBclass.h"
#include "rBRIEF.h"

using namespace cv;
using namespace std;
void rBRIEF(cv::Mat** imgPyr, vector<Feature> *keyPlist, vector<FingerPrint> *FP);

#endif
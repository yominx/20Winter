#ifndef MATCHER_H
#define MATCHER_H

#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include "ORBclass.h"
#define MIN_HAMMING_DIST 1

#define BORDER 15
using namespace std;
using namespace cv;

class Matcher{
public:
	vector<Feature>		*Flist1,  *Flist2;
	vector<FingerPrint> *FPlist1, *FPlist2;
	vector<int> answerlist1, answerlist2;

	Matcher(vector<Feature> 	 *_Flist1, vector<Feature> 	  *_Flist2,
		 	vector<FingerPrint> *_FPlist1, vector<FingerPrint> *_FPlist2);
	void match();
	int datascore(FingerPrint a,FingerPrint b);
	void showImage(cv::Mat *image1, cv::Mat *image2, float factor);

};

#endif
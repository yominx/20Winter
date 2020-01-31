#ifndef OFAST_H
#define OFAST_H

#include "ORBclass.h"
class oFAST
{
public:
    cv::Mat *image, *imgGray;
    std::vector<Feature> Featurelist; 

	oFAST();
	cv::Mat* makeGray(cv::Mat* image);
	void	 get4Pix(cv::Mat* image, int x, int y,int* ret);
	int 	 available(int curpix, int* pix4,int thres);
	void	 get16Pix(cv::Mat* image, int x, int y,int* ret);
	bool 	 isFeature(int curpix, int* pixlist, int bright, int thres);
	double 	 getFeatureAngle(int* pixlist);
	void	 findFeature(cv::Mat* image,int thres=20);
	cv::Mat  featureImg();

};
#endif
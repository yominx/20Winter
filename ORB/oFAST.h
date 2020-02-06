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
	bool 	 isFeature(int curpix, int* pixlist, int thres);
	void	 findFeature(cv::Mat* image,float factorScale,int level,int border,int thres=50);
	cv::Mat  featureImg();
	std::vector<Feature> featureList();

};
#endif
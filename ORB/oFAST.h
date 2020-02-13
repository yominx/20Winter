#ifndef OFAST_H
#define OFAST_H

#include "ORBclass.h"
class oFAST
{
public:
    cv::Mat *image, *imgGray;
    std::vector<Feature> Featurelist; 
    int thres;

	oFAST();
	bool 	 isFeature(int* pixlist, uchar* lightdark, uchar* cur, float* magori);
	void	 findFeature(cv::Mat* image,int level,int border,int thres);
	std::vector<Feature> featureList();

};
#endif
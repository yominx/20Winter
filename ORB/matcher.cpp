#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include "ORBclass.h"
#include "matcher.h"



using namespace cv;
using namespace std;

Matcher::Matcher(vector<Feature> 	 *_Flist1, vector<Feature> 	   *_Flist2,
				 vector<FingerPrint> *_FPlist1,vector<FingerPrint> *_FPlist2){
	
	this->Flist1 = _Flist1;
	this->Flist2 = _Flist2;
	this->FPlist1 = _FPlist1;
	this->FPlist2 = _FPlist2;
}

void Matcher::match(){
	int size1 = FPlist1->size(), size2 = FPlist2->size();
	int dataset[size1][size2];
	for (int i = 0; i < size1; ++i)
		for (int j = 0; j < size2; ++j){
			dataset[i][j] = datascore((*FPlist1)[i],(*FPlist2)[j]);
			//cout << dataset[i][j];
		}
	
	cout << size1 << "    " << size2 << endl;
	

	if(size1<=size2){
		for (int i = 0; i < size1; ++i){
			int min_score = 1000, index = -1;
			for (int j = 0; j < size2; ++j){
				if(min_score > dataset[i][j]){
					min_score = dataset[i][j]; 
					index = j;
				}
			}
			//cout << index << endl;
			if (min_score <= MIN_HAMMING_DIST){
				answerlist1.push_back(i);
				answerlist2.push_back(index);
			}
		}
	}
	else{
		for (int i = 0; i < size2; ++i){
			int min_score = 1000, index = -1;
			for (int j = 0; j < size1; ++j){
				if(min_score > dataset[j][i]){
					min_score = dataset[j][i]; 
					index = j;
				}
			}
			//cout << min_score << endl;

			if (min_score <= MIN_HAMMING_DIST){
				answerlist1.push_back(index);
				answerlist2.push_back(i);
			}
		}
	}
}

int Matcher::datascore(FingerPrint a,FingerPrint b){
	int score = 0;
	int mask = 0x11111111, mask2 = 0xF;
	//cout << "ddd " << a.data[0] << endl;
	for (int i = 0; i < 8; i++){ //count bit
		int hamming = a.data[i] ^ b.data[i];
		for(int j = 1; j < 8; j++){
			if(hamming & 0x1)
				score++;
			hamming /= 2;
		}
	}
	return score;
}

void Matcher::showImage(cv::Mat *image1, cv::Mat *image2, float factor){
	float factorList[8];
    factorList[0] = 1;
    for(int i=1; i<8;i++)
        factorList[i] = factorList[i-1] * factor;


	cv::Size size1 = image1->size(), size2 = image2->size();
	cv::Mat mergeImg(Size(size1.width+size2.width, std::max(size1.height,size2.height)),image1->type());
	image1->copyTo(mergeImg(Rect(0,0,size1.width,size1.height)));
	image2->copyTo(mergeImg(Rect(size1.width,0,size2.width,size2.height)));
	int ansSize = answerlist1.size();
    //cout << ansSize << "       " << FPlist1->size() << endl;

	for(int i = 0; i < ansSize; i++){
		Feature a = (*Flist1)[answerlist1[i]];
		Feature b = (*Flist2)[answerlist2[i]];
		cv::Scalar color(255,255,255);
		cv::Point point1((a.x-BORDER)*factorList[a.level],(a.y-BORDER)*factorList[a.level]);
		cv::Point point2(size1.width+
			(b.x-BORDER)*factorList[b.level],(b.y-BORDER)*factorList[b.level]);
		//cout << point1 << "         " <<point2 << endl;

        cv::circle(mergeImg, point1,1,color,1,8);
        cv::circle(mergeImg, point2,1,color,1,8);
        cv::line(mergeImg,point1,point2,color,1,8);
	}
	/*
    for(int i=0; i<FPlist1->size(); i++){
        Feature a = Flist1[i];
        cv::circle(mergeImg, Point((a.x-BORDER)*factorList[a.level],(a.y-BORDER)*factorList[a.level]),
                                    1,Scalar(255,255,255),1,8);
    }

    for(int i=0; i<FPlist2->size(); i++){
        Feature a = Flist2[i];
        cv::circle(mergeImg, Point((a.x-BORDER)*factorList[a.level],(a.y-BORDER)*factorList[a.level]),
                                    1,Scalar(255,255,255),1,8);
    }
	*/
	char window_name[] = "Matched picture";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name,mergeImg);
    waitKey(0);
}

#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include "ORBclass.h"
#include "oFAST.h"

/*
        (-1,-3)(0,-3)(1,-3)
    (-2,-2)                 (2,-2)
(-3,-1)                         (3,-1)
(-3, 0)                         (3, 0)
(-3, 1)                         (3, 1)
    (-2,-2)                 (2, 2)
        (-1, 3)(0, 3)(1, 3)

*/

//This is FAST-12.
using namespace cv;
using namespace std;

oFAST::oFAST(){}

void oFAST::get4Pix(cv::Mat* image, int x, int y, int* ret){
    ret[0]  = image->at<uchar>(y-3,x  ); ret[1]  = image->at<uchar>(y  ,x+3);
    ret[2]  = image->at<uchar>(y+3,x  ); ret[3]  = image->at<uchar>(y  ,x-3);
}

int oFAST::available(int curpix, int* pix4,int thres){
    int bright  = curpix+thres;
    int dark    = curpix-thres;
    //cout << bright << "   " <<dark << endl;
    if((pix4[0]>bright && pix4[1]>bright && pix4[2]>bright)||
       (pix4[1]>bright && pix4[2]>bright && pix4[3]>bright)||
       (pix4[2]>bright && pix4[3]>bright && pix4[0]>bright)||
       (pix4[3]>bright && pix4[0]>bright && pix4[1]>bright)) return 1;
    if(dark<0) return 0;
    if((pix4[0]<dark && pix4[1]<dark && pix4[2]<dark)||
       (pix4[1]<dark && pix4[2]<dark && pix4[3]<dark)||
       (pix4[2]<dark && pix4[3]<dark && pix4[0]<dark)||
       (pix4[3]<dark && pix4[0]<dark && pix4[1]<dark)) return 2;
    return 0;

}

void oFAST::get16Pix(cv::Mat* image, int x, int y, int* ret){
    ret[0]  = image->at<uchar>(y-3,x  );     ret[1]  = image->at<uchar>(y-3,x+1);
    ret[2]  = image->at<uchar>(y-2,x+2);     ret[3]  = image->at<uchar>(y-1,x+3);
    ret[4]  = image->at<uchar>(y,  x+3);     ret[5]  = image->at<uchar>(y+1,x+3);
    ret[6]  = image->at<uchar>(y+2,x+2);     ret[7]  = image->at<uchar>(y+3,x+1);
    ret[8]  = image->at<uchar>(y+3,x  );     ret[9]  = image->at<uchar>(y+3,x-1);
    ret[10] = image->at<uchar>(y+2,x-2);     ret[11] = image->at<uchar>(y+1,x-3);
    ret[12] = image->at<uchar>(y,  x-3);     ret[13] = image->at<uchar>(y-1,x-3);
    ret[14] = image->at<uchar>(y-2,x-2);     ret[15] = image->at<uchar>(y-3,x-1);

}

bool oFAST::isFeature(int curpix, int* pixlist, int brightP, int thres){
    int comparelist[16],index;
    for(int i=0;i<16;i++)
        comparelist[i] = (curpix - pixlist[i]);

    
    for(int i=0;i<16;i++){
        if(comparelist[i]>0) brightP = 1;
        else                 brightP = 0;
        for(int k=0;;k++){
            index = (i+k)%16;
            if(k==9) {//cout << "this is feature";
                     return true;} //n=12 -> true
            if((brightP && comparelist[index]<thres) || (!brightP && comparelist[index]>-thres)) break;
        }
    }
    return false;
}


void oFAST::findFeature(cv::Mat* image,float factorScale, int level, int border, int thres){
    this->image = image;
    cv::Mat temp;
    cv::cvtColor(*(this->image), temp, CV_BGR2GRAY);
    this->imgGray = &temp;

    int pix4list[4], pix16list[16], curpix, brightP;

    //imshow("test",temp);
    //waitKey(0);

    int num = 0;
    for(int x=3; x<imgGray->size().width-3; x++){
        for(int y=3; y<imgGray->size().height-3; y++){
            curpix     = imgGray->at<uchar>(y,x);
            get4Pix(imgGray,x,y,pix4list);
            //brightP = available(curpix,pix4list,thres);
            //if(!brightP) continue;

            brightP = 1;
            //cout << endl << "current pixel : " << x<< " " << y<< endl ;
            get16Pix(imgGray,x,y,pix16list);
            if(isFeature(curpix, pix16list, brightP, thres)){
                num++;
                int weightX = pix16list[1] + 2*pix16list[2]  + 3*pix16list[3]  + 3*pix16list[4]  + 3*pix16list[5]  + 2*pix16list[6]  + pix16list[7]
                            -(pix16list[9] + 2*pix16list[10] + 3*pix16list[11] + 3*pix16list[12] + 3*pix16list[13] + 2*pix16list[14] + pix16list[15]);
                int weightY = pix16list[5] + 2*pix16list[6]  + 3*pix16list[7]  + 3*pix16list[8]  + 3*pix16list[9]  + 2*pix16list[10] + pix16list[11]
                            -(pix16list[13]+ 2*pix16list[14] + 3*pix16list[15] + 3*pix16list[0]  + 3*pix16list[1]  + 2*pix16list[2]  + pix16list[3]);
                this->Featurelist.push_back(Feature(level, x*factorScale+border, y*factorScale+border,
                                                    sqrt(weightX*weightX+weightY*weightY),
                                                        atan2((float)weightY,(float)weightX)));
            }
        }
    }
    cout << "# of feature is " << num << endl;
}

cv::Mat oFAST::featureImg(){
    cv::Mat showImg = this->image->clone();
    Feature key(0,0,0,0.0,0.0);
    cv::cvtColor(showImg,showImg,CV_BGR2GRAY);

    for(int i=0; i<this->Featurelist.size(); i++){
        key = this->Featurelist[i];
        cv::circle(showImg, Point(key.x, key.y),1,Scalar(255,255,255),1,8);
        //showImg.at<uchar>(key.y,key.x) = 255;
    }
    return showImg;
}

vector<Feature> oFAST::featureList(){
    return this->Featurelist;
}
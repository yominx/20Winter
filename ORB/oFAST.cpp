
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <chrono>
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

bool oFAST::isFeature(int* pixlist, uchar* lightdark, uchar* cur, float* magori){
    int thres = this->thres;
    int k;
    int curpix = cur[0];

    const uchar* tab = &lightdark[0] - (curpix) + 255;

    //cout << thres;
    int d = tab[cur[pixlist[0]]] | tab[cur[pixlist[8]]];

    if( d == 0 )    return false;

    d &= tab[cur[pixlist[2]]] | tab[cur[pixlist[10]]];
    d &= tab[cur[pixlist[4]]] | tab[cur[pixlist[12]]];
    d &= tab[cur[pixlist[6]]] | tab[cur[pixlist[14]]];

    if( d == 0 )    return false;

    d &= tab[cur[pixlist[1]]] | tab[cur[pixlist[9]]];
    d &= tab[cur[pixlist[3]]] | tab[cur[pixlist[11]]];
    d &= tab[cur[pixlist[5]]] | tab[cur[pixlist[13]]];
    d &= tab[cur[pixlist[7]]] | tab[cur[pixlist[15]]];


    if( d & 1 ){ //brighter than surround
        int vt = curpix + thres, count = 0;
        for( k = 0; k < 25; k++ ){
            int x = cur[pixlist[k]];
            if(x > vt){
                if( ++count > 8 ){
                    int weightX = cur[pixlist[1]] + 2*cur[pixlist[2]]  + 3*cur[pixlist[3]]  + 3*cur[pixlist[4]]  + 3*cur[pixlist[5]]  + 2*cur[pixlist[6]]  + cur[pixlist[7]]
                                -(cur[pixlist[9]] + 2*cur[pixlist[10]] + 3*cur[pixlist[11]] + 3*cur[pixlist[12]] + 3*cur[pixlist[13]] + 2*cur[pixlist[14]] + cur[pixlist[15]]);
                    int weightY = cur[pixlist[5]] + 2*cur[pixlist[6]]  + 3*cur[pixlist[7]]  + 3*cur[pixlist[8]]  + 3*cur[pixlist[9]]  + 2*cur[pixlist[10]] + cur[pixlist[11]]
                                -(cur[pixlist[13]]+ 2*cur[pixlist[14]] + 3*cur[pixlist[15]] + 3*cur[pixlist[0]]  + 3*cur[pixlist[1]]  + 2*cur[pixlist[2]]  + cur[pixlist[3]]);
                    magori[0] = (weightX*weightX+weightY*weightY);
                    magori[1] = atan2(weightY,weightX);

                    return true;
                }
            }
            else
                count = 0;
        }
    }

    if( d & 2 ){
        int vt = curpix - thres, count = 0;
        for( k = 0; k < 25; k++ ){
            int x = cur[pixlist[k]];
            if(x < vt){
                if( ++count > 8 ){
                    int weightX = cur[pixlist[1]] + 2*cur[pixlist[2]]  + 3*cur[pixlist[3]]  + 3*cur[pixlist[4]]  + 3*cur[pixlist[5]]  + 2*cur[pixlist[6]]  + cur[pixlist[7]]
                                -(cur[pixlist[9]] + 2*cur[pixlist[10]] + 3*cur[pixlist[11]] + 3*cur[pixlist[12]] + 3*cur[pixlist[13]] + 2*cur[pixlist[14]] + cur[pixlist[15]]);
                    int weightY = cur[pixlist[5]] + 2*cur[pixlist[6]]  + 3*cur[pixlist[7]]  + 3*cur[pixlist[8]]  + 3*cur[pixlist[9]]  + 2*cur[pixlist[10]] + cur[pixlist[11]]
                                -(cur[pixlist[13]]+ 2*cur[pixlist[14]] + 3*cur[pixlist[15]] + 3*cur[pixlist[0]]  + 3*cur[pixlist[1]]  + 2*cur[pixlist[2]]  + cur[pixlist[3]]);
                    magori[0] = (weightX*weightX+weightY*weightY);
                    magori[1] = atan2(weightY,weightX);

                    return true;
                }
            }
            else count = 0;
        }
    }




    return true;
}

static const int offsets[][2] ={
    {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
};


void oFAST::findFeature(cv::Mat* image, int level, int border, int thres){
    //cout<< "FAST!!" <<endl;
    this->thres = thres;
    cv::Mat imgGray = image->clone();
    if(image->type()!=0)
        cv::cvtColor(imgGray, imgGray, CV_BGR2GRAY);
    int curpix, brightP;
    float magori[2];


    int pixel[25], rowlength = (int)imgGray.step, k=0;
    for(;k < 16;k++)
        pixel[k] = offsets[k][0] + offsets[k][1] * rowlength;
    for(;k < 25;k++)
        pixel[k] = pixel[k - 16];

    uchar lightdark[512];
    for(int i = -255; i <= 255; i++ )
        lightdark[i+255] = (i > thres ? 1 : i < -thres ? 2 : 0);
    int height = imgGray.size().height, width = imgGray.size().width;
    
    int ncols = imgGray.cols;
    float mag[3][ncols],ori[3][ncols];
    memset(mag[0], 0, sizeof(mag[0]));    memset(mag[1], 0, sizeof(mag[1]));    memset(mag[2], 0, sizeof(mag[2]));
    memset(ori[0], 0, sizeof(ori[0]));    memset(ori[1], 0, sizeof(ori[1]));    memset(ori[2], 0, sizeof(ori[2]));
    for(int y=border+3; y<height-border-3; y++){
        uchar* cur = imgGray.ptr<uchar>(y);
        int pprev = (y+1)%3,prev = (y+2)%3,current = y%3; 

        for(int x=border+3; x<width-border-3; x++){
            mag[current][x] = 0; ori[current][x] = 0;
            if(isFeature(pixel,lightdark,&cur[x], magori))
                {mag[current][x] = magori[0], ori[current][x] = magori[1];} 
        }

        for(int x=border+3; x<width-border-3; x++){
            int point = mag[prev][x];
            bool max =  point > mag[pprev][x-1]   && point > mag[pprev][x]   && point > mag[pprev][x+1] &&
                        point > mag[prev][x-1]    &&                            point > mag[prev][x+1] && 
                        point > mag[current][x-1] && point > mag[current][x] && point > mag[current][x+1];
            if(max)  {this->Featurelist.push_back(Feature(level, x, y-1, mag[prev][x],ori[prev][x]));}
        }
    }

}

vector<Feature> oFAST::featureList(){
    return this->Featurelist;
}
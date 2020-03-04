#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include "oFAST.h"
#include "ORBclass.h"
#include "rBRIEF.h"
#include "matcher.h"

#define N_LEVELS 8
#define SCALE_FACTOR 1.2f
using namespace cv;
using namespace std;



int main(int argc, char * argv[])
{

    std::vector<Feature> allFeatureList;
    std::vector<FingerPrint> FPlist;
    int fastThres = 40; //Default threshold for FAST
    if(argv[1]!=NULL)
        fastThres = atoi(argv[1]);
    float factorList[N_LEVELS];
    factorList[0] = 1;
    for(int i=1; i<N_LEVELS;i++)
        factorList[i] = factorList[i-1] * SCALE_FACTOR;
    

    char window_name[] = "ORB test";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    cv::Mat img = imread("./images/myimage.jpg");
    if(img.empty()){
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    cv::Mat *imgPyr[N_LEVELS];
    for(int i=0;i<N_LEVELS;i++)
        imgPyr[i] = new Mat(img.size(),CV_32FC1); 

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    Mat image;
    cv::cvtColor(img, image, CV_BGR2GRAY);
    //cout << img.size() << endl;
    //Make Img Pyramid.
    for(int i=0;i<N_LEVELS;i++){
        cv::Mat outImgi;
        cv::Size newSize = cv::Size((int)((image.cols)/factorList[i]),
                                    (int)((image.rows)/factorList[i]));
        cv::resize(image,*(imgPyr[i]),newSize,0,0,INTER_LINEAR_EXACT);
        copyMakeBorder(*(imgPyr[i]), *(imgPyr[i]), BORDER, BORDER, BORDER, BORDER, BORDER_REFLECT_101+BORDER_ISOLATED);
    }
    std::chrono::duration<double> sec2 = std::chrono::system_clock::now() - start;

    // oFAST operation.
    allFeatureList.clear();
    int featureNum = 1000;
    float minimum = (float)featureNum*(SCALE_FACTOR-1.f)/ (float)pow(SCALE_FACTOR,N_LEVELS);
    for(int level=0;level<N_LEVELS;level++){
        int nfeaturelvl = (int)(minimum * pow(SCALE_FACTOR,(N_LEVELS-level)));
        oFAST fast = oFAST();
        int x = imgPyr[level]->cols, y=imgPyr[level]->rows;
        fast.findFeature(imgPyr[level], level, BORDER,fastThres);
        sort(fast.Featurelist.begin(), fast.Featurelist.end(), &Feature::compare);
        fast.Featurelist.size() >= nfeaturelvl ? std::copy(fast.Featurelist.begin(), fast.Featurelist.begin() 
                                                                    + nfeaturelvl, std::back_inserter(allFeatureList)):
                                                 std::copy(fast.Featurelist.begin(), fast.Featurelist.end(), 
                                                                                   std::back_inserter(allFeatureList));

    }
        
    std::chrono::duration<double> sec1 = std::chrono::system_clock::now() - start;
    // rBRIEF operation.
    FPlist.clear();
    for(int i=0;i<N_LEVELS;i++)
        GaussianBlur(*imgPyr[i], *imgPyr[i], Size(7, 7), 2, 2, BORDER_REFLECT_101);//Blur Img
    rBRIEF(imgPyr,&allFeatureList, &FPlist);
///////////////////////////////////////////////////////////////////

    cv::Mat showImg = img.clone();
    cvtColor(showImg,showImg,CV_BGR2RGB);
    Feature key(0,0,0,0.0,0.0);
    for(int i=0; i<allFeatureList.size(); i++){
        key = allFeatureList[i];
        cv::circle(showImg, Point((key.x-BORDER)*factorList[key.level],
                                  (key.y-BORDER)*factorList[key.level]),
                                    1,Scalar(255,255,255),1,8);
    }
    imshow(window_name,showImg);

    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    cout << "Pyramid time : " << 1000*sec2.count()<<" ms" <<endl;
    cout << "FAST time : " << 1000*(sec1.count() - sec2.count())<<" ms" <<endl;
    cout << "BRIEF time : " << 1000*(sec.count() - sec1.count())<<" ms" <<endl;
    cout << "-------------------------------"  << endl;
    cout << "Total : " << 1000*sec.count()<<" ms"<< endl;
    cout << "Frequency : " << 1.f/sec.count()<<" Hz" <<endl;
    cout << "-------------------------------" << endl <<
     "Number of the feature is " << allFeatureList.size() << endl<<endl<<endl;
    //waitKey(0);
















    std::vector<Feature> allFeatureList1;
    std::vector<FingerPrint> FPlist1;
    int fastThres1 = 40; //Default threshold for FAST
    if(argv[1]!=NULL)
        fastThres1 = atoi(argv[1]);
    float factorList1[N_LEVELS];
    factorList1[0] = 1;
    for(int i=1; i<N_LEVELS;i++)
        factorList1[i] = factorList1[i-1] * SCALE_FACTOR;
    

    char window_name1[] = "ORB test2";
    namedWindow(window_name1, WINDOW_AUTOSIZE);

    cv::Mat img1 = imread("./images/myimage2.jpg");
    if(img1.empty()){
        cout << "Could not open or find the image1" << endl;
        return -1;
    }

    cv::Mat *img1Pyr[N_LEVELS];
    for(int i=0;i<N_LEVELS;i++)
        img1Pyr[i] = new Mat(img1.size(),CV_32FC1); 

    std::chrono::system_clock::time_point start1 = std::chrono::system_clock::now();
    Mat image1;
    cv::cvtColor(img1, image1, CV_BGR2GRAY);
    //cout << img1.size() << endl;
    //Make Img1 Pyramid.
    for(int i=0;i<N_LEVELS;i++){
        cv::Mat outImg1i;
        cv::Size newSize = cv::Size((int)((image1.cols)/factorList1[i]),
                                    (int)((image1.rows)/factorList1[i]));
        cv::resize(image1,*(img1Pyr[i]),newSize,0,0,INTER_LINEAR_EXACT);
        copyMakeBorder(*(img1Pyr[i]), *(img1Pyr[i]), BORDER, BORDER, BORDER, BORDER, BORDER_REFLECT_101+BORDER_ISOLATED);
    }
    std::chrono::duration<double> secsec2 = std::chrono::system_clock::now() - start1;

    // oFAST operation.
    allFeatureList1.clear();
    //float minimum = (float)featureNum*(SCALE_FACTOR-1.f)/ (float)pow(SCALE_FACTOR,N_LEVELS);
    for(int level=0;level<N_LEVELS;level++){
        int nfeaturelvl = (int)(minimum * pow(SCALE_FACTOR,(N_LEVELS-level)));
        oFAST fast = oFAST();
        int x = img1Pyr[level]->cols, y=img1Pyr[level]->rows;
        fast.findFeature(img1Pyr[level], level, BORDER,fastThres1);
        sort(fast.Featurelist.begin(), fast.Featurelist.end(), &Feature::compare);
        fast.Featurelist.size() >= nfeaturelvl ? std::copy(fast.Featurelist.begin(), fast.Featurelist.begin() 
                                                                    + nfeaturelvl, std::back_inserter(allFeatureList1)):
                                                 std::copy(fast.Featurelist.begin(), fast.Featurelist.end(), 
                                                                                   std::back_inserter(allFeatureList1));

    }
        
    std::chrono::duration<double> secsec1 = std::chrono::system_clock::now() - start1;
    // rBRIEF operation.
    FPlist1.clear();
    for(int i=0;i<N_LEVELS;i++)
        GaussianBlur(*img1Pyr[i], *img1Pyr[i], Size(7, 7), 2, 2, BORDER_REFLECT_101);//Blur Img1
    rBRIEF(img1Pyr,&allFeatureList1, &FPlist1);
///////////////////////////////////////////////////////////////////
    cout << FPlist1[1].data[0] << "\n\n\n\n\n"<< endl;
    cv::Mat showImg1 = img1.clone();
    cvtColor(showImg1,showImg1,CV_BGR2RGB);
    Feature key2(0,0,0,0.0,0.0);
    for(int i=0; i<allFeatureList1.size(); i++){
        key2 = allFeatureList1[i];
        cv::circle(showImg1, Point((key2.x-BORDER)*factorList1[key2.level],
                                  (key2.y-BORDER)*factorList1[key2.level]),
                                    1,Scalar(255,255,255),1,8);
    }
    imshow(window_name1,showImg1);

    std::chrono::duration<double> secsec = std::chrono::system_clock::now() - start1;
    cout << "Pyramid time : " << 1000*secsec2.count()<<" ms" <<endl;
    cout << "FAST time : " << 1000*(secsec1.count() - secsec2.count())<<" ms" <<endl;
    cout << "BRIEF time : " << 1000*(secsec.count() - secsec1.count())<<" ms" <<endl;
    cout << "-------------------------------"  << endl;
    cout << "Total : " << 1000*secsec.count()<<" ms"<< endl;
    cout << "Frequency : " << 1.f/secsec.count()<<" Hz" <<endl;
    cout << "-------------------------------" << endl <<
     "Number of the feature is " << allFeatureList1.size() << endl<<endl<<endl;
    //waitKey(0);







    Matcher BFMatch(&allFeatureList,&allFeatureList1,&FPlist,&FPlist1);
    BFMatch.match();
    BFMatch.showImage(&img,&img1,SCALE_FACTOR);

    return 0;
}

// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

#define OCT_NUM         5
#define BLUR_NUM        5
#define MIN_BRIGHT      6
#define MIN_CURVE       10
#define NUM_BINS        36
#define MAX_KERNEL_SIZE 20
#define M_PI            3.14159265358979323846
#define FEATURE_WINDOW_SIZE 16
#define CUT_OFF         0.001
#define DESC_NUM_BINS   8
#define FVSIZE          128
#define FV_THRESHOLD    0.2


void makeOctave(Mat& mat, Size size);

void DetectExtrema();
    void isExtrema(Mat* up, Mat* target, Mat* down, int x, int y);

void AssignOrientations();
    void makeMagAndOri(Mat& mag, Mat& ori, int i, int j);
    int  GetKernelSize(double sigma);
    int chkKeyPoint(int i, int xi, int yi);
    void saveKeyP(Mat& imgWeight, Mat& magnitude, Mat& orientation,
                                     int i, int j, int scale, int sizeK);
    //void saveKeyP(Mat& imgWeight, int width, int height, int i, int j);

void ExtractKeypointDescriptors();
    void BuildInterpolatedGaussianTable(cv::Mat* ret, int size, double sigma);
    double   gaussian2D(double x, double y, double sigma);



class Descriptor{
    public:
        int               xi, yi;     // The location
        vector<double>   fv;         // The feature vector

        Descriptor(){}

        Descriptor(int x, int y, vector<double> f){
            xi = x;
            yi = y;
            fv = f;
        }
    };
class Keypoint{
    public:
        int             xi;
        int             yi;     // Location
        vector<int>     mag;    // The list of magnitudes at this point
        vector<int>     orien;  // The list of orientations detected
        int             scale;  // The scale where this was detected

        Keypoint() {}
        Keypoint(int x, int y) { xi=x; yi=y; }
        Keypoint(int x, int y, vector<int> m, vector<int> o,int s){
            xi = x;
            yi = y;
            mag = m;
            orien = o;
            scale = s;
        }
    };

char window_name[] = "SIFT test";
vector<Keypoint*> keyPoints;
vector<Descriptor*> keyDescs;
double absSigma[OCT_NUM];

////////////////////// Allocate Octave, DoG, ExtImg  //////////

int temp = 0;
cv::Mat *** octave, ***dogList, ***extImg;

void alloc(){
    octave  = new cv::Mat** [OCT_NUM];
    dogList = new cv::Mat** [OCT_NUM];
    extImg  = new cv::Mat** [OCT_NUM];
    for(temp = 0; temp < OCT_NUM; temp++)
        octave[temp]  = new Mat*[BLUR_NUM];
    for(temp = 0; temp<OCT_NUM; temp++)
        dogList[temp] = new Mat*[BLUR_NUM-1];
    for(temp = 0; temp<OCT_NUM; temp++)
        extImg[temp]  = new Mat*[BLUR_NUM-3];
}

int main(int argc, char * argv[]) {

    namedWindow(window_name, WINDOW_AUTOSIZE);
    cv::Mat image = imread("./SIFT/myimage.jpg", IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    cout << "Size of the picture is" << image.size() << endl;
    keyPoints.clear();
    alloc();
    makeOctave(image, image.size());
    DetectExtrema();
    AssignOrientations();
    ExtractKeypointDescriptors();

    waitKey(0);
    
    return 0;
    }

void makeOctave(Mat& mat, Size size){
    cv::Mat imgGray, temp, bigImg;

    double initSigma = 1.0/sqrt(2.0f);
    double sigmaF = sqrt(2.0f);
    double curSigma;
 
    std::cout << "!! making Octave !!" << endl;

    cv::cvtColor(mat, imgGray, CV_BGR2GRAY);
    //imshow(window_name, imgGray);
    //cv::pyrUp(imgGray, bigImg); 
    //GaussianBlur(bigImg, octave[0][0], Size(0,0), initSigma, initSigma);
    //GaussianBlur(bigImg, octave[0][0], Size(0,0), initSigma, initSigma);
    octave[0][0] = &imgGray;
    // Making pyramid
    std::cout << "Making img pyramid" << endl;

    for(int i = 0; i < OCT_NUM-1; i++){
        cv::pyrDown(*(octave[i][0]), *(octave[i+1][0])  );
    }

    std::cout << "Gaussian blurring! Make all octave member." << endl;
    for(int i = 0; i < OCT_NUM; i++){
        //std::cout << "showwwwwwwwwww" << endl;

        curSigma = initSigma;

        double factor = 2/pow(2,i);
        int newH = (int)(size.height * factor);
        int newW = (int)(size.width  * factor);

        for(int j = 1; j < BLUR_NUM; j++){
            //Mat sizeMat(newH, newW, CV_32FC1, Scalar(0));
          //  cout << i << "     " << j << endl;
            *octave[i][j] = octave[i][0]->clone();
            GaussianBlur(*octave[i][j], *octave[i][j],Size(0,0),curSigma,curSigma);
            absSigma[j] = curSigma; 

            curSigma *=sigmaF;
            }
        }

    //std::cout << "show octave 14" << endl;
    //imshow(window_name, octave[1][4]);waitKey(0);

    std::cout << "Computing DoG." << endl;
    for(int i = 0; i < OCT_NUM; i++){
        for(int j = 0; j < BLUR_NUM-1; j++){
            cv::subtract(*octave[i][j], *octave[i][j+1], *dogList[i][j]);
            }
        }
    std::cout << "OK." << endl;
    sleep(0.1);
    std::cout << "show" << endl;
    //imshow(window_name, dogList[1][2]);waitKey(0);

    }

void DetectExtrema(){
    cout << endl << "!! Detecting Extrema !!" << endl;
    int i, j;
    cv::Mat *up, *target, *down;
    for (i=0;i<OCT_NUM;i++){
        for (j=0;j<BLUR_NUM-3;j++){
            up = dogList[i][j]; target = dogList[i][j+1]; down = dogList[i][j+2];
            cout << i << "        " << j<< endl;
            isExtrema(up,target,down,i,j);
        }
    }
    }

void isExtrema(Mat* up, Mat* target, Mat* down, int x, int y){
    Size size = target->size();
    extImg[x][y] = new Mat(size,CV_32FC1);
    uchar curPix;
    double dxx, dyy, dxy, trH, detH, curvature_ratio;
    cout << size ;
    int number = 0,dark = 0, edge = 0;
    for(int i = 1; i < size.width-1 ;i++){
        for(int j = 1; j < size.height-1 ; j++){
            curPix = target->at<uchar>(j,i);
            if ((up->at<uchar>(j-1,i-1)    > curPix && up->at<uchar>(j,i-1)    > curPix && up->at<uchar>(j+1,i-1)     > curPix &&
                up->at<uchar>(j-1,i  )     > curPix && up->at<uchar>(j,i)      > curPix && up->at<uchar>(j+1,i  )     > curPix &&
                up->at<uchar>(j-1,i+1)     > curPix && up->at<uchar>(j,i+1)    > curPix && up->at<uchar>(j+1,i+1)     > curPix &&
                target->at<uchar>(j-1,i-1) > curPix && target->at<uchar>(j,i-1)> curPix && target->at<uchar>(j+1,i-1) > curPix &&
                target->at<uchar>(j-1,i  ) > curPix &&                                    target->at<uchar>(j+1,i  ) > curPix &&
                target->at<uchar>(j-1,i+1) > curPix && target->at<uchar>(j,i+1)> curPix && target->at<uchar>(j+1,i+1) > curPix &&
                down->at<uchar>(j-1,i-1)   > curPix && down->at<uchar>(j,i-1)  > curPix && down->at<uchar>(j+1,i-1)   > curPix &&
                down->at<uchar>(j-1,i  )   > curPix && down->at<uchar>(j,i)    > curPix && down->at<uchar>(j+1,i  )   > curPix &&
                down->at<uchar>(j-1,i+1)   > curPix && down->at<uchar>(j,i+1)  > curPix && down->at<uchar>(j+1,i+1)   > curPix 
                )||(
                up->at<uchar>(j-1,i-1)     < curPix && up->at<uchar>(j,i-1)    < curPix && up->at<uchar>(j+1,i-1)     < curPix &&
                up->at<uchar>(j-1,i  )     < curPix && up->at<uchar>(j,i)      < curPix && up->at<uchar>(j+1,i  )     < curPix &&
                up->at<uchar>(j-1,i+1)     < curPix && up->at<uchar>(j,i+1)    < curPix && up->at<uchar>(j+1,i+1)     < curPix &&
                target->at<uchar>(j-1,i-1) < curPix && target->at<uchar>(j,i-1)< curPix && target->at<uchar>(j+1,i-1) < curPix &&
                target->at<uchar>(j-1,i  ) < curPix &&                                    target->at<uchar>(j+1,i  ) < curPix &&
                target->at<uchar>(j-1,i+1) < curPix && target->at<uchar>(j,i+1)< curPix && target->at<uchar>(j+1,i+1) < curPix &&
                down->at<uchar>(j-1,i-1)   < curPix && down->at<uchar>(j,i-1)  < curPix && down->at<uchar>(j+1,i-1)   < curPix &&
                down->at<uchar>(j-1,i  )   < curPix && down->at<uchar>(j,i)    < curPix && down->at<uchar>(j+1,i  )   < curPix &&
                down->at<uchar>(j-1,i+1)   < curPix && down->at<uchar>(j,i+1)  < curPix && down->at<uchar>(j+1,i+1)   < curPix 
                )){
                    dxx = (target->at<uchar>(j-1, i) + target->at<uchar>(j+1, i) -
                        2.0*target->at<uchar>(j, i));
                    dyy = (target->at<uchar>(j, i-1) + target->at<uchar>(j, i+1) -
                        2.0*target->at<uchar>(j, i));
                    dxy = (target->at<uchar>(j-1, i-1) + target->at<uchar>(j+1, i+1) -
                        target->at<uchar>(j+1, i-1) - target->at<uchar>(j-1, i+1)) / 4.0;
                    trH = dxx + dyy;
                    detH = dxx*dyy - dxy*dxy;
                    curvature_ratio = trH*trH/detH;


                if (curPix < MIN_BRIGHT){ //brightness check
                    //cout << "too dark.." << endl;
                    dark++;
                }
                else if( detH<0 || curvature_ratio > (double)(MIN_CURVE+1)*(MIN_CURVE+1)/MIN_CURVE){ //edge check
                    //cout << "Maybe it's edge.." << endl;
                    edge++;
                }
                else{
                    extImg[x][y]->at<uchar>(j,i) = 255;
                    number++;
                    }
                }
        }
    }

    //imshow(window_name, extImg[x][y]);
    printf("Found %d keypoints\n", number);
    printf("Rejected keypoints\ndark : %d\nedge : %d\n\n", dark,edge);
    //waitKey(0);
    }

void AssignOrientations(){
    cout << endl << "!! Assigning Orientations !!" << endl;
    int i,j;
    cv::Mat magnitude[OCT_NUM][BLUR_NUM-3],orientation[OCT_NUM][BLUR_NUM-3];


    for (int i=0;i<OCT_NUM;i++){
        for (int j=0;j<BLUR_NUM-3;j++){
            magnitude[i][j]   = Mat(octave[i][0]->size(),CV_32FC1);
            orientation[i][j] = Mat(octave[i][0]->size(),CV_32FC1);
            makeMagAndOri(magnitude[i][j], orientation[i][j], i, j+1);
            
            //cout << i << j << endl;
            //imshow(window_name, magnitude[i][j]);waitKey(0);
            //imshow(window_name, orientation[i][j]);waitKey(0);

            }
        }

    for(int i=0;i<OCT_NUM;i++){
        // Store current scale, width and height
        int scale = (int)pow(2.0, (double)i);
        int width = octave[i][0]->size().width;
        int height= octave[i][0]->size().height;

        // Go through all intervals in the current scale
        for(int j=0;j<BLUR_NUM-3;j++){
            double abs_sigma = absSigma[j];
            // This is used for magnitudes
            cv::Mat imgWeight(width, height, CV_32FC1);
            cv::Mat imgMask(width, height, CV_32FC1);
            int sizeK = GetKernelSize(1.5*abs_sigma);
            if (sizeK%2==0) sizeK++;
            GaussianBlur(magnitude[i][j], imgWeight, Size(sizeK,sizeK), 1.5*abs_sigma, 1.5*abs_sigma);
            //imshow(window_name,magnitude[i][j]);waitKey(0);
            //imshow(window_name,imgWeight);waitKey(0);
            saveKeyP(imgWeight, magnitude[i][j],orientation[i][j],i,j,scale,sizeK);

            cout << "Method : SaveKeyP completed  ij : " << i << j <<  endl << endl;        
        }
    }


    for (int i=0;i<OCT_NUM;i++){
        for (int j=0;j<BLUR_NUM-3;j++){
              magnitude[i][j].release();
            orientation[i][j].release();
            }
        }
    }
void makeMagAndOri(Mat& mag, Mat& ori, int i, int j){
    int norm255,xi,yi;
    for(xi=1;xi < octave[i][j]->size().width-1;xi++){
        for(yi=1;yi < octave[i][j]->size().height-1;yi++){
            // Calculate gradient
            double dx = octave[i][j]->at<uchar>(yi, xi+1)-octave[i][j]->at<uchar>(yi, xi-1);
            double dy = octave[i][j]->at<uchar>(yi+1, xi)-octave[i][j]->at<uchar>(yi-1, xi);
            // Store magnitude
            mag.at<uchar>(yi, xi) = sqrt(dx*dx + dy*dy);

            double angRadian;
            if (dx==0)
                dy>0 ? angRadian = 0.5 * M_PI : angRadian = -0.5 * M_PI;
            else 
                angRadian = atan2(dy, dx);

            norm255 = (int)(255.0 * (atan2(dy,dx) + M_PI) / 2.0/M_PI); 
            if (norm255 >= 256) norm255-=256; 
            if (norm255 < 0)   norm255+=256;

            ori.at<uchar>(yi, xi) = norm255;
            }
        }
    }

int GetKernelSize(double sigma){
    int i=0;
    for (;i<MAX_KERNEL_SIZE;i++)
        if (exp(-((double)(i*i))/(2.0*sigma*sigma)) < CUT_OFF)
            break;
    int size = 2*i-1;
    return size;
    }

int chkKeyPoint(int i, int xi, int yi){
    for(int j = 0; j < BLUR_NUM-3; j++)
        if (extImg[i][j]->at<uchar>(yi, xi)!=0)
            return 1;
    return 0;
}

void saveKeyP(Mat& imgWeight, Mat& magnitude, Mat& orientation, 
                                int i, int j, int scale, int sizeK){
    int hist_orient[NUM_BINS];
    int xi, yi, xk, yk, k, ori255;
    printf("Save Keypoints : current picture is, oct : %d, blur : %d.\n",i,j);
    int width  = magnitude.size().width;
    int height = magnitude.size().height;

    for(xi=0;xi<width ;xi++){
    for(yi=0;yi<height;yi++){
        // At the keypoint
    if(chkKeyPoint(i, xi, yi)){
        // Reset the histogram
        for(int kk=0; kk<NUM_BINS; kk++)
            hist_orient[kk]=0;
        // xk and tk : kernel x, y position, center is (0,0)
        for(xk = -sizeK; xk <= sizeK; xk++){
        for(yk = -sizeK; yk <= sizeK; yk++){
            
            // out of the picture : ignore 
            if(xi+xk<0 || xi+xk>=width || yi+yk<0 || yi+yk>=height)
                continue;
            double curOrient255 = orientation.at<uchar>(yi+yk, xi+xk);
            int orinum = (int)curOrient255 / (256/NUM_BINS);
            //cout << "weight : " << (int) imgWeight.at<uchar>(yi+yk, xi+xk) << endl;
            hist_orient[orinum] += (int) imgWeight.at<uchar>(yi+yk, xi+xk);
            //cout << "hist orient is " << hist_orient[(int)curOrient255 / (256/NUM_BINS)] << endl
            //     << "img Weight is " << imgWeight.at<uchar>(yi+yk, xi+xk) << endl;
            
            //imgMask.at<uchar>(yi+yk, xi+xk, 255);
            }}

        // We've computed the histogram. Now check for the maximum
        int max_peak = hist_orient[0];
        int max_peak_index = 0;

        /*
        cout << "__hist ori__"  << endl
             << hist_orient[0]   << "   " << hist_orient[1]  << "   " << hist_orient[2]  << endl
             << hist_orient[3]   << "   " << hist_orient[4]  << "   " << hist_orient[5]  << endl
             << hist_orient[6]   << "   " << hist_orient[7]  << "   " << hist_orient[8]  << endl
             << hist_orient[9]   << "   " << hist_orient[10] << "   " << hist_orient[11] << endl
             << hist_orient[12]  << "   " << hist_orient[13] << "   " << hist_orient[14] << endl
             << hist_orient[15]  << "   " << hist_orient[16] << "   " << hist_orient[17] << endl
             << hist_orient[18]  << "   " << hist_orient[19] << "   " << hist_orient[20] << endl
             << hist_orient[21]  << "   " << hist_orient[22] << "   " << hist_orient[23] << endl
             << hist_orient[24]  << "   " << hist_orient[25] << "   " << hist_orient[26] << endl
             << hist_orient[27]  << "   " << hist_orient[28] << "   " << hist_orient[29] << endl
             << hist_orient[30]  << "   " << hist_orient[31] << "   " << hist_orient[32] << endl   
             << hist_orient[33]  << "   " << hist_orient[34] << "   " << hist_orient[35] << endl
             << "____________"   << endl;
        */

        for(k=0;k<NUM_BINS;k++){
            if(hist_orient[k]>max_peak){
                max_peak = hist_orient[k]; max_peak_index = k;
            }}
        if(max_peak < 5) continue;

            // List of magnitudes and orientations at the current extrema
        vector<int> orien;
        vector<int> mag;


        for(k=0;k<NUM_BINS;k++){
            //cout << "ddd" << hist_orient[k] << endl << 0.8*max_peak << endl;
            if(hist_orient[k] > 0.8*max_peak){
        /*
                cout << endl << "index and maxpeak : " << k << "   " << max_peak << endl << endl;
                double x1 = k-1, x2 = k, x3 = k+1;
                double y1, y2 = hist_orient[k], y3;
            
                if(k==0){
                    y1 = hist_orient[NUM_BINS-1];
                    y3 = hist_orient[1];}
                else if(k==NUM_BINS-1){
                    y1 = hist_orient[NUM_BINS-1];
                    y3 = hist_orient[0];}
                else{
                    y1 = hist_orient[k-1];
                    y3 = hist_orient[k+1];}
            
                double b[3];
                cv::Mat X(3, 3, CV_32FC1), matInv(3, 3, CV_32FC1);

                // (Least-square method) for parabola :
                // parabola is constructed by near points, output b is parameters

                X.at<uchar>(0, 0)= x1*x1; X.at<uchar>(1, 0)= x1; X.at<uchar>(2, 0)= 1;
                X.at<uchar>(0, 1)= x2*x2; X.at<uchar>(1, 1)= x2; X.at<uchar>(2, 1)= 1;
                X.at<uchar>(0, 2)= x3*x3; X.at<uchar>(1, 2)= x3; X.at<uchar>(2, 2)= 1;

                matInv = X.inv();

                // inv(X) * y = parameters
                b[0] = matInv.at<uchar>(0, 0)*y1 + matInv.at<uchar>(1, 0)*y2 + matInv.at<uchar>(2, 0)*y3;
                b[1] = matInv.at<uchar>(0, 1)*y1 + matInv.at<uchar>(1, 1)*y2 + matInv.at<uchar>(2, 1)*y3;
                b[2] = matInv.at<uchar>(0, 2)*y1 + matInv.at<uchar>(1, 2)*y2 + matInv.at<uchar>(2, 2)*y3;

                //x0 is center z-position of the of the parabola
                double x0 = -b[1] / (2*b[0]);
                cout << "X is : " <<  endl << X << endl << "x0 is : " << x0 << endl;
                // Anomalous situation
                // if(fabs(x0) > 2*NUM_BINS)
                //    x0=x2;
            {
                while(x0 < 0)
                    x0 += NUM_BINS;
                while(x0 >= NUM_BINS)
                    x0-= NUM_BINS;
            }
                // Normalize it, convert 0~255
                ori255 = (int)( x0 * (256/NUM_BINS) );
                while (ori255 >= 256) ori255-=256; 
                while (ori255 < 0)    ori255+=256;

            */

                //cout << "Calc ori255" << "   " << i << "  " << j << "  " << k << "  " << endl;
                ori255 = 256 * (double)(2*k+1)/2.0/NUM_BINS;
                if (ori255 >= 256) ori255 -=256 ; 
                if (ori255 < 0)    ori255 +=256 ;
/*
                cout << "    index :   " << k+1 << endl
                     << "     ori :    " << ori255 << endl 
                     << "hist orient : " << hist_orient[k] << endl << endl;
*/
                orien.push_back(ori255);
                  mag.push_back(hist_orient[k]);
            }
            //cout << k << endl;
        }

            // Save this keypoint into the keyPoints vector

            keyPoints.push_back(new Keypoint(xi*scale/2, yi*scale/2, mag, orien, i*BLUR_NUM + j));
        }
        }
        }
    }


void ExtractKeypointDescriptors(){

    printf("\n\nExtract keypoint descriptors...\n");
    int width, height, scale, octInd, blurInd, targetX, targetY, curX, curY;
    int main_mag, main_orien;
    int i, j, ii, jj, starti, startj, limiti, limitj, t;
    float keyPx,keyPy, descxi, descyi;
    double dx, dy, sample_orien, actBin;
    int halfSize = FEATURE_WINDOW_SIZE/2;
    int oneBlockSize = FEATURE_WINDOW_SIZE/4;
    int bin, norm255;

    // Allocate magnitudes and orientations
    vector<int> orien, mag;
    cv::Mat imgInterpolMag[OCT_NUM][BLUR_NUM], imgInterpolOri[OCT_NUM][BLUR_NUM];
    cv::Mat weight(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_32FC1);

    // These two loops calculate the interpolated thingy for all octaves and subimages
    // These two loops calculate the interpolated thingy for all octaves and subimages
    for(i=0;i<OCT_NUM; i++){
    width  = octave[i][0]->size().width;
    height = octave[i][0]->size().height;
        for(j=0;j<BLUR_NUM-3;j++){
        // Allocate memory
        imgInterpolMag[i][j] = Mat(width,height,CV_32FC1);
        imgInterpolOri[i][j] = Mat(width,height,CV_32FC1);
        }
    }
    cv::Mat* G = new Mat(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_32FC1);
    BuildInterpolatedGaussianTable(G, FEATURE_WINDOW_SIZE, 0.5*FEATURE_WINDOW_SIZE);

    for(i=0;i<OCT_NUM; i++){
    width  = octave[i][0]->size().width;
    height = octave[i][0]->size().height;
    for(j=0;j<BLUR_NUM-3;j++){

        // Do the calculations
        for(ii=0; ii<width -1; ii++){
        for(jj=0; jj<height-1; jj++){
            // "inbetween" change
            // 01 11  ---\
            // 00 10  ---/ 0,0
            dx = (octave[i][j]->at<uchar>(jj+1, ii+1) + octave[i][j]->at<uchar>(jj, ii+1) - octave[i][j]->at<uchar>(jj+1, ii) - octave[i][j]->at<uchar>(jj, ii))/2;
            dy = (octave[i][j]->at<uchar>(jj+1, ii+1) + octave[i][j]->at<uchar>(jj+1, ii) - octave[i][j]->at<uchar>(jj, ii+1) - octave[i][j]->at<uchar>(jj, ii))/2;
            imgInterpolMag[i][j].at<uchar>(jj, ii) = sqrt(dx*dx + dy*dy);
            if (atan2(dy,dx) == M_PI)
                norm255 = 0;
            else 
                norm255 = (int)(255.0 * (atan2(dy,dx) + M_PI) / 2.0/M_PI);

            imgInterpolOri[i][j].at<uchar>(jj, ii) = norm255;
            }
            }

        // Pad the edges with zeros
        for(ii=0;ii<width;ii++){
            imgInterpolMag[i][j].at<uchar>(height, ii) = 0;
            imgInterpolOri[i][j].at<uchar>(height, ii) = 0;
            }
        for(jj=0;jj<height;jj++){
            imgInterpolMag[i][j].at<uchar>(jj, width) = 0;
            imgInterpolOri[i][j].at<uchar>(jj, width) = 0;
            }
        }
        }


    vector<double> hist(DESC_NUM_BINS);

    cout << endl << "keyP loop" << endl;
    cout << "Keypoints size is : " << keyPoints.size() << endl << endl;


    for(int ikp = 0;ikp < keyPoints.size();ikp++){
        Descriptor* D;
        cout << "keypoint index is : " <<ikp << endl;

        scale   = keyPoints[ikp]->scale;
        keyPx   = keyPoints[ikp]->xi; descxi  = keyPx;
        keyPy   = keyPoints[ikp]->yi; descyi  = keyPy;

        octInd  = scale/BLUR_NUM;
        blurInd = scale%BLUR_NUM;

        // position of the exact size picture
        targetX = (int)(keyPx*2) / (int)(pow(2.0, (double)octInd));
        targetY = (int)(keyPy*2) / (int)(pow(2.0, (double)octInd));

        width  = octave[octInd][0]->size().width;
        height = octave[octInd][0]->size().height;

        orien = keyPoints[ikp]->orien;
        mag   = keyPoints[ikp]->mag;

        // Find the orientation and magnitude that have the "maximum impact"
        // on the feature

        cout << "____First____"         << endl
             << "mag is : " << mag[0]   << endl
             << "ori is : " << orien[0] << endl
             << "_____________"         << endl;

        main_mag   = mag[0];
        main_orien = orien[0];



cout << mag.size() <<endl;
        for(int orient_count=1;orient_count < mag.size();orient_count++){
            cout << orient_count << endl;
            if(mag[orient_count] > main_mag){

                main_orien  = orien[orient_count];
                main_mag    = mag[orient_count];
                }
            }

        cout << "____Main____"             << endl
             << "mag is : " << main_mag     << endl
             << "ori is : " << main_orien   << endl
             << "____________"             << endl;

        vector<double> fv(FVSIZE);


        cout << "Making Weight..." << endl << endl;
        for(i=0;i<FEATURE_WINDOW_SIZE;i++){
        for(j=0;j<FEATURE_WINDOW_SIZE;j++){
            //out of boundary
            if(targetX-halfSize+i < 0 || targetX-halfSize+i > width ||
               targetY-halfSize+j < 0 || targetY-halfSize+j > height)
                weight.at<uchar>(j, i) = 0;
            else
                weight.at<uchar>(j, i) = G->at<uchar>(j, i) *
                                            imgInterpolMag[octInd][blurInd].at<uchar>(targetY - halfSize + j, targetX - halfSize + i);
            }
            }

        // Now that we've weighted the required magnitudes, we proceed to generating
        // the feature vector

        // The next two two loops are for splitting the 16x16 window
        // into sixteen 4x4 blocks
        assert(FEATURE_WINDOW_SIZE%4 == 0);

        cout << "Making Fingerprint" << endl;
        for(i=0;i<oneBlockSize;i++){
        for(j=0;j<oneBlockSize;j++){
            // Clear the histograms
            for(t=0;t < DESC_NUM_BINS;t++)
                hist[t]=0.0;

            // Calculate the coordinates of the 4x4 block
            starti = targetX - halfSize + oneBlockSize*i;
            startj = targetY - halfSize + oneBlockSize*j;
            limiti = starti + oneBlockSize;
            limitj = startj + oneBlockSize;

            // Go though this 4x4 block and do the thingy :D
            for(curX = starti; curX < limiti; curX++){
            for(curY = startj; curY < limitj; curY++){
                if(curX<0 || curX>=width || curY<0 || curY>=height)
                    continue;

                // Independent from rotation
                sample_orien  = imgInterpolOri[octInd][blurInd].at<uchar>(curY, curX);
                sample_orien -= main_orien;
                // SAMPLE_ORIEN is 255 uchar scale
                while(sample_orien < 0)
                    sample_orien += 256;

                while(sample_orien >= 256)
                    sample_orien -= 256;

                actBin  = (double)(sample_orien*DESC_NUM_BINS/360.0);       // The actual entry
                bin = (int)actBin;
                //bin   = sample_orien * DESC_NUM_BINS/256;                         // The bin


                // Add to the bin
                hist[bin] += (1.0-fabs(actBin-(bin+0.5))) * weight.at<uchar>(curY-targetY + halfSize, curX-targetX + halfSize);
                }
                }


            // Keep adding these numbers to the feature vector
            int inBlockInd = i*oneBlockSize + j; 
            cout << inBlockInd << endl;
            for(t=0;t<DESC_NUM_BINS;t++)
                fv[inBlockInd*DESC_NUM_BINS+t] = hist[t];
            }
            }

        cout << endl << endl << "Normalize.." << endl;

        // Now, normalize the feature vector to ensure illumination independence
        double norm=0;
        for(t=0;t<FVSIZE;t++)
            norm+=pow(fv[t], 2.0);

        cout << "power" << endl;
        norm = sqrt(norm);
        for(t=0;t<FVSIZE;t++)
            fv[t]/=norm;

        cout << "threshold" << endl;

        for(t=0;t<FVSIZE;t++)
            if(fv[t]>FV_THRESHOLD)
                fv[t] = FV_THRESHOLD;

        cout << "Re-norm" << endl;
        // Normalize again
        norm=0;
        for(t=0;t<FVSIZE;t++)
            norm+=pow(fv[t], 2.0);

        cout << "Divide" << endl;
        norm = sqrt(norm);
        for(t=0;t<FVSIZE;t++)
            fv[t]/=norm;

        // We're done with this descriptor. Store it into a list
        cout << "PUSH" << endl;
        D = new Descriptor(keyPx, keyPy, fv);
        cout << "asdadsasdasdfdfassgae" << endl;
        keyDescs.push_back(D);
        cout << "PULL" << endl;

        }
    }

void BuildInterpolatedGaussianTable(cv::Mat* ret, int size, double sigma){
    int i, j;
    double half_kernel_size = size/2 - 0.5;
    cout << "Making gaussian table" << endl;
    double sog=0;
    assert(size%2==0);

    double temp=0;
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            //center : 4 cells. i.e. 8 X 8->[1~8][1~8] -> [4~5][4~5]
            temp = gaussian2D(i-half_kernel_size, j-half_kernel_size, sigma);
            ret->at<uchar>(j, i) = temp;
            sog+=temp;
        }
    }
    for(i=0;i<size;i++)
        for(j=0;j<size;j++)
            ret->at<uchar>(j, i) /= sog;
    }

// gaussian2D
// Returns the value of the bell curve at a (x,y) for a given sigma
double gaussian2D(double x, double y, double sigma){
    double ret = 1.0/(2*M_PI*sigma*sigma) * exp(-(x*x+y*y)/(2.0*sigma*sigma));
    return ret;
    }



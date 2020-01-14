// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>

using namespace cv;
using namespace std;

#define OCT_NUM     5
#define BLUR_NUM    5
#define MIN_BRIGHT  2
#define MIN_CURVE   10
#define NUM_BINS    16
#define MAX_KERNEL_SIZE 20
#define M_PI        3.14159265358979323846
#define CUT_OFF     0.001

void makeOctave(Mat& mat, Size size);

void DetectExtrema();
    void isExtrema(Mat& up, Mat& target, Mat& down, Size size);

void AssignOrientations();
    int GetKernelSize(double sigma);
    void makeMagAndOri(Mat& mag, Mat& ori, int i, int j);
    void saveKeyP(Mat& imgWeight, int width, int height, int i, int j);

void ExtractKeypointDescriptors();

//const auto
char window_name[] = "SIFT test";
cv::Mat octave[OCT_NUM][BLUR_NUM], dogList[OCT_NUM][BLUR_NUM-1], extImg[OCT_NUM][BLUR_NUM-3];
vector<Keypoint> keyPoints;
double absSigma[OCT_NUM];

class Keypoint{
    public:
        float           xi;
        float           yi;     // Location
        vector<double>  mag;    // The list of magnitudes at this point
        vector<double>  orien;  // The list of orientations detected
        unsigned int    scale;  // The scale where this was detected

        Keypoint() { }
        Keypoint(float x, float y) { xi=x; yi=y; }
        Keypoint(float x, float y, vector<double> const& m, vector<double> const& o, unsigned int s)
        {
            xi = x;
            yi = y;
            mag = m;
            orien = o;
            scale = s;
        }
};


int main(int argc, char * argv[]) 
{

    namedWindow(window_name, WINDOW_AUTOSIZE);
    Mat image = imread("./SIFT/myimage.jpg", IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    cout << "Size of the picture is" << image.size() << endl;
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
    imshow(window_name, imgGray);
    cv::pyrUp(imgGray, bigImg); 
    GaussianBlur(bigImg, octave[0][0], Size(0,0), initSigma, initSigma);

    // Making pyramid
    std::cout << "Making img pyramid" << endl;

    for(int i = 0; i < OCT_NUM-1; i++)
        cv::pyrDown(octave[i][0], octave[i+1][0]);

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
            octave[i][j] = octave[i][0].clone();
            GaussianBlur(octave[i][j], octave[i][j],Size(0,0),curSigma,curSigma);
            absSigma[j] = cursigma; 

            curSigma *=sigmaF;
            }
        }

    std::cout << "show octave 14" << endl;
    imshow(window_name, octave[1][4]);
    waitKey(0);


    std::cout << "Computing DoG." << endl;
    for(int i = 0; i < OCT_NUM; i++){
        for(int j = 0; j < BLUR_NUM-1; j++){
            cv::subtract(octave[i][j], octave[i][j+1], dogList[i][j]);
            }
        }
    std::cout << "OK." << endl;
    sleep(0.1);
    std::cout << "show" << endl;
    imshow(window_name, dogList[1][2]);
    waitKey(0);

    }

void DetectExtrema(){
    cout << endl << "!! Detecting Extrema !!" << endl;
    int i, j;
    cv::Mat up, target, down;
    for (i=0;i<OCT_NUM;i++){
        for (j=0;j<BLUR_NUM-3;j++){
            up = dogList[i][j]; target = dogList[i][j+1]; down = dogList[i][j+2];
            cout << i << "        " << j<< endl;
            isExtrema(up,target,down,target.size(),i,j);
        }
    }
}

void isExtrema(Mat& up, Mat& target, Mat& down, Size size, int i, int j){
    cv::Mat temp(size,CV_32FC1);
    extImg[i][j] = temp.clone();
    uchar curPix;
    double dxx, dyy, dxy, trH, detH, curvature_ratio;
    cout << size ;
    int number = 0,dark = 0, edge = 0;
    for(int i = 1; i < size.width-2 ;i++){
        for(int j = 1; j < size.height-2 ; j++){
            curPix = target.at<uchar>(j,i);
            if ((up.at<uchar>(j-1,i-1)    > curPix && up.at<uchar>(j,i-1)    > curPix && up.at<uchar>(j+1,i-1)     > curPix &&
                up.at<uchar>(j-1,i  )     > curPix && up.at<uchar>(j,i)      > curPix && up.at<uchar>(j+1,i  )     > curPix &&
                up.at<uchar>(j-1,i+1)     > curPix && up.at<uchar>(j,i+1)    > curPix && up.at<uchar>(j+1,i+1)     > curPix &&
                target.at<uchar>(j-1,i-1) > curPix && target.at<uchar>(j,i-1)> curPix && target.at<uchar>(j+1,i-1) > curPix &&
                target.at<uchar>(j-1,i  ) > curPix &&                                    target.at<uchar>(j+1,i  ) > curPix &&
                target.at<uchar>(j-1,i+1) > curPix && target.at<uchar>(j,i+1)> curPix && target.at<uchar>(j+1,i+1) > curPix &&
                down.at<uchar>(j-1,i-1)   > curPix && down.at<uchar>(j,i-1)  > curPix && down.at<uchar>(j+1,i-1)   > curPix &&
                down.at<uchar>(j-1,i  )   > curPix && down.at<uchar>(j,i)    > curPix && down.at<uchar>(j+1,i  )   > curPix &&
                down.at<uchar>(j-1,i+1)   > curPix && down.at<uchar>(j,i+1)  > curPix && down.at<uchar>(j+1,i+1)   > curPix 
                )||(
                up.at<uchar>(j-1,i-1)     < curPix && up.at<uchar>(j,i-1)    < curPix && up.at<uchar>(j+1,i-1)     < curPix &&
                up.at<uchar>(j-1,i  )     < curPix && up.at<uchar>(j,i)      < curPix && up.at<uchar>(j+1,i  )     < curPix &&
                up.at<uchar>(j-1,i+1)     < curPix && up.at<uchar>(j,i+1)    < curPix && up.at<uchar>(j+1,i+1)     < curPix &&
                target.at<uchar>(j-1,i-1) < curPix && target.at<uchar>(j,i-1)< curPix && target.at<uchar>(j+1,i-1) < curPix &&
                target.at<uchar>(j-1,i  ) < curPix &&                                    target.at<uchar>(j+1,i  ) < curPix &&
                target.at<uchar>(j-1,i+1) < curPix && target.at<uchar>(j,i+1)< curPix && target.at<uchar>(j+1,i+1) < curPix &&
                down.at<uchar>(j-1,i-1)   < curPix && down.at<uchar>(j,i-1)  < curPix && down.at<uchar>(j+1,i-1)   < curPix &&
                down.at<uchar>(j-1,i  )   < curPix && down.at<uchar>(j,i)    < curPix && down.at<uchar>(j+1,i  )   < curPix &&
                down.at<uchar>(j-1,i+1)   < curPix && down.at<uchar>(j,i+1)  < curPix && down.at<uchar>(j+1,i+1)   < curPix 
                )){
                    dxx = (target.at<uchar>(j-1, i) + target.at<uchar>(j+1, i) -
                        2.0*target.at<uchar>(j, i));
                    dyy = (target.at<uchar>(j, i-1) + target.at<uchar>(j, i+1) -
                        2.0*target.at<uchar>(j, i));
                    dxy = (target.at<uchar>(j-1, i-1) + target.at<uchar>(j+1, i+1) -
                        target.at<uchar>(j+1, i-1) - target.at<uchar>(j-1, i+1)) / 4.0;
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
                    extImg.at<uchar>(j,i) = 255;
                    number++;
                    }
                }
        }
    }

    imshow(window_name, extImg);
    printf("Found %d keypoints\n", number);
    printf("Rejected keypoints\ndark : %d\neadge : %d", dark,edge);
    waitKey(0);
}

void AssignOrientations(){
    cout << endl << "!! Assigning Orientations !!" << endl;
    cv::Mat*** magnitude    = new cv::Mat** [OCT_NUM],
    cv::Mat*** orientations = new cv::Mat** [OCT_NUM];

    // Allocate memory
    for(i=0;i<m_numOctaves;i++){
        magnitude[i]   = new IplImage* [BLUR_NUM]];
        orientation[i] = new IplImage* [BLUR_NUM];
        }

    for (i=0;i<OCT_NUM;i++){
        for (j=0;j<BLUR_NUM;j++){
            magnitude[i][j]  (octave[i][0].size(),CV_32FC1);
            orientation[i][j](octave[i][0].size(),CV_32FC1);
            makeMagAndOri(magnitudep[i][j], orientation[i][j], i, j)
            }
        }
    
    
    
    for(i=0;i<OCT_NUM;i++){
        // Store current scale, width and height
        unsigned int scale = (int)pow(2.0, (double)i);
        unsigned int width = octave[i][0].size().width;
        unsigned int height= octave[i][0].size().height;
        // Go through all intervals in the current scale
        for(j=0;j<BLUR_NUM;j++){
            double abs_sigma = absSigma[j];
            // This is used for magnitudes
            cv::Mat imgWeight(width, height, CV_32FC1);
            GaussianBlur(magnitude[i][j], imgWeight, Size(0,0), 1.5*abs_sigma, 1.5*abs_sigma);
            int sizeK = GetKernelSize(1.5*abs_sigma)/2;
            cv::Mat imgMask(width, height, CV_32FC1);
            saveKeyP();
        }
    }

    // Make memory free
    for(i=0; i<m_numOctaves; i++){
        for(j=0; j<m_numIntervals; j++){
              magnitude[i][j].release();
            orientation[i][j].release();
        }
        delete [] magnitude[i];
        delete [] orientation[i];
    }
    delete [] magnitude;
    delete [] orientation;
}

int GetKernelSize(double sigma){
    for (int i=0;i<MAX_KERNEL_SIZE;i++)
        if (exp(-((double)(i*i))/(2.0*sigma*sigma)) < CUT_OFF)
            break;
    int size = 2*i-1;
    return size;
}

void makeMagAndOri(Mat& imgMask, Mat& imgWeight, Mat& mag, Mat& ori, int i, int j, int scale){
    for(xi=1;xi < octave[i][j].size().width-1;xi++){
        for(yi=1;yi < octave[i][j].size().height-1;yi++){
            // Calculate gradient
            double dx = octave[i][j].at<uchar>(yi, xi+1)-octave[i][j].at<uchar>(yi, xi-1);
            double dy = octave[i][j].at<uchar>(yi+1, xi)-octave[i][j].at<uchar>(yi-1, xi);
            // Store magnitude
            mag.at<uchar>(yi, xi) = sqrt(dx*dx + dy*dy);

            double angRadian;
            if (dx==0)
                dy>0 ? angRadian = 0.5*math.pi : angRadian = -0.5 * math.pi;
            else 
                angRadian = atan(dy/dx);


            double norm255 = (angRadian/math.pi + 0.5) * 255.0;
            if (norm255 >= 255.0) norm255-=255.0; 
            if (norm255 < 0.0)   norm255+=255.0;

            ori.at<uchar>(yi, xi) = norm255;
            }
        }
}

void saveKeyP(Mat& imgWeight, int width, int height, int i, int j){
    double hist_orient[NUM_BINS];

    for(xi=0;xi<width; xi++){
    for(yi=0;yi<height;yi++){
        // At the keypoint
    if(extImg[i][j].at<uchar>(yi, xi)!=0){
        // Reset the histogram
        // xk and tk : kernel x, y position, center is (0,0)
        for(xk = -sizeK; xk <= sizeK; xk++){
        for(yk = -sizeK; yk <= sizeK; yk++){
            // out of the picture : ignore 
            if(xi+xk<0 || xi+xk>=width || yi+yk<0 || yi+yk>=height)
                continue;
            double curOrient255 = orientation[i][j].at<uchar>(yi+yk, xi+xk);
            hist_orient[(int)curOri255 / (256/NUM_BINS)] += imgWeight.at<uchar>(yi+yk, xi+xk);
            imgMask.at<uchar>(yi+yk, xi+xk, 255);
            }}

        // We've computed the histogram. Now check for the maximum
        double max_peak = hist_orient[0];
        unsigned int max_peak_index = 0;
        for(k=0;k<NUM_BINS;k++){
            if(hist_orient[k]>max_peak){
                max_peak = hist_orient[k]; max_peak_index = k;
            }}

            // List of magnitudes and orientations at the current extrema
        vector<double> orien;
        vector<double> mag;
        for(k=0;k<NUM_BINS;k++){
            if(hist_orient[k] > 0.8*max_peak){
                double x1 = k-1, x2 = k, x3 = k+1;
                double y1, y2 = hist_orient[k], y3;
        {
                if(k==0){
                    y1 = hist_orient[NUM_BINS-1];
                    y3 = hist_orient[1];}
                else if(k==NUM_BINS-1){
                    y1 = hist_orient[NUM_BINS-1];
                    y3 = hist_orient[0];}
                else{
                    y1 = hist_orient[k-1];
                    y3 = hist_orient[k+1];}
        }
                double b[3];
                cv::Mat X(3, 3, CV_32FC1), matInv(3, 3, CV_32FC1);

                // (Least-square method) for parabola :
                // parabola is constructed by near points, output b is parameters

                X.at<uchar>(0, 0, x1*x1); X.at<uchar>(1, 0, x1); X.at<uchar>(2, 0, 1);
                X.at<uchar>(0, 1, x2*x2); X.at<uchar>(1, 1, x2); X.at<uchar>(2, 1, 1);
                X.at<uchar>(0, 2, x3*x3); X.at<uchar>(1, 2, x3); X.at<uchar>(2, 2, 1);

                matInv = X.inv();

                // inv(X) * y = parameters
                b[0] = matInv.at<uchar>(0, 0)*y1 + matInv.at<uchar>(1, 0)*y2 + matInv.at<uchar>(2, 0)*y3;
                b[1] = matInv.at<uchar>(0, 1)*y1 + matInv.at<uchar>(1, 1)*y2 + matInv.at<uchar>(2, 1)*y3;
                b[2] = matInv.at<uchar>(0, 2)*y1 + matInv.at<uchar>(1, 2)*y2 + matInv.at<uchar>(2, 2)*y3;

                //x0 is center z-position of the of the parabola
                double x0 = -b[1] / (2*b[0]);

                // Anomalous situation
                // if(fabs(x0) > 2*NUM_BINS)
                //    x0=x2;
            {
                while(x0 < 0)
                    x0 += NUM_BINS;
                while(x0 >= NUM_BINS)
                    x0-= NUM_BINS;
            }
                // Normalize it, convert -PI ~ PI
                double x0_n = x0 * (2*M_PI/NUM_BINS);
                x0_n -= M_PI;

                orien.push_back(x0_n);
                  mag.push_back(hist_orient[k]);
            }
        }

            // Save this keypoint into the keyPoints vector
            keyPoints.push_back(Keypoint(xi*scale/2, yi*scale/2, mag, orien, i*BLUR_NUM + j));
        }
        }
        }
}


void ExtractKeypointDescriptors(){
    cout << endl << "!! Extracting Keypoint Descriptors !!" << endl;
}
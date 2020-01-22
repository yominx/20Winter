void ExtractKeypointDescriptors(){

    printf("\n\nExtract keypoint descriptors...\n");
    vector<int> orien, mag;
    int main_mag, main_orien, t;
    float keyPx,keyPy;
    double dx, dy, sample_orien, actBin, norm;
    int bin, norm255;
    cv::Mat imgInterpolMag[OCT_NUM][BLUR_NUM], imgInterpolOri[OCT_NUM][BLUR_NUM];
    cv::Mat weight(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_32FC1);

    allocMagOriDesc(imgInterpolOri,imgInterpolMag);
    cv::Mat* G = new Mat(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_32FC1);
    BuildInterpolatedGaussianTable(G, FEATURE_WINDOW_SIZE, 0.5*FEATURE_WINDOW_SIZE);
    vector<double> hist(DESC_NUM_BINS);

    cout << endl << "keyP loop" << endl;
    cout << "Keypoints size is : " << keyPoints.size() << endl << endl;
    
    for(int ikp = 0;ikp < keyPoints.size();ikp++){
        cout << "keypoint index is : " <<ikp << endl;
        keyPx   = keyPoints[ikp]->xi;
        keyPy   = keyPoints[ikp]->yi;
        orien   = keyPoints[ikp]->orien;
        mag     = keyPoints[ikp]->mag;

        main_mag   = mag[0];
        main_orien = orien[0];

        for(int orient_count=1;orient_count < mag.size();orient_count++){
            cout << orient_count << endl;
            if(mag[orient_count] > main_mag){

                main_orien  = orien[orient_count];
                main_mag    = mag[orient_count];
                }
            }
        cout << mag.size() <<endl;
        cout << "____First____"             << endl
             << "mag is : " << mag[0]       << endl
             << "ori is : " << orien[0]     << endl
             << "____Main____"              << endl
             << "mag is : " << main_mag     << endl
             << "ori is : " << main_orien   << endl
             << "____________"              << endl;

        vector<double> fv = makeWeightAndFP(imgInterpolOri, imgInterpolMag, ikp);
        cout << endl << endl << "Normalize.." << endl;

        // Now, normalize the feature vector to ensure illumination independence
        norm=0;
        for(t=0;t<FVSIZE;t++)
            norm+=pow(fv[t], 2.0);
        norm = sqrt(norm);
        for(t=0;t<FVSIZE;t++)
            fv[t]/=norm;
        for(t=0;t<FVSIZE;t++)
            if(fv[t]>FV_THRESHOLD)
                fv[t] = FV_THRESHOLD;
        
        norm=0;
        for(t=0;t<FVSIZE;t++)
            norm+=pow(fv[t], 2.0);
        norm = sqrt(norm);
        for(t=0;t<FVSIZE;t++)
            fv[t]/=norm;

        // We're done with this descriptor. Store it into a list
        cout << "PUSH" << endl;
        keyDescs.push_back(new Descriptor(keyPx, keyPy, fv));
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
    cout << "Made!" << endl;

    }

// gaussian2D
// Returns the value of the bell curve at a (x,y) for a given sigma
double gaussian2D(double x, double y, double sigma){
    double ret = 1.0/(2*M_PI*sigma*sigma) * exp(-(x*x+y*y)/(2.0*sigma*sigma));
    return ret;
    }

void allocMagOriDesc(Mat** imgInterpolOri, Mat** imgInterpolMag){
    int i,j,ii,jj,width,height,dx,dy,norm255;
    for(i=0;i<OCT_NUM; i++){
        width  = octave[i][0].size().width;
        height = octave[i][0].size().height;
            for(j=0;j<BLUR_NUM-3;j++){
            // Allocate memory
            imgInterpolMag[i][j] = Mat(width,height,CV_32FC1);
            imgInterpolOri[i][j] = Mat(width,height,CV_32FC1);
            }
        }


    for(i=0;i<OCT_NUM; i++){
    width  = octave[i][0].size().width;
    height = octave[i][0].size().height;
    for(j=0;j<BLUR_NUM-3;j++){

        // Do the calculations
        for(ii=0; ii<width -1; ii++){
        for(jj=0; jj<height-1; jj++){
            // "inbetween" change
            // 01 11  ---\
            // 00 10  ---/ 0,0
            dx = (octave[i][j].at<uchar>(jj+1, ii+1) + octave[i][j].at<uchar>(jj, ii+1) - octave[i][j].at<uchar>(jj+1, ii) - octave[i][j].at<uchar>(jj, ii))/2;
            dy = (octave[i][j].at<uchar>(jj+1, ii+1) + octave[i][j].at<uchar>(jj+1, ii) - octave[i][j].at<uchar>(jj, ii+1) - octave[i][j].at<uchar>(jj, ii))/2;
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
    }



vector<double> makeWeightAndFP(Mat** imgInterpolOri, Mat** imgInterpolMag, int ikp){
    int width, height, scale, octInd, blurInd, targetX, targetY, curX, curY;
    int i, j, starti, startj, limiti, limitj, t;
    float keyPx,keyPy;
    double  sample_orien, actBin;
    int bin;

    int halfSize = FEATURE_WINDOW_SIZE/2, oneBlockSize = FEATURE_WINDOW_SIZE/4;

    vector<double> fv(FVSIZE);
    scale   = keyPoints[ikp]->scale;
    octInd  = scale/BLUR_NUM;
    blurInd = scale%BLUR_NUM;
    keyPx   = keyPoints[ikp]->xi;
    keyPy   = keyPoints[ikp]->yi;
    targetX = (int)(keyPx*2) / (int)(pow(2.0, (double)octInd));
    targetY = (int)(keyPy*2) / (int)(pow(2.0, (double)octInd));
    width  = octave[octInd][0].size().width;
    height = octave[octInd][0].size().height;

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
    return fv;
}
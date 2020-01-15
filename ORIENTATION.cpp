void SIFT::ExtractKeypointDescriptors()
{
	printf("Extract keypoint descriptors...\n");

	// Allocate magnitudes and orientations
	cv::Mat*** imgInterpolMag = new cv::Mat** [m_numOctaves];
	cv::Mat*** imgInterpolOri = new cv::Mat** [m_numOctaves];
	for(i=0;i<m_numOctaves;i++){
		imgInterpolMag[i] = new cv::Mat* [BLUR_NUM];
		imgInterpolOri[i] = new cv::Mat* [BLUR_NUM];
	}

	// These two loops calculate the interpolated thingy for all octaves and subimages
	for(int i=0;i<m_numOctaves;i++){
	for(int j=0;j<BLUR_NUM;j++){

		// Scale up. This will give us access to in betweens		
		int width  = octave[i][j].size().width;
		int height = octave[i][j].size().height;

		// Allocate memory
		imgInterpolMag[i][j](width+1, height+1, 32, 1);
		imgInterpolOri[i][j](width+1, height+1, 32, 1);

		// Do the calculations
		for(float ii=1; ii<width -1; ii++){
		for(float jj=1; jj<height-1; jj++){
			// "inbetween" change
			int dx = (octave[i][j].at<uchar>(jj, ii+1) - octave[i][j].at<uchar>(jj, ii-1)) /2;
			int dy = (octave[i][j].at<uchar>(jj+1, ii) - octave[i][j].at<uchar>(jj-1, ii)) /2;

			// Set the magnitude and orientation
			imgInterpolMag[i][j].at<uchar>(jj+1, ii+1, sqrt(dx*dx + dy*dy));
			imgInterpolOri[i][j].at<uchar>(jj+1, ii+1, (atan2(dy,dx)==M_PI) ? -M_PI : atan2(dy,dx) );
		}
		}

		// Pad the edges with zeros
		for(int ii=0;ii<width+1;ii++){
			imgInterpolMag[i][j].at<uchar>(0, ii, 0); imgInterpolMag[i][j].at<uchar>(height, ii, 0);
			imgInterpolOri[i][j].at<uchar>(0, ii, 0); imgInterpolOri[i][j].at<uchar>(height, ii, 0);
		}
		for(int jj=0;jj<height+1;jj++){
			imgInterpolMag[i][j].at<uchar>(jj, 0, 0); imgInterpolMag[i][j].at<uchar>(jj, width, 0);
			imgInterpolOri[i][j].at<uchar>(jj, 0, 0); imgInterpolOri[i][j].at<uchar>(jj, width, 0);
		}
	}
	}

	// Build an Interpolated Gaussian Table of size FEATURE_WINDOW_SIZE
	// Lowe suggests sigma should be half the window size
	// This is used to construct the "circular gaussian window" to weight 
	// magnitudes
	cv::Mat *G = BuildInterpolatedGaussianTable(FEATURE_WINDOW_SIZE, 0.5*FEATURE_WINDOW_SIZE);
	
	vector<double> hist(DESC_NUM_BINS);

	// Loop over all keypoints
	for(int ikp = 0;ikp < m_numKeypoints;ikp++){
		int   scale = keyPoints[ikp].scale;
		float keyPx = keyPoints[ikp].xi;
		float keyPy = keyPoints[ikp].yi;

		float descxi = keyPx;
		float descyi = keyPy;

		int ii = (int)(keyPx*2) / (int)(pow(2.0, (double)scale/BLUR_NUM));
		int jj = (int)(keyPy*2) / (int)(pow(2.0, (double)scale/BLUR_NUM));

		int width  = octave[scale/BLUR_NUM][0].size().width;
		int height = octave[scale/BLUR_NUM][0].size().height;

		vector<double> orien = m_keyPoints[ikp].orien;
		vector<double> mag   = m_keyPoints[ikp].mag;

		// Find the orientation and magnitude that have the maximum impact
		// on the feature
		double main_mag = mag[0];
		double main_orien = orien[0];
		for(int orient_count=1;orient_count<mag.size();orient_count++){
			if(mag[orient_count]>main_mag){
				main_orien = orien[orient_count];
				main_mag = mag[orient_count];
			}
		}

		int hfsz = FEATURE_WINDOW_SIZE/2;
		cv::Mat weight(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_32FC1);
		vector<double> fv(FVSIZE);

		for(i=0;i<FEATURE_WINDOW_SIZE;i++){
			for(j=0;j<FEATURE_WINDOW_SIZE;j++){
				if(ii+i+1<hfsz || ii+i+1>width+hfsz || jj+j+1<hfsz || jj+j+1>height+hfsz)
                    cvSetReal2D(weight, j, i, 0);
				else
					cvSetReal2D(weight, j, i, cvGetReal2D(G, j, i)*cvGetReal2D(imgInterpolMag[scale/BLUR_NUM][scale%BLUR_NUM], jj+j+1-hfsz, ii+i+1-hfsz));
			}
		}

		// Now that we've weighted the required magnitudes, we proceed to generating
		// the feature vector

		// The next two two loops are for splitting the 16x16 window
		// into sixteen 4x4 blocks
		for(i=0;i<FEATURE_WINDOW_SIZE/4;i++){
			for(j=0;j<FEATURE_WINDOW_SIZE/4;j++){
				// Clear the histograms
				for(int t=0;t<DESC_NUM_BINS;t++)
					hist[t]=0.0;

				// Calculate the coordinates of the 4x4 block
				int starti = (int)ii - (int)hfsz + 1 + (int)(hfsz/2*i);
				int startj = (int)jj - (int)hfsz + 1 + (int)(hfsz/2*j);
				int limiti = (int)ii + (int)(hfsz/2)*((int)(i)-1);
				int limitj = (int)jj + (int)(hfsz/2)*((int)(j)-1);

				// Go though this 4x4 block and do the thingy :D
				for(int k=starti;k<=limiti;k++)
				{
					for(int t=startj;t<=limitj;t++)
					{
						if(k<0 || k>width || t<0 || t>height)
							continue;

						// This is where rotation invariance is done
						double sample_orien = cvGetReal2D(imgInterpolOri[scale/BLUR_NUM][scale%BLUR_NUM], t, k);
						sample_orien -= main_orien;

						while(sample_orien<0)
							sample_orien+=2*M_PI;

						while(sample_orien>2*M_PI)
							sample_orien-=2*M_PI;

						// This should never happen
						if(!(sample_orien>=0 && sample_orien<2*M_PI))
							printf("BAD: %f\n", sample_orien);
						assert(sample_orien>=0 && sample_orien<2*M_PI);

						int sample_orien_d = sample_orien*180/M_PI;
						assert(sample_orien_d<360);

						int bin = sample_orien_d/(360/DESC_NUM_BINS);					// The bin
						double bin_f = (double)sample_orien_d/(double)(360/DESC_NUM_BINS);		// The actual entry

						assert(bin<DESC_NUM_BINS);
						assert(k+hfsz-1-ii<FEATURE_WINDOW_SIZE && t+hfsz-1-jj<FEATURE_WINDOW_SIZE);

						// Add to the bin
						hist[bin]+=(1-fabs(bin_f-(bin+1))) * cvGetReal2D(weight, t+hfsz-1-jj, k+hfsz-1-ii);
					}
				}

				// Keep adding these numbers to the feature vector
				for(int t=0;t<DESC_NUM_BINS;t++)
				{
					fv[(i*FEATURE_WINDOW_SIZE/4+j)*DESC_NUM_BINS+t] = hist[t];
				}
			}
		}

		// Now, normalize the feature vector to ensure illumination independence
		double norm=0;
		for(int t=0;t<FVSIZE;t++)
			norm+=pow(fv[t], 2.0);
		norm = sqrt(norm);

		for(int t=0;t<FVSIZE;t++)
			fv[t]/=norm;

		// Now, threshold the vector
		for(int t=0;t<FVSIZE;t++)
			if(fv[t]>FV_THRESHOLD)
				fv[t] = FV_THRESHOLD;

		// Normalize yet again
		norm=0;
		for(int t=0;t<FVSIZE;t++)
			norm+=pow(fv[t], 2.0);
		norm = sqrt(norm);

		for(int t=0;t<FVSIZE;t++)
			fv[t]/=norm;

		// We're done with this descriptor. Store it into a list
		m_keyDescs.push_back(Descriptor(descxi, descyi, fv));
	}

	assert(m_keyDescs.size()==m_numKeypoints);

	// Get rid of memory we don't need anylonger
	for(i=0;i<m_numOctaves;i++)
	{
		for(j=0;j<BLUR_NUM;j++)
		{
			cvReleaseImage(&imgInterpolMag[i][j]);
			cvReleaseImage(&imgInterpolOri[i][j]);
		}

		delete [] imgInterpolMag[i];
		delete [] imgInterpolOri[i];
	}

	delete [] imgInterpolMag;
	delete [] imgInterpolOri;
}

// GetKernelSize()
// Returns the size of the kernal for the Gaussian blur given the sigma and
// cutoff value.


// BuildInterpolatedGaussianTable()
// This function actually generates the bell curve like image for the weighted
// addition earlier.
CvMat* SIFT::BuildInterpolatedGaussianTable(int size, double sigma)
{
	int i, j;
	double half_kernel_size = size/2 - 1;

	double sog=0;
	CvMat* ret = cvCreateMat(size, size, CV_32FC1);

	assert(size%2==0);

	double temp=0;
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			temp = gaussian2D(i-half_kernel_size, j-half_kernel_size, sigma);
			cvSetReal2D(ret, j, i, temp);
			sog+=temp;
		}
	}

	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			cvSetReal2D(ret, j, i, 1.0/sog * cvGetReal2D(ret, j, i));
		}
	}

	return ret;
}

// gaussian2D
// Returns the value of the bell curve at a (x,y) for a given sigma
double SIFT::gaussian2D(double x, double y, double sigma)
{
	double ret = 1.0/(2*M_PI*sigma*sigma) * exp(-(x*x+y*y)/(2.0*sigma*sigma));
	return ret;
}



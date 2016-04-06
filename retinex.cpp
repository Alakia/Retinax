
#include"stdafx.h"
#include "retinex.h" 
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <imgproc\imgproc_c.h>
#include <imgproc\imgproc.hpp>
#include <math.h>   

 using namespace cv; 
#define USE_EXACT_SIGMA   
  
  
#define pc(image, x, y, c) image->imageData[(image->widthStep * y) + (image->nChannels * x) + c]
  
#define INT_PREC 1024.0   
#define INT_PREC_BITS 10   
  
inline double int2double(int x) { return (double)x / INT_PREC; }  
inline int double2int(double x) { return (int)(x * INT_PREC + 0.5); }  
  
inline int int2smallint(int x) { return (x >> INT_PREC_BITS); }  
inline int int2bigint(int x) { return (x << INT_PREC_BITS); }  



typedef struct
{
	int width;
	int height;
	char pImgData[400*500];
	int Img_flag;
}t_ReturnInfo;

t_ReturnInfo return_info;

//IplImage *A, *fA, *fB, *fC;
// CreateKernel   
//   
// Summary:   
// Creates a normalized 1 dimensional gaussian kernel.   
//   
// Arguments:   
// sigma - the standard deviation of the gaussian kernel.   
//   
// Returns:   
// double* - an array of values of length ((6*sigma)/2) * 2 + 1.   
//   
// Note:   
// Caller is responsable for deleting the kernel.   

int* CreateKernel(double sigma)  
{  
    int i, x; 
	int	filter_size; //高斯窗口大小 
    double* filter;
	int* kernel;
    double sum;  
  
    // Reject unreasonable demands   
    if ( sigma > 300 ) sigma = 200;  
  
    // get needed filter size (enforce oddness)   
    filter_size = (int)floor(sigma*6) / 2;  
    filter_size = filter_size * 2 + 1;  
  
    // Allocate kernel space   
    filter = new double[filter_size]; 
	kernel = new int[filter_size];
  
    // Calculate exponential   openCV函数 getGaussianKernel()
	// 产生filter_size*1维的高斯滤波核
    sum = 0;  
    for (i = 0; i < filter_size; i++) {  
        x = i - (filter_size / 2);  
        filter[i] = exp( -(x*x) / (2*sigma*sigma) );  
  
        sum += filter[i];  
    }  
  
    // Normalize   
    for (i = 0; i < filter_size; i++)  
	{
		filter[i] /= sum;
		kernel[i] = double2int(filter[i]);
	}
  
	delete filter;

    return kernel;  
} 

//   
// CreateFastKernel   
//   
// Summary:   
// Creates a faster gaussian kernal using integers that   
// approximate floating point (leftshifted by 8 bits)   
//   
// Arguments:   
// sigma - the standard deviation of the gaussian kernel.   
//   
// Returns:   
// int* - an array of values of length ((6*sigma)/2) * 2 + 1.   
//   
// Note:   
// Caller is responsable for deleting the kernel.   
//   
int* CreateFastKernel(double sigma)  
{  
    double* fp_kernel;  
    int* kernel;  
    int i, filter_size;  
      
    // Reject unreasonable demands   
    if ( sigma > 300 ) sigma = 200;  
  
    // get needed filter size (enforce oddness)   
    filter_size = (int)floor(sigma*6) / 2;  
    filter_size = filter_size * 2 + 1;  
  
    // Allocate kernel space   
    kernel = new int[filter_size];  
  
    //fp_kernel = CreateKernel(sigma);  
  
    for (i = 0; i < filter_size; i++)  
        kernel[i] = double2int(fp_kernel[i]);  
  
    delete fp_kernel;  
  
    return kernel;  
}  
//   
// FilterGaussian   
//   
// Summary:   
// Performs a gaussian convolution for a value of sigma that is equal   
// in both directions.   
//   
// Arguments:   
// img - the image to be filtered in place.   
// sigma - the standard deviation of the gaussian kernel to use.   
// 
void FilterGaussian(IplImage* img, double sigma)  
{  
    int i, j, k, source, filter_size;  
    int* kernel;
    IplImage* temp;  
    int v1, v2, v3;  
  
    // Reject unreasonable demands   
    if ( sigma > 300 ) sigma = 200;  
  
    // get needed filter size (enforce oddness)   
    filter_size = (int)floor(sigma*6) / 2;  
    filter_size = filter_size * 2 + 1;  
  
    //kernel = CreateFastKernel(sigma);
	kernel = CreateKernel(sigma);
  
    temp = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);  
      
    // filter x axis   
    for (j = 0; j < temp->height; j++)  
    for (i = 0; i < temp->width; i++) {  
  
        // inner loop has been unrolled   
        v1 = v2 = v3 = 0;  
        for (k = 0; k < filter_size; k++) {  
              
            source = i + filter_size / 2 - k;  
      
            if (source < 0) source *= -1;  
            if (source > img->width - 1) source = 2*(img->width - 1) - source;  
			//if (source < 0) source *= -1;

            v1 += kernel[k] * (unsigned char)pc(img, source, j, 0);  
            if (img->nChannels == 1) continue;  
            v2 += kernel[k] * (unsigned char)pc(img, source, j, 1);  
            v3 += kernel[k] * (unsigned char)pc(img, source, j, 2); 

        }  
  
        // set value and move on   
        pc(temp, i, j, 0) = (char)int2smallint(v1);  
        if (img->nChannels == 1) continue;  
        pc(temp, i, j, 1) = (char)int2smallint(v2);  
        pc(temp, i, j, 2) = (char)int2smallint(v3); 
  
    }  
      
    // filter y axis   
    for (j = 0; j < img->height; j++)  
    for (i = 0; i < img->width; i++) {  
  
        v1 = v2 = v3 = 0;  
        for (k = 0; k < filter_size; k++) {  
              
            source = j + filter_size / 2 - k;  
      
			if (source < 0) source *= -1;
			if (source > temp->height - 1) source = 2*(temp->height - 1) - source;
            //if (source < 0) source *= -1;        
  
            v1 += kernel[k] * (unsigned char)pc(temp, i, source, 0);  
            if (img->nChannels == 1) continue;  
            v2 += kernel[k] * (unsigned char)pc(temp, i, source, 1);  
            v3 += kernel[k] * (unsigned char)pc(temp, i, source, 2); 

        }  
  
        // set value and move on   
        pc(img, i, j, 0) = (char)int2smallint(v1);  
        if (img->nChannels == 1) continue;  
        pc(img, i, j, 1) = (char)int2smallint(v2);  
        pc(img, i, j, 2) = (char)int2smallint(v3);
  
    }  
  
    cvReleaseImage( &temp );  
  
    delete kernel;  
  
}  
//   
// FastFilter   
//   
// Summary:   
// Performs gaussian convolution of any size sigma very fast by using   
// both image pyramids and seperable filters.  Recursion is used.   
//   
// Arguments:   
// img - an IplImage to be filtered in place.   

void Subsample(IplImage *src, IplImage *dest, int cols, int rows)
{
	int i,j;
	char *pSrc = src->imageData, *pDst = dest->imageData;
	for (i=0;i<rows;i+=2) {

		for (j=0;j<cols;j+=2) {
			pDst[0] = pSrc[0];
			pDst ++;
			pSrc += 2;
		}
		pDst += cols/2 %4;
		pSrc += cols;
	}
}

void  FastFilter(IplImage *img, double sigma, int flag = 0)  
{  
    int filter_size;
	int width = img->width, height = img->height;
	
  
    // Reject unreasonable demands   
    if ( sigma > 700 ) sigma = 200;  
  
    // get needed filter size (enforce oddness)   
    filter_size = (int)floor(sigma*6) / 2;  
    filter_size = filter_size * 2 + 1;  
  
    // If 3 sigma is less than a pixel, why bother (ie sigma < 2/3)   
    if(filter_size < 3) return ;  
  
    // Filter, or downsample and recurse
    if (filter_size <= 13) 
	{    
		double time = (double)getTickCount();
        FilterGaussian(img, sigma);
            
    }  
    else {  
        if (img->width < 2 || img->height < 2) return;  
		
        IplImage* sub_img = cvCreateImage(cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels);
		//IplImage* sub_img = cvCreateImage(cvSize(img->width / 2, img->height / 2), 8, 1);
  
		//Subsample(img, sub_img);
        cvPyrDown( img, sub_img ); 
  
        FastFilter( sub_img, sigma / 2.0);  

        cvResize( sub_img, img, CV_INTER_LINEAR );  
  
        cvReleaseImage( &sub_img );  
    }  
  
}  

void auto_level(IplImage* src, int width, int height, double avg)
{
	int i, j, total_num = 0, Avg = 0;
		
	total_num = width * height;

	// 直方图统计
	int histData[1] = {256};
	int map_hist[256] = {0};

	CvHistogram *hist = cvCreateHist(1, histData, CV_HIST_ARRAY);
	cvCalcHist( &src, hist );//计算单通道图像直方图


	// 直方图动态非线性拉伸
	int max_bin, min_bin, delta_bin, mean_num;
	int stdSum0 = 0, stdSum1 = 0;
	unsigned char  *pt;
	float  std = 0;

	
	//// 2015.8 求图像均值
	i = 0;
	while(i < 256) 
	{
		Avg += hist->mat.data.fl[i]*i;
		stdSum1 += hist->mat.data.fl[i];
		i++;
	}
	Avg = Avg / total_num;

	// 2015.8 求图像方差 std = sum( (x_i - x_avg)*(x_i - x_avg) )
	i = 0;
	while(i < 256)//width需为8的倍数
	{			
		stdSum0 = stdSum0 + (i - avg)*(i - avg)*hist->mat.data.fl[i];			
		i++;
	}
    std = sqrt((float)(stdSum0 / total_num));

	// 2015.8 动态拉伸上下限
	max_bin = (int)(avg + 3*std);
	min_bin = (int)(avg - 3*std);

	// 求原始图像最大和最小的bin
	delta_bin = max_bin - min_bin;

	// map = 255*(src - min)/(max - min)
	for( i = 0; i < 256; i++ )
	{
		if( i <= min_bin ) 
			map_hist[i] = 0;
		else if( i >= max_bin )
			map_hist[i] = 255;
		else
			map_hist[i] = 255*(i - min_bin) / delta_bin;
	}

	// 根据拉伸狈酵糾ap_hist[256]求增强图像  
	//直接在src上修改像素值获得增强后图像
	j = 0;
	while(j < total_num)
	{
		src->imageData[j] = map_hist[src->imageData[j]];
		j++;
	}
	
}
   
  /**********************************************************************
    * MultiScaleRetinex 单通道
	* Summary:   
	* Multiscale retinex restoration.  The image and a set of filtered images are   
	* converted to the log domain and subtracted from the original with some set   
	* of weights. Typicaly called with three equaly weighted scales of fine,   
	* medium and wide standard deviations.   
	*   
	* Arguments:   
	* img - an IplImage to be enhanced in place.   
	* sigma - the standard deviation of the gaussian kernal used to filter.   
	* gain - the factor by which to scale the image back into visable range.   
	* offset - an offset similar to the gain.
	********************************************************************/
void MSR(IplImage *img, int scales, double *weights, double *sigmas)  
{  
    int i=0, j = 0, r = 0, c = 0, filter_size = 0;  
	IplImage *A, *fA, *fB, *fC, *subsample;
	double time;
	int total = img->height * img->width;
	//t_ReturnInfo return_img;

    // Initialize temp images
	int img_width = img->width, img_height = img->height;
    fA = cvCreateImage(cvSize(img_width, img_height), IPL_DEPTH_32F, img->nChannels);  
    fB = cvCreateImage(cvSize(img_width, img_height), IPL_DEPTH_32F, img->nChannels);  
    fC = cvCreateImage(cvSize(img_width, img_height), IPL_DEPTH_32F, img->nChannels);  

    // Compute log image   
	ImageAvg = cvAvg(img);
    cvConvert( img, fA ); 
	float* p=(float*)(fA->imageData);

	for( r = 0; r < img_height; r++)
		for( c = 0; c < img_width; c++)
		{
			if(0.0f == p[r * img_width + c])  
				p[r * img_width + c] = 1;
		}//将原图为0的像素值置为1

    cvLog( fA,fB); //原图求对数
  

    // Filter at each scale 
	
    for (i = 0; i < scales;i++) 
	{  
        A = cvCloneImage( img );
		FastFilter( A, sigmas[i]);		
        cvConvert( A, fA );

		for(r = 0;r < img_height; r++)
		  for(c = 0;c < img_width; c++)
			{
				if(0.0f == p[r*img_width+c])  
					p[r*img_width+c] = 1;
			}//将滤波后图为0的像素值置为1

        cvLog( fA, fC ); 
  
        // Compute weighted difference 
		cvScale( fC, fC, weights[i] );
		cvSub( fB, fC, fB ); 
    }

	//使用图像均值代替大尺度卷积结果
	//float* pA = (float*)fA->imageData;
	//for( i = 0; i<total; i++)
	//{
	//	pA[i] = (float)ImageAvg.val[0];
	//}

	//cvLog( fA, fC );
	//cvScale(fC, fC, weights[2] );
	//cvSub( fB, fC, fB );

    // Restore   
	cvConvertScale( fB, img, 400, 128);

    // auto_level
//	auto_level(img, img_width, img_height, ImageAvg.val[0]);
	
    // Release temp images 
	cvReleaseImage( &A );
    cvReleaseImage( &fA );  
    cvReleaseImage( &fB );  
    cvReleaseImage( &fC );
} 

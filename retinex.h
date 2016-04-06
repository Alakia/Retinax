#include <cv.h>  
 
extern CvScalar ImageAvg;
extern float LogLUT[256];

int* CreateKernel(double sigma);  
int* CreateFastKernel(double sigma);  
  
 void FilterGaussian(IplImage* img, double sigma);  
//t_ReturnInfo FastFilter(IplImage *img, double sigma);  
  
 void Retinex  
(IplImage *img, double sigma, int gain = 128, int offset = 128);  
  
 void MSR(IplImage *img, int scales, double *weights, double *sigmas);  
 void MSR2(IplImage *img, int scales, double *weights, double *sigmas, int gain = 128, int offset = 128); 
 void MSR3(IplImage *img, int scales, double *weights, double *sigmas, int gain = 128, int offset = 128); 
 void MSRCR(IplImage *img, int scales, double *weights, double *sigmas, int gain = 128, int offset = 128,  
 double restoration_factor = 6, double color_gain = 2);  
 void MSRSim(IplImage *img,IplImage *temp,IplImage *t2,CvMat *Km);
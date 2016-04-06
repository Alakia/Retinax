// retinex1.cpp : 定义控制台应用程序的入口点。
#include "stdafx.h"
#include<stdio.h>
#include<math.h>
#include "retinex.h"
#include "highgui.h"

using namespace std;
using namespace cv;
 float DLUT[255][255];
void MSR2DLUT(IplImage *img, int scales, double *weights, double *sigmas, int gain, int offset);

//int _tmain(int argc, _TCHAR* argv[])
//{
//	float l=log10(10.0f);
//	int i=0,j=0;
//	//for(i=1;i<=255;i++) //2DLUT[i][j]=log10(i)-log10(j)
//	//	for(j=1;j<255;j++)
//	//	{
//	//	  DLUT[i][j]=(log10(float(i))-log10(float(j)))*128+128;
//	//	}
//	IplImage * pImg,*pImgHist,*pImg1,*pImg2;
//	pImg=cvLoadImage("D:\\haze20.jpg");
//	if(NULL==pImg) return 0;
//	//if(pImg->nChannels!=3) return 0;
//	IplImage *pImgGray = cvCreateImage( cvSize(pImg->width, pImg->height),pImg->depth, 1);
//	cvCvtColor(pImg, pImgGray, CV_RGB2GRAY);
//	cvNamedWindow( "OriImage", 0);
//    cvShowImage( "OriImage", pImgGray );
//
//	double w[3]={0.4, 0.3, 0.3};//{1.0/3.0f,1.0/3.0f,1.0/3.0f};//
//	double sigmas[3]={5.0f,20.0f,240.0f};
//	
//	double t = (double)getTickCount();
//	MSR(pImgGray,3,w,sigmas);
//	t = (((double)getTickCount() - t)/getTickFrequency())*1000;//t ms
//	printf("%f ms",t);
//	int r=0,c=0,n=0;
//	unsigned char val=0,max=0,min=255;
//	char* data=pImg->imageData;
//
//	//for(n=0;n<pImg->nChannels;n++) 
//	//	for(r=0;r<pImg->height;r++)
//	//		for(c=0;c<pImg->width;c++)
//	//			{
//	//				val=data[r*pImg->widthStep+c];
//	//				if(val>255) 
//	//					data[r*pImg->widthStep+c]=255;
//	//				if(val<0) data[r*pImg->widthStep+c]=0;		
//	//		    }
//
//	//pImgHist=cvCreateImage(cvSize(pImg->width, pImg->height), IPL_DEPTH_8U,1); 
// //   pImg1=cvCreateImage(cvSize(pImg->width, pImg->height), IPL_DEPTH_8U,1);
// //   pImg2=cvCreateImage(cvSize(pImg->width, pImg->height), IPL_DEPTH_8U,1);
//
//	//cvSplit(pImg, pImgHist, pImg1, pImg2, NULL);
//	//cvEqualizeHist(pImgGray,pImgHist); //直方图均衡化
//	// cvEqualizeHist(pImg1,pImg1);
//	// cvEqualizeHist(pImg2,pImg2);
//
//	 //cvMerge(pImgHist, pImg1, pImg2, NULL, pImg);
//
//     cvNamedWindow( "MSR", 0 );
//     cvShowImage( "MSR", pImgGray );
//	 //cvShowImage("MSR",pImg);
//	 //cvNamedWindow("EqualizeHist");
//	 //cvShowImage("EqualizeHist",pImgHist );
//     //cvSaveImage("D:\\ImgGray.jpg",pImgGray);
//     cvWaitKey(0); 
//
//	 cvDestroyAllWindows();
//	 //cvReleaseImage( &pImgHist);
//	 cvReleaseImage( &pImgGray); 
//	 //cvReleaseImage( &pImg2); 
//     cvReleaseImage( &pImg ); 
//	 return 0;
//}
float k[121*121]={0};//sigma=20,3*sigma=20*3=60
float k2[31*31]={0};//sigma=5,3*sigma=5*3=15

void main() //只考虑单通道灰度图
{
 int i=0,j=0;

//生成对数查找表
 static float LogLUT[256]={0};

   for(i=2;i<256;i++)
	   LogLUT[i]=log10((float)i);

 IplImage *pImg, *pImgGray;
 IplImage *pImg1;
 pImg=cvLoadImage("D:\\image4-1full.jpg",1);

 pImgGray = cvCreateImage( cvSize(pImg->width, pImg->height),pImg->depth, 1);
 cvCvtColor(pImg, pImgGray, CV_RGB2GRAY);
 cvNamedWindow("OriImage");
 cvShowImage("OriImage",pImgGray );

 IplImage *t1=cvCreateImage(cvSize(pImg->width, pImg->height), IPL_DEPTH_32F,1); 
 IplImage *t2=cvCreateImage(cvSize(pImg->width, pImg->height), IPL_DEPTH_32F,1); 
 pImg1=cvCreateImage(cvSize(pImg->width, pImg->height), IPL_DEPTH_8U,1); 

 double t = (double)getTickCount();

 //MSRSim(pImgGray,t1,t2,&Km);
 //MSRSim(pImg2,t1,t2,&Km);
 //MSRSim(pImg3,t1,t2,&Km);

	double w[3]={0.4, 0.3, 0.3};//{1.0/3.0f,1.0/3.0f,1.0/3.0f};//
	double sigmas[3]={10, 56, 177};//{5.0f,20.0f,240.0f};//
	//double c[3] = {15, 80, 250};

	MSR2(pImgGray,3,w,sigmas);

	/*MSR(pImg2,3,w,sigmas);
	MSR(pImg3,3,w,sigmas);*/
	t = (((double)getTickCount() - t)/getTickFrequency())*1000;//t ms
	printf("%f ms",t);

 cvNamedWindow( "MSRSim");
 cvShowImage( "MSRSim", pImgGray );
 cvWaitKey(0);

 cvSaveImage("D:\\MSR_auto4.jpg",pImgGray);
 cvReleaseImage( &pImg ); 
 cvReleaseImage( &pImgGray);
 cvReleaseImage( &pImg1);
 cvReleaseImage( &t1);
 cvReleaseImage( &t2); 
}

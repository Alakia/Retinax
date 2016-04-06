// retinex1.cpp : 定义控制台应用程序的入口点。
#include "stdafx.h"
#include<stdio.h>
#include<math.h>
#include "retinex.h"
#include "highgui.h"

//#define READ_IMAGE

using namespace std;
using namespace cv;


// 全局变量
CvScalar ImageAvg;
float LogLUT[256] = {0};

int main() //只考虑单通道灰度图
{
	//生成对数查找表
	for(int i = 2; i < 256; i++)
		LogLUT[i] = log10((float)i);

	//设置retinex参数
	double w[3]={0.4, 0.3, 0.3};	
	double sigmas[3] = {5.0f, 20.0f, 300.0f};//small scale: 1%-5% image size

	//图像变量定义
	Mat frame;
	IplImage *InputImg, *InputImgGray, *OutputImg;
	char imagename[1024];


#ifdef READ_IMAGE
	// 读图像
	const char* input_filename = "image_list.txt";//text file with a list of the images of the board
    FILE* f = 0;
	f = fopen(input_filename,"rt");
	if(!f)
	{
		printf("The input file could not be opened\n");
		return 0;
	}

#else

	//读视频
	VideoCapture InputVideo;
	const char* VideoPath = "car1.avi";
	InputVideo.open( VideoPath );
	if( !InputVideo.isOpened())
		return -1;
	int fps = (int)InputVideo.get(CV_CAP_PROP_FPS);

#endif

	cvNamedWindow("OriImage",1);
	cvNamedWindow( "MSR",1);

	while(1)
	{
#ifdef READ_IMAGE
		if( fgets( imagename, sizeof(imagename), f ))
		{
			int l = strlen(imagename);
			if( l > 0 && imagename[l-1] == '\n' )
			{
				imagename[--l] = '\0';
				frame = imread( imagename);

			}
		}
		if( !frame.data )
			break;
#else
		// 读视频
		InputVideo>>frame;
		if( frame.empty() )
			break;
#endif
		else
		{
			InputImg = &frame.operator IplImage();
			InputImgGray = cvCreateImage( cvSize(InputImg->width, InputImg->height),InputImg->depth, 1);
			cvCvtColor(InputImg, InputImgGray, CV_RGB2GRAY);

			OutputImg = cvCreateImage(cvSize(InputImgGray->width, InputImgGray->height), InputImgGray->depth, 1); 
			cvConvertScale( InputImgGray, OutputImg );

			double t = (double)getTickCount();
			MSR(OutputImg,3,w,sigmas); 


			//显示原图及处理结果
			cvShowImage("OriImage",InputImgGray );
			cvShowImage( "MSR", OutputImg );
			char c = waitKey(fps);
			if( c == 27 )
				break;

			cvReleaseImage( &InputImgGray);
			cvReleaseImage( &OutputImg);
			}
	}
}

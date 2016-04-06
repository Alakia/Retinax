// retinex1.cpp : �������̨Ӧ�ó������ڵ㡣
#include "stdafx.h"
#include<stdio.h>
#include<math.h>
#include "retinex.h"
#include "highgui.h"

//#define READ_IMAGE

using namespace std;
using namespace cv;


// ȫ�ֱ���
CvScalar ImageAvg;
float LogLUT[256] = {0};

int main() //ֻ���ǵ�ͨ���Ҷ�ͼ
{
	//���ɶ������ұ�
	for(int i = 2; i < 256; i++)
		LogLUT[i] = log10((float)i);

	//����retinex����
	double w[3]={0.4, 0.3, 0.3};	
	double sigmas[3] = {5.0f, 20.0f, 300.0f};//small scale: 1%-5% image size

	//ͼ���������
	Mat frame;
	IplImage *InputImg, *InputImgGray, *OutputImg;
	char imagename[1024];


#ifdef READ_IMAGE
	// ��ͼ��
	const char* input_filename = "image_list.txt";//text file with a list of the images of the board
    FILE* f = 0;
	f = fopen(input_filename,"rt");
	if(!f)
	{
		printf("The input file could not be opened\n");
		return 0;
	}

#else

	//����Ƶ
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
		// ����Ƶ
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


			//��ʾԭͼ��������
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

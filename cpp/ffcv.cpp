#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
 
using namespace cv;
using namespace std;

ofstream outfile;
char imname[30];
char outname[30];
int imnum = 0;
int imnummax = 10;
Mat image, h;
char numstr[10];

int prepro(){
   sprintf(numstr,"%04d",imnum);
   strcpy(imname,"../images/");
   strcat(imname,"a");
   strcat(imname,numstr);
   strcat(imname,".jpg");
   cout << imname << std::endl;
   image = imread(imname, CV_LOAD_IMAGE_COLOR); // Read the file
   //add smoothing, contrast adjustment
}

int segment(){
}

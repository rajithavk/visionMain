/*
 * functions.h
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_


#endif /* FUNCTIONS_H_ */

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <stdlib.h>
#include <string>

using namespace std;
using namespace cv;

int buildVocabulary();

Mat getDescriptors(Mat image,vector<KeyPoint> keypoints);
vector<KeyPoint> calcKeyPoints(Mat image);

void drawKeyPoints(Mat image, vector<KeyPoint> keypoints);
void showImage(Mat image);

void openCamera(int index);

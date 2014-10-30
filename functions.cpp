/*
 * functions.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

int buildVocabulary(char *filepath){
	DIR *dir;
	struct dirent *entry;
	struct stat filestat;
	dir = opendir(filepath);
	if(!dir) return -1;

	while((entry = readdir(dir))){
		cout << entry->d_name << " " << entry->d_type << endl;
	}

	return closedir(dir);
	return 0;
}


Mat getDescriptors(Mat image,vector<KeyPoint> keypoints){
	Mat descriptors;																	// descriptors of the current image
	SiftDescriptorExtractor descriptorExtractor;										// feature extractor
	descriptorExtractor.compute(image,keypoints,descriptors);							// extract
	return descriptors;																	// return the descriptors for the input image
}

vector<KeyPoint> calcKeyPoints(Mat image){
	vector<KeyPoint> keypoints;															// SIFT keypoints of the current image
	SiftFeatureDetector featureDetector;												// feature Detector
	featureDetector.detect(image,keypoints);
	return keypoints;
}

void drawKeyPoints(Mat image, vector<KeyPoint> keypoints){
	Mat outimage;
	drawKeypoints(image,keypoints,outimage,Scalar(255,0,0));
	imshow("Keypoints",outimage);
	//waitKey(0);
}

void showImage(Mat image){
	imshow("Image",image);
	//waitKey(0);
}

void openCamera(int index=0){	// index - video device - 0,1,2... == video0, video1
	Mat frame;
	char key;
	namedWindow("Camera");
	VideoCapture capture(index);
	if(capture.isOpened()){
		while(true){
			capture >> frame;
			imshow("Output",frame);
			key = waitKey(10);
			if(char(key) == 27)
				break;
		}
	}
	destroyWindow("Camera");
}

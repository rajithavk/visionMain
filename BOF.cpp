/*
 * main.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

Mat image;
vector<KeyPoint> keypoints;
Mat descriptors;


int main(int argc , char **argv){
//	cout << argv[1] << endl;
//	image = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
//	if(image.empty()) return -1;
//	keypoints = calcKeyPoints(image);
//	descriptors = getDescriptors(image,keypoints);
//	namedWindow("Window");
//	drawKeyPoints(image,keypoints);
//	waitKey(0);
//	openCamera(1);
//	if(buildVocabulary("images")==0)
//		cout << "Success";

	if(trainSVM()==0){
		cout << "Success";
	}
	return 0;
}

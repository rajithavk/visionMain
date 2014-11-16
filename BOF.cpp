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
char input;

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
	vision* dev = new vision();

	while(1){
		cin >> input;
		if(input == '1'){
			dev->loadTrainingSet();
		}else
			if(input == '2'){
				dev->buildVocabulary();
			}else
				if(input=='3'){
					dev->trainSVM();
				}else
					if(input=='q'){
						break;
					}
	}

//		dev->initVocabulary();
//		dev->trainSVM();

//	if(*argv[1] == '1'){
//		if(dev->buildVocabulary("images")==0)
//		cout << "Success Vocabulary";
//
//	}
//	else
//		if(dev->trainSVM()==0){
//			cout << "Success SVM";
//		}
	return 0;
}

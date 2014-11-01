/*
 * functions.h
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_




#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/legacy/legacy.hpp>
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

class vision{

private :
		Mat vocabulary;
		map<string,CvSVM> classes_classifiers;
		vector < vector <KeyPoint> > keypointsvec;

public:
		void drawKeyPoints(Mat image, vector<KeyPoint> keypoints);
		void showImage(Mat image);
		vector<KeyPoint> calcKeyPoints(Mat image);
		Mat getDescriptors(Mat image,vector<KeyPoint> keypoints);
		int buildVocabulary(String filepath);
		int trainSVM();

		void openCamera(int index);

		vision();
		~vision();
};
#endif /* FUNCTIONS_H_ */

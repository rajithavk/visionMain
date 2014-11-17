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
#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>
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
using namespace boost::filesystem;

class vision{

private :

		Ptr<FeatureDetector> featureDetector;
		Ptr<DescriptorExtractor> descriptorExtractor;
		Ptr<BOWKMeansTrainer> bowTrainer;
		Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor;
		Ptr<DescriptorMatcher> descriptorMatcher;

		Mat vocabulary;
		multimap<string,Mat> training_set;
		map<string,CvSVM> classes_classifiers;
		vector < vector <KeyPoint> > keypoints_vector;
		vector <string> classes;
		int num_of_classes;
		int num_of_samples;

		double edgeThreshold = 10;

		const String TRAINING_DATA_FILE = "training_data.dat";
		const String KEYPOINTS_FILE = "keypoints.yml";
		const String TRAINING_DESCRIPTORS_FILE = "training_descriptors.yml";
		const String VOCABULARY_FILE = "vocabulary.yml";
		const String TRAINING_FOLDER = "images";
		const int CLUSTERS = 2000;

public:
		int loadTrainingSet();										// Load training images set into memory
		int initVocabulary();										// Load default vocabulary from file
		int initVocabulary(String filename);						// For loading custom vocabulary in case
		void drawKeyPoints(Mat image, vector<KeyPoint> keypoints);	// Draw keypoints for given image + keypoints vector
		void showImage(Mat image);									// Showing an image
		vector<KeyPoint> getKeyPoints(Mat image);					// Get the keypoints vector for given image
		Mat getDescriptors(Mat image,vector<KeyPoint> keypoints);	// Get image descriptors for a given image keypoints combination
		int buildVocabulary();										// build the vocabulary
		int trainSVM();												// train the SVMs
		int testImage(Mat testimage);											// testing an input image against SVMs
		int initClassifiers();										// load SVMs

		void openCamera(VideoCapture cap);

		vision();
		~vision();
};
#endif /* FUNCTIONS_H_ */

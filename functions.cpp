/*
 * functions.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

int buildVocabulary(String  filepath){
	DIR *dir,*subdir;
	char CWD[2049],ROOT[2049];
	Mat input , descriptor, featuresUnclustered , vocabulary;
	vector<KeyPoint> keypoints;
	ofstream trainingdata;
	struct dirent *entry,*imagefile;
	struct stat filestat;

	trainingdata.open("trainingdata.dat",ofstream::out | ofstream::app);

	if(getcwd(ROOT,2049) == NULL) return -1;


	dir = opendir(filepath.c_str());
	if(!dir) return -1;


	if(chdir(filepath.c_str()) < 0){
		cout << "Error in CHDIR" << endl;
		return -1;
	}

	if(getcwd(CWD,2049) == NULL) return -1;

	cout << CWD << endl;


	while((entry = readdir(dir))){
		//cout << entry->d_name << endl;
		if(stat(entry->d_name, &filestat) <0){
			cout << "Error\n";
			continue;
		}

		if(!strcmp(entry->d_name,"."))	continue;
		if(!strcmp(entry->d_name,".."))	continue;

		if(S_ISDIR(filestat.st_mode)){
			cout << "\n" << entry->d_name<<endl;
			subdir = opendir(entry->d_name);
			if(!subdir){
				cout << "Err";
				return -1;
			}
			while((imagefile = readdir(subdir))){
				if(!strcmp(imagefile->d_name,"."))	continue;
				if(!strcmp(imagefile->d_name,".."))	continue;

				String impath = String(CWD) + "/" + String(entry->d_name) + "/" +  imagefile->d_name;
				cout << impath << endl;

				input = imread(impath.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
				keypoints = calcKeyPoints(input);
				descriptor = getDescriptors(input,keypoints);
				featuresUnclustered.push_back(descriptor);
				trainingdata << impath << " " << entry->d_name << endl;
				//cout << imagefile->d_name << " ";
			}
			closedir(subdir);
		}

	}

	closedir(dir);
	trainingdata.close();
	if(chdir(ROOT) < 0) return -1;

	cout << "Total Descriptors : " << featuresUnclustered.rows << endl;
	FileStorage fs("training_descriptors.yml",FileStorage::WRITE);
	fs << "training_descriptors" << featuresUnclustered;
	fs.release();
	cout << "Training Descriptors => " << ROOT << "\training_descriptors.yml" << endl;

	BOWKMeansTrainer bowtrainer(1000);
	bowtrainer.add(featuresUnclustered);
	cout << "Cluster BOW Features" << endl;
	vocabulary = bowtrainer.cluster();

	FileStorage fs1("vocabulary.yml",FileStorage::WRITE);
	fs1 << "Vocabulary" << vocabulary;
	fs1.release();
	cout << "Vocabulary => " << ROOT << "\vocabulary.yml" << endl;
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

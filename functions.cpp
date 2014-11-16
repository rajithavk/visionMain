/*
 * functions.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

vision::vision(){
	create_directories("./classifiers");

	bowTrainer = (new BOWKMeansTrainer(1000));
	descriptorMatcher = (new BruteForceMatcher<L2<float> >);
	//Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	descriptorExtractor = (new SiftDescriptorExtractor);

	bowDescriptorExtractor = (new BOWImgDescriptorExtractor(descriptorExtractor,descriptorMatcher));

};
vision::~vision(){};



int vision::buildVocabulary(){
	Mat input , descriptor, features_unclustered;
	vector<KeyPoint> keypoints;

	for(multimap<string,Mat>::iterator it = training_set.begin();it!=training_set.end();it++){

				input = (*it).second;
				keypoints = getKeyPoints(input);
				keypoints_vector.push_back(keypoints);
				descriptor = getDescriptors(input,keypoints);
				features_unclustered.push_back(descriptor);
				//cout << imagefile->d_name << " ";

		}
	//==================== Saving All the Keypoints ==========================
	//FileStorage fs2(KEYPOINTS_FILE.c_str(),FileStorage::WRITE);
	//for(size_t i =1;i!=keypoints_vector.size();++i){
	//	cout << String("KP" + i) << endl;
	//	fs2 << String("KP" + i) << keypoints_vector[i+1];
	//}
	//fs2.release();
	//========================================================================

	cout << "Total Descriptors : " << features_unclustered.rows << endl;
	FileStorage fs(TRAINING_DESCRIPTORS_FILE.c_str(),FileStorage::WRITE);
	fs << "training_descriptors" << features_unclustered;
	fs.release();
	cout << "Training Descriptors => " << "/training_descriptors.yml" << endl;


	bowTrainer->add(features_unclustered);
	cout << "Cluster BOW Features" << endl;
	vocabulary = bowTrainer->cluster();


	FileStorage fs1(VOCABULARY_FILE.c_str(),FileStorage::WRITE);
	fs1 << "Vocabulary" << vocabulary;
	fs1.release();
	cout << "Vocabulary => " << "/vocabulary.yml" << endl;
	//fs2.release();
	return 0;
}




int vision::trainSVM(){
	Mat hist, image;									//vocabulary -> moved to private global
	vector<KeyPoint> keypoints;

	if(keypoints_vector.size()==0){
		for(multimap<string,Mat>::iterator it = training_set.begin();it!=training_set.end();it++){

						Mat input = (*it).second;
						keypoints = getKeyPoints(input);
						keypoints_vector.push_back(keypoints);
				}
	}


	bowDescriptorExtractor->setVocabulary(vocabulary);

	map<string,Mat> classes_training_data;
	classes_training_data.clear();

	vector < vector <KeyPoint> >::iterator itr = keypoints_vector.begin();

	for(multimap<string,Mat>::iterator it=training_set.begin();it!=training_set.end();it++){

		bowDescriptorExtractor->compute((*it).second,(*itr),hist);

		string _class = (*it).first;
		if(classes_training_data.count(_class) == 0){
			classes_training_data[_class].create(0,hist.cols,hist.type());
		}

		classes_training_data[_class].push_back(hist);
		itr++;
	}

	//CvSVMParams svmparams;
	//svmparams.svm_type	=	CvSVM::C_SVC;
	//svmparams.kernel_type	= CvSVM::RBF;
	//svmparams.nu = 0.5;



	for(map<string,Mat>::iterator it = classes_training_data.begin();it != classes_training_data.end();it++){
		string class_ = (*it).first;
		cout << "training.class : " << class_ << " .. " << endl;

		Mat samples(0,hist.cols,hist.type());
		Mat labels(0,1,CV_32FC1);
		samples.push_back(classes_training_data[class_]);

		Mat class_label = Mat::ones(classes_training_data[class_].rows,1,CV_32FC1);
		labels.push_back(class_label);


		for(map<string,Mat>::iterator it1 = classes_training_data.begin();it1!=classes_training_data.end();++it1){
			string not_class = (*it1).first;
			if(not_class.compare(class_) == 0) continue;
			samples.push_back(classes_training_data[not_class]);
			class_label = Mat::zeros(classes_training_data[not_class].rows,1,CV_32FC1);
			labels.push_back(class_label);
		}
		//cout << "going to train" << endl;
		Mat samples_32f; samples.convertTo(samples_32f,CV_32F);
		//classes_classifiers[class_].train(samples_32f,labels,Mat(),Mat(),svmparams);
		classes_classifiers[class_].train(samples_32f,labels);
		classes_classifiers[class_].save(String("/classifiers/"+ class_+ ".yml").c_str());
		cout << classes_classifiers.count(class_) << endl;
	}

	Mat testimage;
	testimage = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	keypoints = getKeyPoints(testimage);
	bowDescriptorExtractor->compute(testimage,keypoints,hist);

	//cout << hist.cols << endl;
	for(map<string,CvSVM>::iterator it=classes_classifiers.begin();it!=classes_classifiers.end();++it){
		float res = (*it).second.predict(hist,true);
		cout << "class: " << (*it).first << " --> " << res << endl;
	}
	return 0;
}

int vision::testImage(){
	Mat testimage,hist;

	if(initVocabulary()!=0) return -1;


	testimage = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	vector<KeyPoint> keypoints;
	keypoints = getKeyPoints(testimage);
	bowDescriptorExtractor->compute(testimage,keypoints,hist);

	//cout << hist.cols << endl;
	for(map<string,CvSVM>::iterator it=classes_classifiers.begin();it!=classes_classifiers.end();++it){
		float res = (*it).second.predict(hist,true);
		cout << "class: " << (*it).first << " --> " << res << endl;
	}
	return 0;
}





Mat vision::getDescriptors(Mat image,vector<KeyPoint> keypoints){
	Mat descriptors;																	// descriptors of the current image
	descriptorExtractor = (new SiftDescriptorExtractor());										// feature extractor
	descriptorExtractor->compute(image,keypoints,descriptors);							// extract
	return descriptors;																	// return the descriptors for the input image
}

vector<KeyPoint> vision::getKeyPoints(Mat image){
	vector<KeyPoint> keypoints;															// SIFT keypoints of the current image
	featureDetector = (new SiftFeatureDetector);												// feature Detector
	featureDetector->detect(image,keypoints);
	return keypoints;
}

void vision::drawKeyPoints(Mat image, vector<KeyPoint> keypoints){
	Mat outimage;
	drawKeypoints(image,keypoints,outimage,Scalar(255,0,0));
	imshow("Keypoints",outimage);
	//waitKey(0);
}

void vision::showImage(Mat image){
	imshow("Image",image);
	//waitKey(0);
}

int  vision::initVocabulary(){
		FileStorage fs(VOCABULARY_FILE.c_str(),FileStorage::READ);
		fs["Vocabulary"] >> vocabulary;
		fs.release();
	if(vocabulary.size >0)
		return 0;
	else
		return -1;
}

int vision::initVocabulary(String filename){
		FileStorage fs(filename.c_str(),FileStorage::READ);
		fs["Vocabulary"] >> vocabulary;
		fs.release();

		if(vocabulary.size >0)
			return 0;
		else
			return -1;
}


int vision::loadTrainingSet(){
	num_of_samples = 0;
	string class_;
	string trainig_path = current_path().string() + "/images/";
	path dirr(trainig_path);

	//cout << dirr << endl;
	for(recursive_directory_iterator end, dir(dirr);dir!=end;dir++){
		if(dir.level()==0){
			class_ = path(*dir).filename().string();
			classes.push_back(class_);
			//cout << class_ << endl;
		}
		else
			if(dir.level()==1){
				string filename = path(*dir).string();
				cout << filename << endl;
				pair<string,Mat> tmp(class_,imread(filename,CV_LOAD_IMAGE_GRAYSCALE));
				training_set.insert(tmp);
				num_of_samples++;
			}
	}
	FileStorage fs1("trainingsetinfo.yml",FileStorage::WRITE);
	fs1 << "num_of_samples" << num_of_samples;
	fs1 << "classes" << classes;
	num_of_classes = classes.size();
	cout << "Total Number of Classes : " << num_of_classes << endl;
	fs1 << "num_of_classes" << num_of_classes;
	fs1.release();
	return 0;
}

void vision::openCamera(int index=0){	// index - video device - 0,1,2... == video0, video1
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


int vision::initClassiers(){


	return 0;
}

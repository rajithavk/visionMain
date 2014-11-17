/*
 * functions.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

vision::vision(){
	create_directories("./classifiers");
	bowTrainer = (new BOWKMeansTrainer(CLUSTERS));
	descriptorMatcher = (new BruteForceMatcher<L2<float> >);
	//Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	SIFT sf;
	sf.set("edgeThreshold",edgeThreshold);
	featureDetector = (new SiftFeatureDetector(sf));
	descriptorExtractor = (new SiftDescriptorExtractor(sf));
	//cout << featureDetector->getDouble("edgeThreshold");
	bowDescriptorExtractor = (new BOWImgDescriptorExtractor(descriptorExtractor,descriptorMatcher));

};
vision::~vision(){};




//=================================================================================================
// 								Build the Visual Vocabulary
//=================================================================================================


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





//=================================================================================================
// 								Train the SVMs
//=================================================================================================

int vision::trainSVM(){
	Mat hist, image;									//vocabulary -> moved to private global
	vector<KeyPoint> keypoints;
	if(initVocabulary()!=0) return -1;

	if(keypoints_vector.size()==0){
		cout << "Making Keypoints"<<endl;
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


		//cout << classes_training_data[_class].rows << endl;
		//cout << hist.cols << hist.type() << endl;
	}

	//CvSVMParams svmparams;
	//svmparams.svm_type	=	CvSVM::C_SVC;
	//svmparams.kernel_type	= CvSVM::RBF;
	//svmparams.nu = 0.5;



	//sfor(map<string,Mat>::iterator it = classes_training_data.begin();it != classes_training_data.end();it++){
	for(vector<string>::iterator it = classes.begin();it!=classes.end();it++){
		//string class_ = (*it).first;
		string class_ = (*it);
		cout << "training.class : " << class_ << " .. " << endl;

		Mat samples(0,hist.cols,hist.type());
		Mat labels(0,1,CV_32S);
		samples.push_back(classes_training_data[class_]);

		Mat class_label = Mat::ones(classes_training_data[class_].rows,1,CV_32S);
		labels.push_back(class_label);


		//for(map<string,Mat>::iterator it1 = classes_training_data.begin();it1!=classes_training_data.end();++it1){
		for(vector<string>::iterator it1=classes.begin();it1!=classes.end();it1++){
			//string not_class = (*it1).first;
			string not_class = (*it1);
			if(not_class.compare(class_) == 0) continue;
			samples.push_back(classes_training_data[not_class]);
			class_label = Mat::ones(classes_training_data[not_class].rows,1,CV_32S);
			class_label *= -1;
			labels.push_back(class_label);
			//cout << samples.rows << " " << labels.rows<< endl;
		}

		//cout << "going to train" << endl;
		//Mat samples_32f;
		//.convertTo(samples_32f,CV_32F);

		//classes_classifiers[class_].train(samples_32f,labels,Mat(),Mat(),svmparams);
		classes_classifiers[class_].train(samples,labels);
		classes_classifiers[class_].save(String("./classifiers/"+ class_+ ".yml").c_str());
		//cout << classes_classifiers.count(class_) << endl;
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



//=================================================================================================
// 								Predict for a Image
//=================================================================================================

int vision::testImage(Mat testimage){
	Mat hist;

	if(initVocabulary()!=0) return -1;
	bowDescriptorExtractor->setVocabulary(vocabulary);

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



//=================================================================================================
// 								Helpers
//=================================================================================================

Mat vision::getDescriptors(Mat image,vector<KeyPoint> keypoints){
	Mat descriptors;																	// descriptors of the current image
	//descriptorExtractor = (new SiftDescriptorExtractor(0,5,0.4,10,1.6));
	descriptorExtractor->compute(image,keypoints,descriptors);							// extract
	return descriptors;																	// return the descriptors for the input image
}

vector<KeyPoint> vision::getKeyPoints(Mat image){
	vector<KeyPoint> keypoints;															// SIFT keypoints of the current image
	//featureDetector = (new SiftFeatureDetector());												// feature Detector
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



//=================================================================================================
// 								Load Training Image Set
//=================================================================================================
int vision::loadTrainingSet(){
	num_of_samples = 0;
	string class_;
	string training_path = current_path().string() + "/images/";
	path dirr(training_path);

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

void vision::openCamera(VideoCapture cap){	// index - video device - 0,1,2... == video0, video1

		SIFT sf;
		sf.set("edgeThreshold",edgeThreshold);
		featureDetector = (new SiftFeatureDetector(sf));
		descriptorExtractor = (new SiftDescriptorExtractor(sf));
		//cout << featureDetector->getDouble("edgeThreshold");
		bowDescriptorExtractor = (new BOWImgDescriptorExtractor(descriptorExtractor,descriptorMatcher));
	cout << "Camera Starting" << endl;
	namedWindow("cap");
	if(initVocabulary()!=0) return ;

	while(char(waitKey(1)) != 'q') {
		float dec = 1000;
		string class_;

		Mat frame, frame_g;
		cap >> frame;
		imshow("Image", frame);

		cvtColor(frame, frame_g, CV_BGR2GRAY);

		vector <KeyPoint> keypoints;
		Mat hist;
	    keypoints = getKeyPoints(frame_g);

	    bowDescriptorExtractor->setVocabulary(vocabulary);
	    bowDescriptorExtractor->compute(frame_g,keypoints,hist);
	    cout << hist.cols<<endl;

	    for(map<string,CvSVM>::iterator it=classes_classifiers.begin();it!=classes_classifiers.end();++it){
	    	float res = (*it).second.predict(hist,true);
	    		cout << "class: " << (*it).first << " --> " << res << endl;
	    		if(res<dec){
	    			dec = res;
	    			class_ = (*it).first;
	    		}
	    	}

	    Mat im = (training_set.find(class_))->second;
	    imshow("Detected",im);
	}

	destroyAllWindows();
}



//=================================================================================================
// 								Load Prebuilt SVMs
//=================================================================================================

int vision::initClassifiers(){
	String cpath = current_path().string() + "/classifiers/";
	path p(cpath);
	classes_classifiers.clear();

	for(recursive_directory_iterator end, file(p);file!=end;file++){
		String class_ = path(*file).filename().string();
		cout << "Loading.. " << class_ << endl;
		class_ = class_.substr(0,class_.length()-4);
		classes_classifiers[class_].load(path(*file).string().c_str());
	}

	return 0;
}

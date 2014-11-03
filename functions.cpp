/*
 * functions.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

vision::vision(){};
vision::~vision(){};



int vision::buildVocabulary(){
	Mat input , descriptor, features_unclustered;
	int count = 0;
	vector<KeyPoint> keypoints;
	ofstream trainingdata;
	struct dirent *entry,*imagefile;
	struct stat filestat;

	trainingdata.open(TRAINING_DATA_FILE.c_str(),ofstream::out);
	FileStorage fs2(KEYPOINTS_FILE.c_str(),FileStorage::WRITE);



	path = get

				input = imread(impath.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
				keypoints = getKeyPoints(input);
				char key[20];
				sprintf(key,"_%d",count);
				write(fs2,key,keypoints);
				count++;

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
	FileStorage fs(TRAINING_DESCRIPTORS_FILE.c_str(),FileStorage::WRITE);
	fs << "training_descriptors" << featuresUnclustered;
	fs.release();
	cout << "Training Descriptors => " << ROOT << "/training_descriptors.yml" << endl;

	BOWKMeansTrainer bowtrainer(1000);
	bowtrainer.add(featuresUnclustered);
	cout << "Cluster BOW Features" << endl;
	vocabulary = bowtrainer.cluster();

	FileStorage fs1(VOCABULARY_FILE.c_str(),FileStorage::WRITE);
	fs1 << "Vocabulary" << vocabulary;
	fs1.release();
	cout << "Vocabulary => " << ROOT << "/vocabulary.yml" << endl;
	fs2.release();
	return 0;
}




int vision::trainSVM(){
	Mat hist, image;									//vocabulary -> moved to private global
	vector<KeyPoint> keypoints;
	FileStorage fs(VOCABULARY_FILE.c_str(),FileStorage::READ);
	fs["Vocabulary"] >> vocabulary;
	fs.release();


	Ptr<DescriptorMatcher> matcher(new BruteForceMatcher<L2<float> >);
	//Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	BOWImgDescriptorExtractor bowde(extractor,matcher);
	bowde.setVocabulary(vocabulary);

	map<string,Mat> classes_training_data;
	classes_training_data.clear();
	ifstream ifs(TRAINING_DATA_FILE.c_str());
	int total_samples=0;
	FileStorage fs1(KEYPOINTS_FILE.c_str(),FileStorage::READ);
	FileNode keypointnode;
	String filepath,_class;

	do{
		ifs >> filepath >> _class;
		cout << filepath << " " << _class << endl;
		image = imread(filepath,CV_LOAD_IMAGE_GRAYSCALE);

		char key[20];
		sprintf(key,"_%d",total_samples);
		keypointnode = fs1[key];
		read(keypointnode,keypoints);
		keypoints_vector.push_back(keypoints);								// --- Vector of All the keypoints vectors of samples ---
		bowde.compute(image,keypoints,hist);

		if(classes_training_data.count(_class) == 0){
			classes_training_data[_class].create(0,hist.cols,hist.type());
		}

		classes_training_data[_class].push_back(hist);

		total_samples++;
	}while(!ifs.eof());

	ifs.close();
	fs1.release();


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
		classes_classifiers[class_].save(String(class_+ ".yml").c_str());
		cout << classes_classifiers.count(class_) << endl;
	}

	Mat testimage;
	testimage = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	keypoints = getKeyPoints(testimage);
	bowde.compute(testimage,keypoints,hist);
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
	featureDetector = (new SiftFeatureDetector());												// feature Detector
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


	return 0;
}


int vision::loadTrainingSet(){
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
				//cout << filename << endl;
				pair<string,Mat> tmp(class_,imread(filename,CV_LOAD_IMAGE_GRAYSCALE));
				training_set.insert(tmp);
			}
	}

	num_of_classes = classes.size();
	cout << "Total Number of Classes : " << num_of_classes << endl;
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

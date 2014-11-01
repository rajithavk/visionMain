/*
 * functions.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: romba
 */

#include "functions.hpp"

vision::vision(){};
vision::~vision(){};

int vision::buildVocabulary(String  filepath){
	DIR *dir,*subdir;
	char CWD[2049],ROOT[2049];
	Mat input , descriptor, featuresUnclustered , vocabulary;
	int count = 0;
	vector<KeyPoint> keypoints;
	ofstream trainingdata;
	struct dirent *entry,*imagefile;
	struct stat filestat;

	trainingdata.open("trainingdata.dat",ofstream::out);
	FileStorage fs2("keypoints.yml",FileStorage::WRITE);
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
	FileStorage fs("training_descriptors.yml",FileStorage::WRITE);
	fs << "training_descriptors" << featuresUnclustered;
	fs.release();
	cout << "Training Descriptors => " << ROOT << "/training_descriptors.yml" << endl;

	BOWKMeansTrainer bowtrainer(1000);
	bowtrainer.add(featuresUnclustered);
	cout << "Cluster BOW Features" << endl;
	vocabulary = bowtrainer.cluster();

	FileStorage fs1("vocabulary.yml",FileStorage::WRITE);
	fs1 << "Vocabulary" << vocabulary;
	fs1.release();
	cout << "Vocabulary => " << ROOT << "/vocabulary.yml" << endl;
	fs2.release();
	return 0;
}




int vision::trainSVM(){
	Mat hist, image;									//vocabulary -> moved to private global
	vector<KeyPoint> keypoints;
	FileStorage fs("vocabulary.yml",FileStorage::READ);
	fs["Vocabulary"] >> vocabulary;
	fs.release();


	Ptr<DescriptorMatcher> matcher(new BruteForceMatcher<L2<float> >);
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	BOWImgDescriptorExtractor bowde(extractor,matcher);
	bowde.setVocabulary(vocabulary);

	map<string,Mat> classes_training_data;
	classes_training_data.clear();
	ifstream ifs("trainingdata.dat");
	int total_samples=0;
	FileStorage fs1("keypoints.yml",FileStorage::READ);
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
		keypointsvec.push_back(keypoints);								// --- Vector of All the keypoints vectors of samples ---
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
		classes_classifiers[class_].save(String(class_+ ".xml").c_str());
		cout << classes_classifiers.count(class_) << endl;
	}

	Mat testimage;
	testimage = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	keypoints = calcKeyPoints(testimage);
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
	SiftDescriptorExtractor descriptorExtractor;										// feature extractor
	descriptorExtractor.compute(image,keypoints,descriptors);							// extract
	return descriptors;																	// return the descriptors for the input image
}

vector<KeyPoint> vision::calcKeyPoints(Mat image){
	vector<KeyPoint> keypoints;															// SIFT keypoints of the current image
	SiftFeatureDetector featureDetector;												// feature Detector
	featureDetector.detect(image,keypoints);
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

#include<iostream>

#include "PanoramicImage.h"

// dataset 1
#define PATH "../data/dolomites"
#define PATTERN "*.png"
#define FOV 54

// dataset 2
#define PATH1 "../data/kitchen"
#define PATTERN1 "*.bmp"
#define FOV1 66

// tuned parameters
#define RANSAC_THRESHOLD 10
#define RATIO 5 // must be >1

// write images of matches
#define DRAW_MATCHES true

using namespace std;

int main(int argc, char *argv[]){
	string path, pattern;
	double fov;

	// check argument
	if(argc < 3){
		cout<<"USAGE: panoramic_generator <path> <pattern> <FoV>"<<endl;
		cout<<"Using default path for data: "<<PATH<<endl;
		cout<<"Using default pattern: "<<PATTERN<<endl;
		cout<<"Using default FOV: "<<FOV<<endl;

		// use hardcoded arguments
		path = PATH;
		pattern = PATTERN;
		fov = FOV;
	}else{
		// use comman line arguments
		path = argv[1];
		pattern = argv[2];
		fov = atof(argv[3]);
	}

	// build panoramic image object
	cout<<"Loading images...";
	PanoramicImage* pi = new PanoramicImage(path, pattern, fov);
	cout<<"Done."<<endl;

	// make panoramic image
	cout<<"Doing some magic tricks..."<<endl;
	pi->doPanoramic(RATIO, RANSAC_THRESHOLD, DRAW_MATCHES);
	cout<<"Done."<<endl;

	// save result
	cv::Mat result = pi->getResult();
	cv::imwrite("0_result.png", result);
	cout<<"Images written to files."<<endl;

	cout<<"Exit."<<endl;
	return 0;
}

#include "PanoramicImage.h"

// number of match for BFMatcher
#define K_MATCH 1

using namespace std;
using namespace cv;

PanoramicImage::PanoramicImage(string path, string pattern, double fov){
	// initialize vars
	this->fov = fov;

	// find files
	std::vector<string> im_files;
	cv::utils::fs::glob(path, pattern, im_files);

	// read images
	for(uint i = 0; i < im_files.size(); i++){
		Mat im = imread(im_files.at(i));
		// project to cylindric coordinates
		Mat cyl_image = PanoramicUtils::cylindricalProj(im, fov/2);
		// save to member vector
		im_set.push_back(cyl_image);
	}

}


void PanoramicImage::doPanoramic(double ratio, double ransac_threshold, bool draw_matches){
	if(ratio <= 1){
		cout<<"Ratio must be >1."<<endl;
		exit(1);
	}
	vector<KeyPoint> prev_keypoints, keypoints;
	Mat prev_descriptors, descriptors;
	Ptr<ORB> orb = ORB::create();

	// the following matrix contains the average of pt.x of matches
	// from the left (col 1) and from the right (col 2)
	// First image isn't left cropped
	x_means = Mat::zeros(im_set.size(), 2, CV_32S);

	// manipulate all images
	for(uint i = 0; i < im_set.size(); i++){
		// save previous keypoints and descriptors
		vector<KeyPoint> prev_keypoints(keypoints);
		prev_descriptors = descriptors.clone();

		// compute keypoints
		orb->detect(im_set.at(i), keypoints);
		// compute keypoints' descriptors
		orb->compute(im_set.at(i), keypoints, descriptors);

		// compare consecutive images -------------------------
		
		// skip first image
		if(i == 0) continue;

		// hamming distance and crosscheck
		BFMatcher matcher = BFMatcher(NORM_HAMMING, true);

		// find matches
		// By documentation the matches are returned in the distance increasing order
		// (but it's not true, it's a bug?)
		vector<vector<DMatch>> matches;
		// prev_descriptors is the query set
		// descriptors is the train set
		matcher.knnMatch(prev_descriptors, descriptors, matches, K_MATCH);

		// find min_distance
		float min_distance = FLT_MAX;
		for(int j = 0; j < matches.size(); j++)
			if((matches.at(j).size() != 0) &&
					(matches.at(j).at(0).distance < min_distance))
						min_distance = matches.at(j).at(0).distance;

		// refine matches
		vector<DMatch> good_matches; // stripped vector with significative matches
		for(uint j = 0; j < matches.size(); j++)
			// there will be always at least a good match beacause of ratio > 1
			if((!matches.at(j).empty()) && (matches.at(j).at(0).distance <= ratio*min_distance)){
				good_matches.push_back(matches.at(j).at(0));
			}

		// draw matches
		if(draw_matches){
			Mat drawn_matches_image;
			drawMatches(im_set.at(i-1), prev_keypoints, im_set.at(i), keypoints, good_matches, drawn_matches_image);

			// write drawn_matches_image to disk
			char filename[32];
			sprintf(filename, "%d_matches.png", i);
			imwrite(filename,drawn_matches_image);
		}

		// obtain points from matches
		vector<Point2f> points_dx, points_sx;
		for(uint j = 0; j < good_matches.size(); j++){
			// obtain descriptors' index from match
			int trainIdx = good_matches.at(j).trainIdx;
			int queryIdx = good_matches.at(j).queryIdx;

			// obtain point from keypoints
			Point2f point_dx = prev_keypoints.at( queryIdx ).pt;
			Point2f point_sx = keypoints.at( trainIdx ).pt;

			// save points
			points_dx.push_back(point_dx);
			points_sx.push_back(point_sx);
		}

		// check homographies and discard outliers through mask
		Mat mask;
		findHomography(points_dx,points_sx, RANSAC, ransac_threshold, mask);

		// draw ransac matches
		if(draw_matches){
			Mat drawn_matches_image;
			drawMatches(im_set.at(i-1), prev_keypoints, im_set.at(i), keypoints, good_matches, drawn_matches_image,
					Scalar::all(-1), Scalar::all(-1), mask);

			// write matches_image to disk
			char filename[32];
			sprintf(filename, "%d_RANSAC_matches.png", i);
			imwrite(filename,drawn_matches_image);
		}

		// compute the mean of x matches

		int x_mean_dx = 0;
		int x_mean_sx = 0;
		int count = 0;
		for(int j = 0; j < good_matches.size(); j++)
			if(mask.at<bool>(0,j)){ // use outlier mask
				// right side of first image
				x_mean_dx += points_dx.at(j).x;

				// left side of second image
				x_mean_sx += points_sx.at(j).x;
				count++;
			}

		// compute the means
		if(count!= 0){
			x_mean_dx /= count;
			x_mean_sx /= count;

			// save mean x coord of matches
			x_means.at<int>(i-1,1) = x_mean_dx;
			x_means.at<int>(i,0) = x_mean_sx;
		}else{
			cout<<"Increase RANSAC_THRESHOLD or RATIO!"<<endl;
			exit(-1);
		}
	} // end for loop

	// last image isn't right cropped
	x_means.at<int>(im_set.size()-1,1) = im_set.at(im_set.size()-1).cols;

	// stick images
	stick();
}

void PanoramicImage::stick(){
	// compute panoramic image's dimension
	int height = im_set.at(0).rows;
	int width = 0;
	for(int i = 0; i < x_means.rows; i++){
		width += x_means.at<int>(i,1) - x_means.at<int>(i,0);

		// equalize all the image (does not improve performance)
		//equalizeHist(im_set.at(i), im_set.at(i));
	}

	// prepare image
	int type = im_set.at(0).type();
	result = Mat::zeros(height, width, type);

	// stick images
	Range rowRange(0, height);
	int x_mark = 0;
	for(int i = 0; i < im_set.size(); i++){
		// compute ranges
		Range colRange = Range(x_means.at<int>(i,0), x_means.at<int>(i,1));
		Range res_colRange(x_mark , x_mark + x_means.at<int>(i,1) - x_means.at<int>(i,0));

		// equalize part of the image (does not improve performance)
		//Mat tmp;
		//equalizeHist(im_set.at(i)(rowRange, colRange), tmp);
		//tmp.copyTo(result(rowRange, res_colRange));

		// copy the image to the big matrix
		im_set.at(i)(rowRange, colRange).copyTo(result(rowRange, res_colRange));

		// update mark_x
		x_mark += colRange.size();
	}
}

Mat PanoramicImage::getResult(){
	return result;
}

#include <string>
#include <vector>
#include <iostream>
#include <cfloat>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "panoramic_utils.h"

class PanoramicImage{
// member vars
private:
	std::vector<cv::Mat> im_set;
	cv::Mat result;
	double fov;
	cv::Mat x_means;

// member funcs
public:
	PanoramicImage(std::string path, std::string pattern, double fov);
	void doPanoramic(double ratio, double ransac_threshold, bool draw_matches);
	cv::Mat getResult();

private:
	void stick();
};

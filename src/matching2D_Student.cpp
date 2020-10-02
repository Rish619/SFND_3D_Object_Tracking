
#include <numeric>
#include "matching2D.hpp"

using namespace std;

enum detector
{
  FAST,
  BRIEF,
  BRISK,
  ORB,
  FREAK,
  AKAZE,
  SIFT
};

std::map<string, detector> levels;

void register_levels()
{
  levels["FAST"] = FAST;
  levels["BRIEF"] = BRIEF;
  levels["BRISK"] = BRISK;
  levels["ORB"] = ORB;
  levels["FREAK"] = FREAK;
  levels["AKAZE"] = AKAZE;
  levels["SIFT"] = SIFT;
}

// Matching methods to match the best keypoints based on various matching algorithms and selector types
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
  // Matcher initialization and configuring
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType == "MAT_BF")
  {

    int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
    matcher = cv::BFMatcher::create(normType, crossCheck);
  }
  else if (matcherType == "MAT_FLANN")
  {
    if (descSource.type() != CV_32F || descRef.type() != CV_32F)
    {
      // Converting binary descriptors to floating type due to a bug in OpenCV
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }
    matcher = cv::BFMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }

  // Matching
  if (selectorType == "SEL_NN")
  {

    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
  }
  else if (selectorType == "SEL_KNN")
  {

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, 2);
    double minDescDistRatio = 0.8;

    for (auto knn_match : knn_matches)
    {
      if (knn_match[0].distance < minDescDistRatio * knn_match[1].distance)
      {
        matches.push_back(knn_match[0]);
      }
    }
  }
  cout << "Number of keypoints matched: " << matches.size() << endl;
}

// Descriptors to uniquely identify keypoints and there selection based on the input provided by the user
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  register_levels();
  int threshold = 30;        // FAST/AGAST detection threshold score.
  int octaves = 3;           // detection octaves (use 0 to do single scale)
  float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

  if (levels.find(descriptorType) != levels.end())
  {
    switch (levels[descriptorType])
    {

    case BRISK:
      extractor = cv::BRISK::create(threshold, octaves, patternScale);
      break;

    case BRIEF:
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
      break;
    case ORB:
      extractor = cv::ORB::create();
      break;
    case FREAK:
      extractor = cv::xfeatures2d::FREAK::create();
      break;
    case AKAZE:
      extractor = cv::AKAZE::create();
      break;
    case SIFT:
      extractor = cv::xfeatures2d::SIFT::create();
      break;
    }
  }
  else
  {
    std::cerr << "Descriptor Type Not found/Implemented!!!" << std::endl;
    return 0;
  }
  // Here is the feature description logic
  double description_time = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  description_time = ((double)cv::getTickCount() - description_time) / cv::getTickFrequency();
  std::cout << descriptorType << " descriptor extraction in " << 1000 * description_time / 1.0 << " ms" << std::endl;
  return description_time;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // compute detector parameters based on image size
  int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double detection_time = (double)cv::getTickCount();
  vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it)
  {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  detection_time = ((double)cv::getTickCount() - detection_time) / cv::getTickFrequency();
  std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * detection_time / 1.0 << " ms" << std::endl;

  return detection_time;
}

// Detect keypoints in image using Harris detector
double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

  int block_size = 2;     // for every pixel, a blockSize x blockSize neighbourhood is considered
  int aperture_size = 3;  // Sobel operator aperture size
  int min_response = 100; // minimum Response value required to qualify as a corner from the Response matrix
  double k = 0.04;        // Harris constant

  // Detect Harris corners and normalize output
  cv::Mat dist, dist_norm, dist_norm_scaled;
  dist = cv::Mat::zeros(img.size(), CV_32FC1);

  double detection_time = (double)cv::getTickCount();
  // Harris corner detection is applied under this

  cv::cornerHarris(img, dist, block_size, aperture_size, k, cv::BORDER_DEFAULT);
  cv::normalize(dist, dist_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dist_norm, dist_norm_scaled);

  // Perform non-maxima suppression (NMS)

  double max_overlap = 0.0; // max. permissible overlap between two features

  for (size_t j = 0; j < dist_norm.rows; j++)
  {
    for (size_t k = 0; k < dist_norm.cols; k++)
    {
      int response = (int)dist_norm.at<float>(j, k);
      if (response > min_response)
      { // Consider corners or points above a certain threshold
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(k, j);
        newKeyPoint.size = 2 * aperture_size;
        newKeyPoint.response = response;

        // Perform non-maxima suppression (NMS) around the new key point

        bool bOverlap = false;
        for (auto keypoint : keypoints)
        {
          double kpt_overlap = cv::KeyPoint::overlap(newKeyPoint, keypoint);
          if (kpt_overlap > max_overlap)
          {
            bOverlap = true;
            if (newKeyPoint.response > keypoint.response)
            {

              keypoint = newKeyPoint;
              break;
            }
          }
        }
        if (!bOverlap)
        {

          keypoints.push_back(newKeyPoint);
        }
      }
    } // eof loop over cols
  }   // eof loop over rows
  detection_time = ((double)cv::getTickCount() - detection_time) / cv::getTickFrequency();
  std::cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * detection_time / 1.0 << " ms.\n";
  return detection_time;
}

// Detectors FAST, BRISK, ORB, AKAZE, SIFT selection and creation
double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{

  cv::Ptr<cv::FeatureDetector> detector;
  register_levels();
  int threshold = 30;
  bool bNMS = true;

  if (levels.find(detectorType) != levels.end())
  {
    switch (levels[detectorType])
    {
    case FAST:
      detector = cv::FastFeatureDetector::create(threshold, bNMS, cv::FastFeatureDetector::TYPE_9_16);
      break;
    case BRISK:
      detector = cv::BRISK::create();
      break;
    case ORB:
      detector = cv::ORB::create();
      break;
    case AKAZE:
      detector = cv::AKAZE::create();
      break;
    case SIFT:
      detector = cv::xfeatures2d::SIFT::create();
      break;
    }
  }
  else
  {
    std::cerr << "Detector Type Not found/Implemented!!!" << std::endl;
    return 0;
  }

  // Here is keypoint detection logic
  double detection_time = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  detection_time = ((double)cv::getTickCount() - detection_time) / cv::getTickFrequency();
  std::cout << detectorType << " detection with n = " << keypoints.size() << " keypoint in " << 1000 * detection_time / 1.0 << " ms.\n";
  return detection_time;
}
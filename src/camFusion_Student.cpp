
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);
    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox_c, BoundingBox &boundingBox_p, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
    double dist_mean = 0;
    vector<cv::DMatch> kptMatches_roi;

    float shrink_factor = 0.15;
    cv::Rect smaller_box_c, smaller_box_p;
    // shrinking current frame bounding box to avoid having outlier points around the edges
    smaller_box_c.x = boundingBox_c.roi.x + shrink_factor * boundingBox_c.roi.width / 2.0;
    smaller_box_c.y = boundingBox_c.roi.y + shrink_factor * boundingBox_c.roi.height / 2.0;
    smaller_box_c.width = boundingBox_c.roi.width * (1 - shrink_factor);
    smaller_box_c.height = boundingBox_c.roi.height * (1 - shrink_factor);

    // shrinking previous frame bounding box slightly to avoid having outlier points around the edges
    smaller_box_p.x = boundingBox_p.roi.x + shrink_factor * boundingBox_p.roi.width / 2.0;
    smaller_box_p.y = boundingBox_p.roi.y + shrink_factor * boundingBox_p.roi.height / 2.0;
    smaller_box_p.width = boundingBox_p.roi.width * (1 - shrink_factor);
    smaller_box_p.height = boundingBox_p.roi.height * (1 - shrink_factor);

    //getting the matches located within  current_frame_boundingBox and previous_frame_boundingBox
    for (auto kptmatch : kptMatches)
    {
        cv::KeyPoint Train = kptsCurr.at(kptmatch.trainIdx);
        auto Train_point = cv::Point(Train.pt.x, Train.pt.y);

        cv::KeyPoint Query = kptsPrev.at(kptmatch.queryIdx);
        auto Query_point = cv::Point(Query.pt.x, Query.pt.y);

        // checking whether the point is inside the shrinked bounding box in both previous and current frames
        if (smaller_box_c.contains(Train_point) && smaller_box_p.contains(Query_point))
            kptMatches_roi.push_back(kptmatch);
    }

    //Mean of all distances between the match points pairs from previous and current frame
    for (auto kptMatch_roi : kptMatches_roi)
    {
        dist_mean += cv::norm(kptsCurr.at(kptMatch_roi.trainIdx).pt - kptsPrev.at(kptMatch_roi.queryIdx).pt);
    }
    if (kptMatches_roi.size() > 0)
        dist_mean = dist_mean / kptMatches_roi.size();
    else
        return;

    //keep the matches  distance < dist_mean * 1.5
    double threshold = dist_mean * 1.5;
    for (auto kptMatch_roi : kptMatches_roi)
    {
        float dist = cv::norm(kptsCurr.at(kptMatch_roi.trainIdx).pt - kptsPrev.at(kptMatch_roi.queryIdx).pt);
        if (dist < threshold)
            boundingBox_c.kptMatches.push_back(kptMatch_roi);
    }

    cout << "curr_bbx_matches_size: " << boundingBox_c.kptMatches.size() << endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
    vector<double> distRatios; // will store the distance ratios between all the keypoint matches in current frame and previous frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        cv::KeyPoint OuterkpCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint OuterkpPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0; // min. required distance
            cv::KeyPoint InnerkpCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint InnerkpPrev = kptsPrev.at(it2->queryIdx);
            // computing distances between all possible keypoints from the given keypoint
            double dist_Curr = cv::norm(OuterkpCurr.pt - InnerkpCurr.pt);
            double dist_Prev = cv::norm(OuterkpPrev.pt - InnerkpPrev.pt);
            if (dist_Prev > std::numeric_limits<double>::epsilon() && dist_Curr >= minDist)
            { // avoid division by zero
                double distRatio = dist_Curr / dist_Prev;
                distRatios.push_back(distRatio); //computing distances ratios
            }
        }
    }
    // continue only if the distRatios have atleast one pair
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    sort(distRatios.begin(), distRatios.end());

    long median_Index = floor(distRatios.size() / 2.0);
    double median_DistRatio = distRatios.size() % 2 == 0 ? (distRatios[median_Index - 1] + distRatios[median_Index]) / 2.0 : distRatios[median_Index]; // computing median distance ratio to remove anomally

    double dT = 1 / frameRate;
    TTC = -dT / (1 - median_DistRatio);
}

// This is a helper function that sorts all the lidar points in the ascending order based on their x coordinate which is the direction drive
void sort_lidarpoints_inX(std::vector<LidarPoint> &lidarPoints)
{
    // This is the sort from standard template library with lamda being passed with lidarpoints those need to be sorted in ascending order
    sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint first, LidarPoint second) {
        return first.x < second.x; // Sorting based on the x coordinates of the lidarpoints
    });
}

void computeTTCLidar(std::vector<LidarPoint> &PrevlidarPoints,
                     std::vector<LidarPoint> &CurrlidarPoints, double frameRate, double &TTC)
{
    double ttc_lidar_calculation_time = (double)cv::getTickCount();

    // For each current and previous frame, taking the median point of the lidar points for the distance estimation
    // If the performance is suffering due to the median calculation for the entire lidarpoint dataset, take the median of the subset of the lidarpoints
    sort_lidarpoints_inX(PrevlidarPoints);
    sort_lidarpoints_inX(CurrlidarPoints);

    double d0 = PrevlidarPoints.size() % 2 == 0 ? (PrevlidarPoints[(PrevlidarPoints.size() / 2) - 1].x + PrevlidarPoints[PrevlidarPoints.size() / 2].x) / 2.0 : PrevlidarPoints[PrevlidarPoints.size() / 2].x; // compute median lidarpoint to remove outlier influence
    //double d0 = PrevlidarPoints[PrevlidarPoints.size()/2].x;
    //double d1 = CurrlidarPoints[CurrlidarPoints.size()/2].x;
    double d1 = CurrlidarPoints.size() % 2 == 0 ? (CurrlidarPoints[(CurrlidarPoints.size() / 2) - 1].x + CurrlidarPoints[CurrlidarPoints.size() / 2].x) / 2.0 : CurrlidarPoints[CurrlidarPoints.size() / 2].x; // compute median lidarpoint to remove outlier influence

    // This calculation is based on the constant velocity model(However, constant acceleration and more complex models are realistic)
    // TTC = d1 * delta_t / (d0 - d1)
    // where: d0 is the previous frame's distance from the ego car to the preceding vehicle's rear bumper
    //        d1 is the current frame's distance from the ego car to the preceding vehicle's rear bumper
    //        delta_t is the time difference between two consecutive frames (1 / frameRate)
    // Note: this implementation of the time to collision using Lidar points doesn't take the distance between the Lidar origin and the ego vehicles front bumper! in account
    // It also does not account the hump caused due to the curvature of the rear bump of the preceding vehicle.
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);

    ttc_lidar_calculation_time = ((double)cv::getTickCount() - ttc_lidar_calculation_time) / cv::getTickFrequency();
    cout << " ttc lidar calculation took " << 1000 * ttc_lidar_calculation_time / 1.0 << " ms.\n";
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    double matching_time = (double)cv::getTickCount();
    int pre = prevFrame.boundingBoxes.size();
    int cur = currFrame.boundingBoxes.size();
    int pt_counts[pre][cur] = {};
    for (auto match : matches)
    {
        cv::KeyPoint Query = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint Train = currFrame.keypoints[match.trainIdx];
        auto Query_point = cv::Point(Query.pt.x, Query.pt.y);
        auto Train_point = cv::Point(Train.pt.x, Train.pt.y);
        bool Query_found = false, Train_found = false;

        std::vector<int> Query_boxes_id, Train_boxes_id;

        for (int i = 0; i < pre; i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(Query_point))
            {
                Query_found = true;
                Query_boxes_id.push_back(i);
            }
        }
        for (int i = 0; i < cur; i++)
        {
            if (currFrame.boundingBoxes[i].roi.contains(Train_point))
            {
                Train_found = true;
                Train_boxes_id.push_back(i);
            }
        }

        if (Query_found && Train_found)
        {
            for (auto id_prev : Query_boxes_id)
                for (auto id_curr : Train_boxes_id)
                    pt_counts[id_prev][id_curr] += 1;
        }
    }

    for (int i = 0; i < pre; i++)
    {
        int max_count = 0;
        int id_max = 0;
        for (int k = 0; k < cur; k++)
            if (pt_counts[i][k] > max_count)
            {
                max_count = pt_counts[i][k];
                id_max = k;
            }
        bbBestMatches[i] = id_max;
    }

    matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();
    cout << " match bounding box took " << 1000 * matching_time / 1.0 << " ms.\n";
}

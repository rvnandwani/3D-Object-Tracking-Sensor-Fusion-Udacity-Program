
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
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

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


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
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

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<double> distances;
    for(auto it = kptMatches.begin(); it != kptMatches.end() ; it++){
        cv::KeyPoint prevKP = kptsPrev[(*it).queryIdx];
        cv::KeyPoint currKP = kptsCurr[(*it).trainIdx];
        if(boundingBox.roi.contains(currKP.pt)){
            distances.push_back(cv::norm(prevKP.pt-currKP.pt));
        }
    }
    double threshold = 0;
    for(int i=0;i<distances.size();i++){
        threshold = threshold+distances[i];
    }
    threshold  = threshold/distances.size();
    for(auto it = kptMatches.begin(); it != kptMatches.end() ; it++){
        cv::KeyPoint prevKP = kptsPrev[(*it).queryIdx];
        cv::KeyPoint currKP = kptsCurr[(*it).trainIdx];
        if(boundingBox.roi.contains(currKP.pt) && cv::norm(prevKP.pt-currKP.pt)<threshold*1.3){
            boundingBox.keypoints.push_back(currKP);
            boundingBox.kptMatches.push_back((*it));
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    double t = 1/frameRate;
    vector<double> distRatio;
    double minDist = 100.0;
    for(auto it = kptMatches.begin() ; it != kptMatches.end();it++){
        cv::KeyPoint prevOuterKP = kptsPrev[(*it).queryIdx];
        cv::KeyPoint currOuterKP = kptsCurr[(*it).trainIdx];
        for(auto itc = kptMatches.begin()+1 ; itc != kptMatches.end();itc++){
            cv::KeyPoint prevInnerKP = kptsPrev[(*itc).queryIdx];
            cv::KeyPoint currInnerKP = kptsCurr[(*itc).trainIdx];
            double prevDist = cv::norm(prevOuterKP.pt - prevInnerKP.pt);
            double currDist = cv::norm(currOuterKP.pt - currInnerKP.pt);
            if(prevDist>std::numeric_limits<double>::epsilon() && currDist>= minDist){
                distRatio.push_back(currDist/prevDist);
            }
        }
    }
    sort(distRatio.begin(),distRatio.end());
    double center = floor(distRatio.size()/2.0);
    TTC = -t/(1-distRatio[center]);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double t = 1/frameRate;
    double avgPrevx = 0;
    double avgCurrx = 0;
    // double minx=lidarPointsCurr[0].x;
    // int tot=0;
    double threshold = 0.8;
    for (auto it = lidarPointsPrev.begin() ; it != lidarPointsPrev.end() ; it++){
        // if((*it).x > threshold * avgPrevx){
            avgPrevx = avgPrevx + (*it).x;
            // tot = tot + 1;
        // }
    }
    if (lidarPointsPrev.size()>1){
        avgPrevx = avgPrevx / lidarPointsPrev.size();
    }
    // tot=0;
    float minx=1e8;
    for (auto it = lidarPointsCurr.begin() ; it != lidarPointsCurr.end() ; it++){
        // minx = minx<(*it).x ? minx : (*it).x;
        // if((*it).x > threshold * avgCurrx){
            avgCurrx = avgCurrx + (*it).x;
            // tot = tot + 1;
            // if((*it).x<minx){
            //     minx =(*it).x; 
            // }
        // }
    }
    if (lidarPointsCurr.size()>1){
        avgCurrx = avgCurrx / lidarPointsCurr.size();
    }
    for (auto it = lidarPointsCurr.begin() ; it != lidarPointsCurr.end() ; it++){
        if ((*it).x>0 && (*it).x > threshold * avgCurrx && (*it).x<minx){
            minx =(*it).x; 
        }
    }
    cout<<"minx = "<<minx<<endl;
    TTC = (minx*t)/(avgPrevx-avgCurrx);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    cv::KeyPoint prevKP, currKP;
    vector<int> prevID;
    vector<int> currID;
    int prevBoxCount = prevFrame.boundingBoxes.size();
    int currBoxCount = currFrame.boundingBoxes.size();

    vector<vector<int>> BBMatch(prevBoxCount, vector<int> (currBoxCount,0));
    for(auto it=matches.begin() ; it!= matches.end() ; it++){
        prevKP = prevFrame.keypoints[(*it).queryIdx];
        currKP = currFrame.keypoints[(*it).trainIdx];

        prevID.clear();
        currID.clear();

        for(auto itb = prevFrame.boundingBoxes.begin() ; itb != prevFrame.boundingBoxes.end() ; itb++){
            if((*itb).roi.contains(prevKP.pt)){
                prevID.push_back((*itb).boxID);
            }
        }
        for(auto itb = currFrame.boundingBoxes.begin() ; itb != currFrame.boundingBoxes.end() ; itb++){
            if((*itb).roi.contains(currKP.pt)){
                currID.push_back((*itb).boxID);
            }
        }
        for(auto prev:prevID){
            for(auto curr:currID){
                BBMatch[prev][curr]+=1;
            }
        }
    }

    for(int i=0;i<BBMatch.size();i++){
        int max=0,curr=0;
        for(int j=0;j<BBMatch[i].size();j++){
            if(BBMatch[i][j]>max){
                max = BBMatch[i][j];
                curr=j;
            }
        }
        bbBestMatches[i] = curr;
        // cout<<"prev = "<<i<<" ; curr = "<<curr<<endl;
    }
}

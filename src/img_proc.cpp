#include "func_header.h"
#include <string>
#include <unistd.h>
using namespace cv;
using namespace cv::xfeatures2d;

#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}


int findPic(imageTransporter& imgTransport, vector<cv::Mat>& imgs_track, int iteration){
  cv::namedWindow("view");
  int foundPic;
  // char imgname[50];
  printf("Check error 1\n");
  cv::Mat video;

  video = imgTransport.getImg();
  printf("Check error 2\n");  
  if(!video.empty()){
    printf("Check error 3\n");
	  //fill with your code
	  //feature2D_homography("/home/lucasius/MIE443/catkin_ws/src/mie443_contest2/pics/tag1.jpg", "/home/lucasius/MIE443/catkin_ws/src/mie443_contest2/pics/tag3.jpg");
    //feature2D_homography(imgs_track.at(1), video);
    cv::imshow("view", video);
    // sprintf(imgname, "/home/turtlebot/catkin_ws/src/mie443_contest2/src/tag%d.jpg", iteration);
    printf("Check error 4\n");
    cv::imwrite("/home/turtlebot/catkin_ws/src/mie443_contest2/src/tag"+ patch::to_string(iteration) +".jpg", video);
    cv::waitKey(10);
    video.release();
    cv::waitKey(10);
    }
    printf("Check error 5\n");
    printf("Check error 6\n");
    foundPic = 0;
  	return foundPic;
}

/** @function main */
int feature2D_homography (cv::Mat img_object, cv::Mat img_scene )//(const char *image1, const char *image2 )
{
  //Mat img_object = imread( image1, IMREAD_GRAYSCALE );
  //Mat img_scene = imread( image2, IMREAD_GRAYSCALE );

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  // printf("REACH 1\n");

//   SurfFeatureDetector detector( minHessian ); 
  Ptr<SURF> detector = SURF::create( minHessian );
  
  // printf("REACH 2\n");
  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector->detect( img_object, keypoints_object );
  detector->detect( img_scene, keypoints_scene );
  // printf("REACH 3\n");

  //-- Step 2: Calculate descriptors (feature vectors)
  Ptr<SurfDescriptorExtractor> extractor = SURF::create();
  // printf("REACH 4\n");

  Mat descriptors_object, descriptors_scene;

  extractor->compute( img_object, keypoints_object, descriptors_object );
  extractor->compute( img_scene, keypoints_scene, descriptors_scene );
  // printf("REACH 5\n");

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );
  // printf("REACH 6\n");

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );

  waitKey(10);
  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
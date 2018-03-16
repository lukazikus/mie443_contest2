#include "func_header.h"
#include <string>
#include <unistd.h>
using namespace cv;
using namespace cv::xfeatures2d;

#include <sstream>

#include <cmath>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

//Added by Andrew to calculate SSD
double getSimilarity(cv::Mat A, cv::Mat B ) {
  //resize(src,dst,size);
  //printf("styupid1444444444444\n");
  //cv::imshow("A", A);
  // cv::imwrite("/home/turtlebot/catkin_ws/src/mie443_contest2/src/A.jpg", A); // CHANGE IMAGE PATH
  //cv::imshow("B", B);
  // cv::imwrite("/home/turtlebot/catkin_ws/src/mie443_contest2/src/B.jpg", B); // CHANGE IMAGE PATH
  
  // printf("%d",A.sameSize(B);
  // printf("\n");
  // printf("%d",A.type());
  // printf("\n");
  // printf("%d",B.type());
  // printf("\n");

  if ( A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols ) {
      // Calculate the L2 relative error between images.
      // printf("styupid15\n");
      double errorL2 = norm( A, B, CV_L2 );
      // printf("styupid16\n");
      // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
      double similarity = errorL2 / (double)( A.rows * A.cols );
      // printf("styupid17\n");
      return similarity;
  }
  else {
      //Images have a different size
      return 100000000.0;  // Return a bad value
  }
}

//int findPic(imageTransporter& imgTransport, vector<cv::Mat>& imgs_track, int iteration){// actual function
int findPic(imageTransporter &imgTransport, cv::Mat &imgs_track, int iteration){//modified function
//int findPic(cv::Mat& imgTransport, cv::Mat& imgs_track, int iteration){
  // printf("styupid1");
  cv::namedWindow("view");
  int foundPic;//1:raisin bran, 2: cinnimen roast crunch, 3: rice krispies, 4:blank image
  // char imgname[50];
  cv::Mat video;
  double SSD1, SSD2, SSD3;
  // printf("styupid3");
  video = imgTransport.getImg(); // For actual funcfor (int i = 0; i < scene_transformed,size(),height; i++)tion
  //video = imgTransport; // Only for debugging  
  if(!video.empty()){
	  // fill with your code
    Mat video_grey;
    cv::cvtColor(video, video_grey, cv::COLOR_BGR2GRAY);
    // printf("styupid2");
    // tag 1
    imgs_track = imread( "/home/turtlebot/catkin_ws/src/mie443_contest2/pics/tag1.jpg", IMREAD_GRAYSCALE );
    // printf("styupid7");
    SSD1 = feature2D_homography(imgs_track, video_grey);
    //cv::threshold(imgs_track, imgs_bw, 128.0, 255.0, THRESH_BINARY);
    //double SSD1 = feature2D_homography(imgs_bw, video_bw);
    // printf("styupid4");
    // tag 2
    imgs_track = imread( "/home/turtlebot/catkin_ws/src/mie443_contest2/pics/tag2.jpg", IMREAD_GRAYSCALE );
    SSD2 = feature2D_homography(imgs_track, video_grey);
    //cv::threshold(imgs_track, imgs_bw, 128.0, 255.0, THRESH_BINARY);
    //double SSD2 = feature2D_homography(imgs_bw, video_bw);

    // tag 3
    imgs_track = imread( "/home/turtlebot/catkin_ws/src/mie443_contest2/pics/tag3.jpg", IMREAD_GRAYSCALE );
    SSD3 = feature2D_homography(imgs_track, video_grey);
    //cv::threshold(imgs_track, imgs_bw, 128.0, 255.0, THRESH_BINARY);
    //double SSD3 = feature2D_homography(imgs_bw, video_bw);
    
    //cv::imwrite("image_bw.jpg", imgs_bw);

    printf("SSD1: %lf, %lf, %lf\n", SSD1, SSD2, SSD3);
    double error_threshold = 0.292;

    if ((SSD1 > 0 && SSD2 > 0 && SSD3 > 0)&& SSD1 < error_threshold || SSD2 < error_threshold || SSD3 < error_threshold){
      if (SSD1 < error_threshold && SSD1 == min(SSD1, min(SSD2, SSD3))){
        printf("This is raisin bran!!\n");
        foundPic = 1;
      }else if (SSD2 < error_threshold && SSD2 == min(SSD1, min(SSD2, SSD3))){
        printf("This is cinnamon toast crunch!!!\n");
        foundPic = 2;
      }else if (SSD3 < error_threshold && SSD3 == min(SSD1, min(SSD2, SSD3))){
        printf("This is rice krispies!!!\n");
        foundPic = 3;
      }
    }else if (SSD1 > 0 && SSD2 > 0 && SSD3 > 0){
      printf("This is a blank image!!!\n");
      foundPic = 4;
    }else{
      printf("ERROROROROROROROROR!!!\n");
      foundPic = -1;
    }
    
    cv::imshow("view", video);
    //change address*******************************************************************************************
    cv::imwrite("/home/turtlebot/catkin_ws/src/mie443_contest2/src/tag"+ patch::to_string(iteration) +".jpg", video); // CHANGE IMAGE PATH
    cv::waitKey(10);
    video.release();
    cv::waitKey(10);
  }
  // foundPic = 0;
  return foundPic;
}

/** @function main */
double feature2D_homography (cv::Mat img_object, cv::Mat img_scene )//(const char *image1, const char *image2 )
{
  // Mat img_object = imread( image1, IMREAD_GRAYSCfor (int i = 0; i < scene_transformed,size(),height; i++)ALE );
  // Mat img_scene = imread( image2, IMREAD_GRAYSCALE );

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  // printf("REACH 1\n");

//   SurfFeatureDetector detector( minHessian ); 
  Ptr<SURF> detector = SURF::create( minHessian );
  // printf("styupid8");
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
  //printf("styupid9");
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
  //printf("styupid10");
  //printf("-- Max dist : %f \n", max_dist );
  //printf("-- Min dist : %f \n", min_dist );
  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )double SSD = getSimilarity(img_object, scene_transformed);
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
  //printf("styupid11");
  Mat H = findHomography( scene, obj, RANSAC ); // Find transformation from object to scene
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
  Mat scene_transformed;
  warpPerspective(img_scene, scene_transformed, H, img_object.size()); // Warp scene image to zoom into object portion
  //change address*******************************************************************************************
  //cv::imwrite("/home/andrew/catkin_ws/src/mie443_contest2/src/tag_from_scene.jpg", scene_transformed); // CHANGE IMAGE PATH
  
  //-- Show detected matchesdouble
  //imshow( "Good Matches & Object detection", img_matches );
  

  // Andrew: Calculate the SSD between images (wraped scene to object:scene_transformed and view:img_object)
  cv::Mat img_object_bw;
  cv::Mat scene_transformed_bw;
  // Convert images to binary
  cv::threshold(scene_transformed, scene_transformed_bw, 128.0, 255.0, THRESH_BINARY);
  cv::threshold(img_object, img_object_bw, 128.0, 255.0, THRESH_BINARY);
  //double SSD = getSimilarity(img_object, scene_transformed);
  double SSD = getSimilarity(img_object_bw, scene_transformed_bw);
  //imshow( "Warped Scene to Object_2", img_object_bw );
  //imshow( "Warped Scene to Object", scene_transformed_bw );
  
  waitKey(10);
  return SSD;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
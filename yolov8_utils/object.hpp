#ifndef __OBJECT__
#define __OBJECT__

#include <iostream>
#include <algorithm>
#include <vector>
#include <assert.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "point.hpp"
#include "bounding_box.hpp"

using namespace std;

class BoundingBox;

class Object
{
 public:
  Object();
  ~Object();

  ///////////////////////////
  /// Member Functions
  //////////////////////////
  int getStatus();
  void init(int _frameStamp);
  void updateStatus(int status);
  void updateBoundingBox(BoundingBox &_box);
  void updatePointCenter(Point &_point);
  void updateBoundingBoxList(vector<BoundingBox> &_boxList);

  //
  void updateKeypoint(vector<cv::KeyPoint> &kpt);
  void updateDescriptor(cv::Mat &desc);
  void updateImage(cv::Mat &img);

  BoundingBox predNextBoundingBox(BoundingBox &currBox, int frameInterval, int frameDisappear, int imgH, int imgW);
  BoundingBox getScaledBoundingBox(float r, int imgH, int imgW);
  void getPrevPredBoundingBox(BoundingBox &prevPredBox);

  ///////////////////////////
  /// Member Variables
  //////////////////////////
  int id = -1;                                // Human ID
  int status = 0;                             // 0: deactivate / 1: activate
  BoundingBox bbox = \
    BoundingBox(-1, -1, -1, -1, -1);          // Bounding Box (x1, y1, x2, y2, label)
  vector<BoundingBox> bboxList;               // Bounding Box list
  vector<BoundingBox> smoothedBBoxList;       // Smoothed Bounding Box list
  Point pCenter = Point(-1, -1);              // Bounding Box's center point
  int disappearCounter = 0;                   // Disappear frame counter
  int aliveCounter = 0;                       // How long dose object show in display
  bool discardPrevBoundingBox = true;         // False to keep object's previous bounding box
  float distanceToCamera = -1;                 // Distance to camera
  float preDistanceToCamera = -1;             // Distance to camera
  float currTTC = -1;
  int ttcCounter = 0;
  bool needWarn = false;                      // Does this object need to be warning?

  // Appearance Matching
  vector<cv::KeyPoint> m_prevKpts;
  vector<cv::KeyPoint> m_currKpts;
  cv::Mat m_prevDesc;
  cv::Mat m_currDesc;
  cv::Mat m_prevImg; // debug
  cv::Mat m_currImg; // debug

  int m_frameStamp;

  // FCW
  vector<float> m_distanceList; //TODO:
  vector<float> m_ttcList;

  // Predict next bounding boxes
  BoundingBox m_lastDetectBoundingBox = BoundingBox(-1, -1, -1, -1, -1);
  BoundingBox m_lastPredBoundingBox = BoundingBox(-1, -1, -1, -1, -1);
  BoundingBox m_prevPredBoundingBox = BoundingBox(-1, -1, -1, -1, -1);
  BoundingBox m_currPrevBoundingBox = BoundingBox(-1, -1, -1, -1, -1);

 private:

  vector<float> _getCenterPointVelocity(int frameInterval);
  float _getAspectRatioVelocity(int frameInterval);
  float _getHeightVelocity(int frameInterval);



  // Debug
  int debugMode = false;
};

#endif


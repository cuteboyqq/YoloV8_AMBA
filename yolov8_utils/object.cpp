#include "object.hpp"


/////////////////////////
// public member functions
////////////////////////
Object::Object()
{};

Object::~Object()
{};


void Object::init(int _frameStamp)
{
  status = 0;
  pCenter.x = -1;
  pCenter.y = -1;
  bbox = BoundingBox(-1, -1, -1, -1, -1);
  bboxList.clear();
  aliveCounter = 0;
  discardPrevBoundingBox = true;
  distanceToCamera = -1;
  preDistanceToCamera = -1;
  ttcCounter = 0;
  currTTC = -1;

  m_distanceList.clear();
  m_ttcList.clear();

  bbox.setFrameStamp(_frameStamp);
  needWarn = false;
}


void Object::updateStatus(int _status)
{
  status = _status;
}


int Object::getStatus()
{
  return status;
}


void Object::updateBoundingBox(BoundingBox &_bbox)
{
  bbox = _bbox;
}


void Object::updatePointCenter(Point &_point)
{
  pCenter.x = _point.x;
  pCenter.y = _point.y;
}


void Object::updateBoundingBoxList(vector<BoundingBox> &_boxList)
{
  bboxList = _boxList;
}


BoundingBox Object::predNextBoundingBox(
  BoundingBox &currBox, int frameInterval, int frameDisappear, int imgH, int imgW)
{
  Point centerPoint(0, 0);
  float aspectRatio = 0;
  float height = 0;
  if ((frameDisappear > 0) && (bboxList.size() > 0))
  {
    centerPoint = bboxList[bboxList.size()-1].getCenterPoint();
    aspectRatio = bboxList[bboxList.size()-1].getAspectRatio();
    height = bboxList[bboxList.size()-1].getHeight();
  }
  else
  {
    centerPoint = currBox.getCenterPoint();
    aspectRatio = currBox.getAspectRatio();
    height = currBox.getHeight();
  }

  float velHeight = _getHeightVelocity(frameInterval);
  float velAspect = _getAspectRatioVelocity(frameInterval);
  vector<float> velCenterPoint = _getCenterPointVelocity(frameInterval);

  float t;
  if (frameDisappear > 0)
  {
    t = 0.5; // + (1/(float)frameInterval) * frameDisappear;
    velAspect = 0;
  }
  else
  {
    t = 0.5;
  }

  //
  vector<float> nextCenterPoint = {
    centerPoint.x + velCenterPoint[0] * t,
    centerPoint.y + velCenterPoint[1] * t
  };

  float nextAspectRatio = aspectRatio + velAspect*t;
  int nextHeight = height + velHeight*t;
  int nextWidth = nextHeight / nextAspectRatio;

  //
  if ((frameDisappear > frameInterval) && (m_lastPredBoundingBox.getCenterPoint().x != -1))
  {
    int lastPredHeight = m_lastPredBoundingBox.getHeight();
    int lastPredWidth = m_lastPredBoundingBox.getWidth();
    Point lastCenterPoint = m_lastPredBoundingBox.getCenterPoint();

    nextHeight = lastPredHeight;
    nextWidth = lastPredWidth;
    nextCenterPoint[0] = lastCenterPoint.x;
    nextCenterPoint[1] = lastCenterPoint.y;
  }
  else if ((frameDisappear > 0) && (m_lastDetectBoundingBox.getCenterPoint().x != -1))
  {
    int lastPredHeight = m_lastDetectBoundingBox.getHeight();
    int lastPredWidth = m_lastDetectBoundingBox.getWidth();
    nextHeight = lastPredHeight;
    nextWidth = lastPredWidth;
  }

  //
  int x1 = nextCenterPoint[0] - (int)(nextWidth*0.5);
  int y1 = nextCenterPoint[1] - (int)(nextHeight*0.5);
  int x2 = nextCenterPoint[0] + (int)(nextWidth*0.5);
  int y2 = nextCenterPoint[1] + (int)(nextHeight*0.5);

  if (x1 < 0)
    x1 = 0;
  else if (x1 > imgW-1)
    x1 = imgW-1;

  if (y1 < 0)
    y1 = 0;
  else if (y1 > imgH-1)
    y1 = imgH - 1;

  if (x2 < 0)
    x2 = 0;
  else if (x2 > imgW-1)
    x2 = imgW - 1;

  if (y2 < 0)
    y2 = 0;
  else if (y2 > imgH-1)
    y2 = imgH - 1;

  // cout << "[Predict Next Bounding Box]" << endl;
  // cout << "=======================================" << endl;
  // cout << "obj.id = " << id << endl;
  // cout << "vel_x = " << velCenterPoint[0] << endl;
  // cout << "vel_y = " << velCenterPoint[1] << endl;
  // cout << "vel_a = " << velAspect << endl;
  // cout << "vel_h = " << velHeight << endl;
  // cout << "frame_disappear = " << frameDisappear << endl;
  // cout << "=======================================" << endl;

  BoundingBox newBox(x1, y1, x2, y2, currBox.label);
  newBox.frameStamp = currBox.frameStamp+1;


  if (frameDisappear > 0)
    bboxList.push_back(newBox);

  if (frameDisappear == frameInterval)
    m_lastPredBoundingBox = newBox;

  if (frameDisappear > 30)
    m_prevPredBoundingBox = BoundingBox(0, 0, 0, 0, -1);

  m_prevPredBoundingBox = newBox;
}


BoundingBox Object::getScaledBoundingBox(float r, int imgH, int imgW)
{
  int newHeight = (int)(bbox.getHeight()*(1+r));
  int newWidth = (int)(bbox.getWidth()*(1+r));
  Point centerPoint = bbox.getCenterPoint();

  int x1 = centerPoint.x - (int)(newWidth*0.5);
  int y1 = centerPoint.y - (int)(newHeight*0.5);
  int x2 = centerPoint.x + (int)(newWidth*0.5);
  int y2 = centerPoint.y + (int)(newHeight*0.5);

  if (x1 < 0)
    x1 = 0;
  else if (x1 > imgW-1)
    x1 = imgW-1;

  if (y1 < 0)
    y1 = 0;
  else if (y1 > imgH-1)
    y1 = imgH - 1;

  if (x2 < 0)
    x2 = 0;
  else if (x2 > imgW-1)
    x2 = imgW - 1;

  if (y2 < 0)
    y2 = 0;
  else if (y2 > imgH-1)
    y2 = imgH - 1;

  return BoundingBox(x1, y1, x2, y2, bbox.label);
}


void Object::getPrevPredBoundingBox(BoundingBox &prevPredBox)
{
  prevPredBox = m_prevPredBoundingBox;
}


vector<float> Object::_getCenterPointVelocity(int frameInterval)
{
  if (bboxList.size() == 0)
    return vector<float>(2);

  vector<BoundingBox> last_n_bboxList;
  if (bboxList.size() < frameInterval)
  {
    last_n_bboxList = bboxList;
  }
  else
  {
    for (int i=bboxList.size()-frameInterval; i<bboxList.size(); i++)
    {
      last_n_bboxList.push_back(bboxList[i]);
    }
  }

  vector<Point> centerPointList;
  for (int i=0; i<last_n_bboxList.size(); i++)
  {
    Point centerPoint = last_n_bboxList[i].getCenterPoint();
    centerPointList.push_back(centerPoint);
  }

  float dx = 0;
  float dy = 0;
  for (int i=1; i<centerPointList.size(); i++)
  {
    dx += (centerPointList[i].x-centerPointList[i-1].x);
    dy += (centerPointList[i].y-centerPointList[i-1].y);
  }

  vector<float> res = {
    dx / (float)frameInterval,
    dy / (float)frameInterval
  };

  return res;
}


float Object::_getAspectRatioVelocity(int frameInterval)
{
  if (bboxList.size() == 0)
    return 0.0;

  vector<BoundingBox> last_n_bboxList;
  if (bboxList.size() < frameInterval)
  {
    last_n_bboxList = bboxList;
  }
  else
  {
    for (int i=bboxList.size()-frameInterval; i<bboxList.size(); i++)
    {
      last_n_bboxList.push_back(bboxList[i]);
    }
  }


  vector<float> aspectRatioList;
  for (int i=0; i<last_n_bboxList.size(); i++)
  {
    aspectRatioList.push_back(last_n_bboxList[i].getAspectRatio());
  }

  float da = 0;
  for (int i=1; i<aspectRatioList.size(); i++)
  {
    da += (aspectRatioList[i]-aspectRatioList[i-1]);
  }

  return da / (float)frameInterval;
}


float Object::_getHeightVelocity(int frameInterval)
{
  if (bboxList.size() == 0)
    return 0.0;

  vector<BoundingBox> last_n_bboxList;
  if (bboxList.size() < frameInterval)
  {
    last_n_bboxList = bboxList;
  }
  else
  {
    for (int i=bboxList.size()-frameInterval; i<bboxList.size(); i++)
    {
      last_n_bboxList.push_back(bboxList[i]);
    }
  }


  vector<float> heightList;
  for (int i=0; i<last_n_bboxList.size(); i++)
  {
    heightList.push_back(last_n_bboxList[i].getHeight());
  }

  float dh = 0;
  for (int i=1; i<heightList.size(); i++)
  {
    dh += (heightList[i]-heightList[i-1]);
  }

  return dh / (float)frameInterval;
}


void Object::updateKeypoint(vector<cv::KeyPoint> &kpt)
{
  if (m_currKpts.size() != 0)
  {
    m_prevKpts.assign(m_currKpts.begin(), m_currKpts.end());
  }
  m_currKpts.assign(kpt.begin(), kpt.end());
}


void Object::updateDescriptor(cv::Mat &desc)
{
  if (!m_currDesc.empty())
  {
    m_prevDesc = m_currDesc.clone();
  }
  m_currDesc = desc;
}


void Object::updateImage(cv::Mat &img)
{
  if (!m_currImg.empty())
  {
    m_prevImg = m_currImg.clone();
  }

  // cv::resize(img, img, cv::Size(256, 256), cv::INTER_LINEAR);
  m_currImg = img;
}
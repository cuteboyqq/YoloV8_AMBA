#include "bounding_box.hpp"


/////////////////////////
// public member functions
////////////////////////
BoundingBox::BoundingBox(int _x1, int _y1, int _x2, int _y2, int _label)
{
  // === initialize parameters === //
  x1 = _x1;
  y1 = _y1;
  x2 = _x2;
  y2 = _y2;
  label = _label;

  if (debugMode)
  {
    std::cout << "[INFO] Create a \
      BBOX[" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << endl;
  }
};


BoundingBox::~BoundingBox()
{};


int BoundingBox::getHeight()
{
  h = y2-y1;
  return h;
}


int BoundingBox::getWidth()
{
  w = x2-x1;
  return w;
}


int BoundingBox::getArea()
{
  w = x2-x1;
  h = y2-y1;
  area = w*h;
  return area;
}


float BoundingBox::getAspectRatio()
{
  w = x2-x1;
  h = y2-y1;
  aspectRatio = (float)h/(float)w;
  return aspectRatio;
}


Point BoundingBox::getCenterPoint()
{
  int x = static_cast<int>((x1+x2) * 0.5);
  int y = static_cast<int>((y1+y2) * 0.5);
  pCenter = Point(x, y);
  return pCenter;
}


vector<Point> BoundingBox::getCornerPoint()
{
  vector<Point> cornerPoints;
  Point pTL = Point(x1, y1);
  Point pTR = Point(x2, y1);
  Point pBL = Point(x1, y2);
  Point pBR = Point(x2, y2);
  cornerPoints.push_back(pTL);
  cornerPoints.push_back(pTR);
  cornerPoints.push_back(pBL);
  cornerPoints.push_back(pBR);
  return cornerPoints;
}


void BoundingBox::setFrameStamp(int _frameStamp)
{
  frameStamp = _frameStamp;
}
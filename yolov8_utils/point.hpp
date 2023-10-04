#ifndef __POINT__
#define __POINT__

// #ifdef __cplusplus
// extern "C" {
// #endif

#include <iostream>

using namespace std;


class Point
{
 public:
  Point(int _x, int _y);
  ~Point();
  int x = -1;
  int y = -1;
  int behevior = 0;  // special use for human behavior
  int needWarn = 0;
  float visionDistance = 65535.0;
  float radarDistance = 65535.0;
  int objID = -1;

 private:

  // Debug
  int debugMode = false;
};

// #ifdef __cplusplus
// }
// #endif
#endif


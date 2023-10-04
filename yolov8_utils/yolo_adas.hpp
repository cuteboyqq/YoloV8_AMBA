#ifndef __YOLOADAS__
#define __YOLOADAS__

#include <chrono>
#include <iostream>
#include <string>

// SNPE SDK
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorMap.hpp"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// WNC
#include "point.hpp"
#include "object.hpp"
#include "bounding_box.hpp"
#include "yolo_adas_decoder.hpp"
#include "lane_line_calib.hpp"
#include "lane_line.hpp"
#include "img_util.hpp"
#include "utils.hpp"
#include "dla_config.hpp"
#include "logger.hpp"



using namespace std;

#define FILE_MODE 0
#define MAX_YOLO_BBX  100
// #define NUM_BBOX 5040     // NOTE: Change this according to image size: 192x320: 1260   288x512: 3024    384x640: 5040
// #define INPUT_WIDTH 640
// #define INPUT_HEIGHT 384
// #define SEG_WIDTH 80
// #define SEG_HEIGHT 48
// #define NUM_DET_CLASSES 6

#define NUM_BBOX 3780     // NOTE: Change this according to image size: 192x320: 1260   288x512: 3024    384x640: 5040
#define INPUT_WIDTH 576
#define INPUT_HEIGHT 320
#define SEG_WIDTH 72
#define SEG_HEIGHT 40
#define NUM_DET_CLASSES 6


enum LaneLabel
{
  BGR = 0,
  VERTICAL_DOUBLE_WHITE = 1,
  VERTICAL_SINGLE_WHITE = 2,
  VERTICAL_YELLOW = 3,
  HORIZONTAL_SINGLE_WHITE = 4,
  ROAD_CRUB = 5
};

enum DriveLabel
{
  DIRECT_AREA = 0,
  ALTERNATIVE_AREA = 1,
  BG = 2
};

enum DetectionLabel
{
  HUMAN = 0,
  SMALL_VEHICLE = 1,
  BIG_VEHICLE = 2,
  STOP_SIGN = 3,
  ROAD_SIGN = 4
};


// YOLO output
unsigned int YOLOADAS_Decode(
  const float *lane,
  const float *drive,
  const float *detection,
  const float conf_thr,
  const float iou_thr,
  struct v8xyxy out[]);


class YOLOADAS
{
 public:
  YOLOADAS(ADAS_Config_S *config);
  ~YOLOADAS();

  ///////////////////////////
  /// Member Functions
  //////////////////////////

  // Inference
  bool run();
  bool run(cv::Mat &imgFrame);
  void close();

  // I/O
  bool loadInput(std::string filePath);
  bool loadInput(cv::Mat &imgFrame);
  bool preProcessingFile(std::string imgPath);
  bool preProcessingMemory(cv::Mat &imgFrame);
  bool postProcessing();

  // Line
  bool getLineMask(cv::Mat &mask);
  bool getMainLineMask(cv::Mat &mask);
  bool getHorizontalLineMask(cv::Mat &mask);

  // Lane
  bool getLaneMask(cv::Mat &mask);
  bool getMainLaneMask(cv::Mat &mask);
  void getLaneLineInfo(LaneLineInfo &laneLineInfo);

  // Detection
  bool getHumanBoundingBox(
    vector<BoundingBox> &_outHumanBboxList,
    float confidenceHuman,
    int videoWidth,
    int videoHeight,
    BoundingBox &fcwROI);

  bool getRiderBoundingBox(
    vector<BoundingBox> &_outRiderBboxList,
    float confidenceRider,
    int videoWidth,
    int videoHeight,
    BoundingBox &fcwROI);

  bool getVehicleBoundingBox(
    vector<BoundingBox> &_outBboxList,
    float confidenceVehicle,
    int videoWidth,
    int videoHeight,
    BoundingBox &fcwROI);

  bool getRoadSignBoundingBox(
    vector<BoundingBox> &_outBboxList,
    float confidenceRoadSign,
    int videoWidth,
    int videoHeight,
    BoundingBox &fcwROI);

  bool getSignBoundingBox(
    vector<BoundingBox> &_outSignBboxList,
    float confidenceSign);

  // Others
  void genResultImage(
    cv::Mat &imgFrame,
    vector<BoundingBox> bboxList,
    int colorIdx
  );

  void genTrackObjectImage(
    cv::Mat &imgFrame,
    vector<Object> objectList
  );

  // Debug
  void getDebugLogs();
  void debugON();
  void showProcTime();

  ///////////////////////////
  /// Member Variables
  //////////////////////////
  //
  vector<vector<Point>> m_detectLeftLinePointList;
  vector<vector<Point>> m_detectRightLinePointList;
  vector<vector<Point>> m_detectHoritLinePointList;

  LaneLineCalib *m_laneLineCalib;

 private:
  ///////////////////////////
  /// Member Functions
  //////////////////////////

  // I/O
  bool _initModelIO();
  bool _loadImageFile(const std::string& inputFile);
  bool _imgPreprocessing();
  bool _getITensor(float* yoloOutput, const zdl::DlSystem::ITensor* tensor);
  bool _getOutputTensor();

  // Segmentation
  void _SEG_postProcessing();

  // Detection
  void _OD_postProcessing();
  float _getBboxOverlapRatio(
    BoundingBox &boxA, BoundingBox &boxB);

  void _rescaleBoundingBox(
    int bbx_num,
    struct v8xyxy *out,
    struct v8xyxy *scaledOut,
    int inputW,
    int inputH,
    int frameW,
    int frameH
  );

  void _bboxMerging(
    BoundingBox &bboxA, BoundingBox &bboxB, int label, BoundingBox &bboxMerge);

  // Lane & Line
  void _colorize(cv::Mat &inputLane, cv::Mat &inputLine, cv::Mat &outputLane, cv::Mat &outputLine);
  void _getMainLaneAndLine(cv::Mat &inputLane, cv::Mat &inputLine, cv::Mat &outLane, cv::Mat &outLine);
  void _calcLaneInfo(cv::Mat &inputLane, cv::Mat &inputLine);

  // Lane Calibration
  void _calibrateLaneMask();
  void _noiseRemoval(cv::Mat &laneMask);
  void _masksMerging(cv::Mat &laneMask);
  void _boundaryFineTuning(cv::Mat &laneMask);

  // yBottom
  void _updateYBottom(int yBottom);


  ///////////////////////////
  /// Member Variables
  //////////////////////////

  // Mat
  cv::Mat m_img;
  cv::Mat m_imgResized;

  // DLC
  std::unique_ptr<zdl::SNPE::SNPE> m_snpe = nullptr;

  // I/O information

  // Input
  int m_inputChannel = 0;
  int m_inputHeight = 0;
  int m_inputWidth = 0;
  int m_detectionSize = 0;

  int m_detectionBoxSize = 0;
  int m_detectionConfSize = 0;
  int m_detectionClassSize = 0;

  std::vector<float> m_inputBuff;
  zdl::DlSystem::TensorShape m_inputTensorShape;
  std::unique_ptr<zdl::DlSystem::ITensor> m_inputTensor;
  cv::Size inputSize;

  // Input (image enhancement)
  float m_brightness;
  int m_calcBrightnessCounter;

  // Output (Line)
  cv::Mat m_rawLine;
  cv::Mat m_lineMask;
  cv::Mat m_mainLineMask;
  cv::Mat m_horiLineMask;
  cv::Mat m_lineColor;

  // Output (Lane)
  cv::Mat m_rawLane;
  cv::Mat m_laneMask;
  cv::Mat m_mainLaneMask;
  cv::Mat m_laneColor;

  // Output (Yolo Decoder)
  YOLOADAS_Decoder *m_decoder;

  float* m_laneBuff;
  float* m_lineBuff;
  float* m_detectionBoxBuff;
  float* m_detectionConfBuff;
  float* m_detectionClsBuff;

  std::vector<std::string> m_outputTensorList = {
    "lane_output",
    "drive_output",
    "det_box",
    "det_conf",
    "det_cls"};

  zdl::DlSystem::TensorMap m_outputTensorMap;

  // Output (Lane Line Calibration)
  vector<cv::Point> m_currLeftPointList;
  vector<cv::Point> m_currRightPointList;
  vector<vector<Point>> m_currLeftPointLists;
  vector<vector<Point>> m_currRightPointLists;
  int m_currLeftX = 0;
  int m_currRightX = 0;
  int m_prevLeftX = 0;
  int m_prevRightX = 0;
  LaneLineInfo m_laneLineInfo;

  // Lane Mask Calibration
  int m_maxLaneWidth = 0;
  int m_yHead = 0;
  int m_yBottom = 0;
  int m_maxLaneMaskListSize = 4;
  int m_prevLaneArea = 0;
  cv::Mat m_prevLaneMask;
  vector<vector<Point>> m_midLinePointLists;
  vector<Point> m_maxWidthPointList;
  vector<cv::Mat> m_prevLaneMaskList;

  // Line Mask Calibration
  cv::Point m_pLeftCarhood;
  cv::Point m_pRightCarhood;

  // Horizontal Line
  int m_horiLineArea = 0;

  // Inference
  bool m_inference = true;

  // yBottom
  vector<int> m_yBottomList;
  int m_yBottomListSize = 10;

  // Bounding Box
  float m_bboxExpandRatio = 1.0;
  struct v8xyxy m_yoloOut[MAX_YOLO_BBX];
  struct v8xyxy m_scaledOut[MAX_YOLO_BBX];
  int m_numBox = 0;

  // Threshold
  float confidenceThreshold = 0.5;
  float iouThreshold = 0.5;

  // Bird Eye View (for future work)
  bool m_enableBEV = false;
  cv::Mat m_warpM;
  cv::Mat m_warpMInv;


  // debug
  bool m_debugMode = false;
  bool m_estimateTime = false;
};

#endif
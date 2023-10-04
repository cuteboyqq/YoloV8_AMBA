/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#include "yolo_adas.hpp"


// Object color
const std::vector<cv::Scalar> object_colors = {
  {255, 51, 123}, // Person (Raw)
  {0, 0, 255},    // Person
  {0, 255, 0},    // Vehicle (Tracked)
  {0, 255, 128},  // Vehicle (Raw)
  {255, 153, 51}, // Bike, Motocycle (Raw)
  {255, 0, 0},    // Bike, Motocycle
  {125, 84, 225}, // Traffic lights and signs
};

const std::vector<cv::Scalar> lane_colors = {
  {255, 0, 0},   // Direct Area = Blue
  {0, 255, 0},   // Alternative Area = Green
  {0, 0, 0}      // Background = Black
};

// YOLO-ADAS v0.3.7
// const std::vector<cv::Scalar> line_colors = {
//   {0, 0, 0},      // Background = Black
//   {255, 255, 0},  // Vertical double white = Aqua
//   {116, 139, 69}, // Vertical single white = Aquamarine 4
//   {87, 207, 227}, // Vertical Yellow = Banana
//   {148, 0, 211},  // Horizontal Single White = Dark Violet
//   {35, 35, 139}   // Road Curb = Brown
// };

// YOLO-ADAS v0.4.6
const std::vector<cv::Scalar> line_colors = {
  {0, 0, 0},      // Background = Black
  {255, 255, 0},  // Vertical double white = Aqua
  {148, 0, 211},  // Horizontal Single White = Dark Violet
  {87, 207, 227}, // Vertical Yellow = Banana
  {35, 35, 139}   // Road Curb = Brown
};


/////////////////////////
// public member functions
////////////////////////
YOLOADAS::YOLOADAS(ADAS_Config_S *config)  // main function
{
  auto m_logger = spdlog::stdout_color_mt("YOLO-ADAS");
  m_logger->set_pattern("[%n] [%^%l%$] %v");

  if (config->stDebugConfig.yoloADAS)
  {
    m_logger->set_level(spdlog::level::debug);
  }
  else
  {
    m_logger->set_level(spdlog::level::info);
  }

  std::string dlcFilePath = config->modelPath;
  std::string rumtimeStr = config->runtime;

  // Runtime
  static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
  static zdl::DlSystem::RuntimeList runtimeList;
  bool usingInitCaching = false;
  bool staticQuantization = false;
  bool useUserSuppliedBuffers = false;

  m_logger->info("Creating YOLO-ADAS Model ...");

  // Check if both runtimelist and runtime are passed in
  if (rumtimeStr == "gpu")
  {
    runtime = zdl::DlSystem::Runtime_t::GPU;
  }
  else if (rumtimeStr == "aip")
  {
    runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
  }
  else if (rumtimeStr == "dsp")
  {
    runtime = zdl::DlSystem::Runtime_t::DSP;
  }
  else if (rumtimeStr == "cpu")
  {
    runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  else
  {
    m_logger->warn("The runtime option provide is not valid. Defaulting to the CPU runtime.");
  }

  if(runtimeList.empty() == false)
  {
    m_logger->error("Invalid option cannot mix runtime order -l with runtime -r ");
    std::exit(1);
  }

  // STEP1: Get Available Runtime
  m_logger->info("Rumtime = {}", rumtimeStr);
  runtime = checkRuntime(runtime, staticQuantization);
  runtimeList.add(runtime);

  // STEP2: Create Deep Learning Container and Load Network File
  m_logger->info("DLC File Path = {}",  dlcFilePath);
  std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlcFilePath);
  if (container == nullptr)
  {
    m_logger->error("Error while opening the container file.");
    std::exit(1);
  }

  // STEP3: Set Network Builder
  zdl::DlSystem::PlatformConfig platformConfig;
  m_snpe = setBuilderOptions(
    container, runtimeList, useUserSuppliedBuffers, m_outputTensorList, platformConfig, usingInitCaching);

  if (m_snpe == nullptr)
  {
    m_logger->error("Error while building SNPE object.");
    std::exit(1);
  }
  if (usingInitCaching)
  {
    if (container->save(dlcFilePath))
    {
      m_logger->info("Saved container into archive successfully");
    }
    else
    {
      m_logger->warn("Failed to save container into archive");
    }
  }

  // STEP4: Init Model Input/Output Tensor
  _initModelIO();

  // STEP5: Parameters Initialization
  m_prevLeftX = 0;
  m_prevRightX = 0;

  // For brighteness calibration
  m_calcBrightnessCounter = 0;

  // Lane Line Calibration
  m_laneLineCalib = new LaneLineCalib(config); //TODO:

  // Output Decoder
  m_decoder = new YOLOADAS_Decoder(m_inputHeight, m_inputWidth);

  // Bird Eye View (Optional)
  if (m_enableBEV)
  {
    vector<cv::Point2f> src;
    vector<cv::Point2f> dst;

    int x1_top = 232;
    int x1_bot = 153;
    int x2_top = 280;
    int x2_bot = 359;

    int y_top = m_inputHeight * 0.68;
    int y_bot = m_inputHeight * 0.83; //  mainLaneInfo.pLeftCarhood.y;

    cv::Point2f pA(x1_bot, y_bot);
    cv::Point2f pB(x2_bot, y_bot);
    cv::Point2f pC(x1_top, y_top);
    cv::Point2f pD(x2_top, y_top);

    cv::Point2f pA_(x1_bot, m_inputHeight);
    cv::Point2f pB_(x2_bot, m_inputHeight);
    cv::Point2f pC_(x1_bot, 0);
    cv::Point2f pD_(x2_bot, 0);

    //
    src.push_back(pA);
    src.push_back(pB);
    src.push_back(pC);
    src.push_back(pD);

    //
    dst.push_back(cv::Point2f(pA));
    dst.push_back(cv::Point2f(pB));
    dst.push_back(cv::Point2f(pC_));
    dst.push_back(cv::Point2f(pD_));

    // Step 5: Get the perspective transformation matrix and its inverse
    m_warpM = cv::getPerspectiveTransform(src, dst);
    m_warpMInv = cv::getPerspectiveTransform(dst, src);
  }
};

YOLOADAS::~YOLOADAS()  // clear object memory
{
  delete m_decoder;
  delete m_laneBuff;
  delete m_lineBuff;
  delete m_detectionBoxBuff;
  delete m_detectionConfBuff;
  delete m_detectionClsBuff;
  delete m_laneLineCalib;

  m_decoder = nullptr;
  m_laneBuff = nullptr;
  m_lineBuff = nullptr;
  m_detectionBoxBuff = nullptr;
  m_detectionConfBuff = nullptr;
  m_detectionClsBuff = nullptr;
  m_laneLineCalib = nullptr;
};

void YOLOADAS::close()
{
  m_snpe.reset();
}


// ============================================
//               Tensor Settings
// ============================================
bool YOLOADAS::_initModelIO()
{
  auto m_logger = spdlog::get("YOLO-ADAS");
  m_logger->info("[YOLO-ADAS] => Create Model Input Tensor");
  m_logger->info("-------------------------------------------");
  m_inputTensorShape = m_snpe->getInputDimensions();
  m_inputHeight = m_inputTensorShape.getDimensions()[1];
  m_inputWidth = m_inputTensorShape.getDimensions()[2];
  m_inputChannel = m_inputTensorShape.getDimensions()[3];
  m_logger->info("Input H: {}", m_inputHeight);
  m_logger->info("Input W: {}", m_inputWidth);
  m_logger->info("Input C: {}", m_inputChannel);

  // Get input names and number
  const auto& inputTensorNamesRef = m_snpe->getInputTensorNames();
  if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
  const zdl::DlSystem::StringList& inputTensorNames = *inputTensorNamesRef;  // inputTensorNames refers to m_snpe->getInputTensorNames()'s returned variable

  // Make sure the network requires only a single input
  assert (inputTensorNames.size() == 1);

  /* Create an input tensor that is correctly sized to hold the input of the network.
    Dimensions that have no fixed size will be represented with a value of 0. */
  const auto &inputDims_opt = m_snpe->getInputDimensions(inputTensorNames.at(0));
  const auto &inputShape = *inputDims_opt;  // 384 * 640

  /* Calculate the total number of elements that can be stored in the tensor
    so that we can check that the input contains the expected number of elements.
    With the input dimensions computed create a tensor to convey the input into the network. */
  m_inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

  // Create a buffer to store image data
  m_inputBuff.resize(m_inputChannel*m_inputHeight*m_inputWidth);

  m_logger->info("Create Model Output Buffers");
  m_logger->info("-------------------------------------------");

  m_detectionBoxSize = 5 * NUM_BBOX;
  m_detectionConfSize = NUM_BBOX;
  m_detectionClassSize = NUM_BBOX;
  m_detectionBoxBuff = new float[m_detectionBoxSize];
  m_detectionConfBuff = new float[m_detectionConfSize];
  m_detectionClsBuff = new float[m_detectionClassSize];

  m_laneBuff = new float [SEG_WIDTH*SEG_HEIGHT];
  m_lineBuff = new float [SEG_WIDTH*SEG_HEIGHT];

  return true;
}


bool YOLOADAS::_getITensor(float *yoloOutputBuff, const zdl::DlSystem::ITensor* tensor)
{
  int batchChunk = tensor->getSize();

  std::memcpy(
    yoloOutputBuff,
    &tensor->cbegin()[0],
    batchChunk * sizeof(float));

  return true;
}


bool YOLOADAS::_getOutputTensor()
{
  auto m_logger = spdlog::get("YOLO-ADAS");
  auto time_0 = std::chrono::high_resolution_clock::now();

  auto lane_tensorPtr = m_outputTensorMap.getTensor(m_outputTensorList[0].c_str());
  auto drive_tensorPtr = m_outputTensorMap.getTensor(m_outputTensorList[1].c_str());
  // auto detection_tensorPtr = m_outputTensorMap.getTensor(m_outputTensorList[2].c_str());
  auto detectionBox_tensorPtr = m_outputTensorMap.getTensor(m_outputTensorList[2].c_str());
  auto detectionConf_tensorPtr = m_outputTensorMap.getTensor(m_outputTensorList[3].c_str());
  auto detectionCls_tensorPtr = m_outputTensorMap.getTensor(m_outputTensorList[4].c_str());

  if(!_getITensor(m_lineBuff, lane_tensorPtr))
  {
    m_logger->error("Failed to get lane line tensor");
    return false;
  }

  if(!_getITensor(m_laneBuff, drive_tensorPtr))
  {
    m_logger->error("Failed to get drivable area tensor");
    return false;
  }

  if(!_getITensor(m_detectionBoxBuff, detectionBox_tensorPtr))
  {
    m_logger->error("Failed to get detection box tensor");
    return false;
  }

  if(!_getITensor(m_detectionConfBuff, detectionConf_tensorPtr))
  {
    m_logger->error("Failed to get detection conf tensor");
    return false;
  }

  if(!_getITensor(m_detectionClsBuff, detectionCls_tensorPtr))
  {
    m_logger->error("Failed to get detection cls tensor");
    return false;
  }

  auto time_1 = std::chrono::high_resolution_clock::now();
  m_logger->debug("[Get Output]: \t{} ms", \
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
  return true;
}


// ============================================
//            Inference Entrypoint
// ============================================

bool YOLOADAS::run()
{
  auto m_logger = spdlog::get("YOLO-ADAS");
  auto time_0 = std::chrono::high_resolution_clock::now();
  m_inference = m_snpe->execute(m_inputTensor.get(), m_outputTensorMap);
  auto time_1 = std::chrono::high_resolution_clock::now();
  m_logger->debug("[Inference]: \t{} ms",\
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  //
  postProcessing();
  return true;
}


bool YOLOADAS::run(cv::Mat &imgFrame)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  if (m_estimateTime)
  {
    m_logger->debug("\n[YOLO-ADAS Processing Time]");
    m_logger->debug("-----------------------------------------");
  }
  // STEP1: load image to input tensor
  if (!loadInput(imgFrame))
  {
    m_logger->error("Load Input Data Failed");

    return false;
  }

  // STEP2: run inference
  auto time_0 = std::chrono::high_resolution_clock::now();
  m_inference = m_snpe->execute(m_inputTensor.get(), m_outputTensorMap);

  if (!m_inference)
  {
    m_logger->error("AI Inference Failed");

    return false;
  }

  auto time_1 = std::chrono::high_resolution_clock::now();
  m_logger->debug("[Inference]: \t{} ms", \
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  // STEP3: post processing
  if (!postProcessing())
  {
    m_logger->error("AI Post Processing Failed");

    return false;
  }

  if (m_estimateTime)
  {
    m_logger->debug("-----------------------------------------");
  }

  return true;
}

// ============================================
//                Load Inputs
// ============================================
bool YOLOADAS::loadInput(std::string filePath)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  m_logger->debug("\n===================================");
  m_logger->debug("[File] => {}", filePath);
  m_logger->debug("===================================");

  // Preprocessing
  preProcessingFile(filePath);

  auto time_0 = std::chrono::high_resolution_clock::now();

  if (m_inputTensor->getSize() != m_inputBuff.size())
  {
    m_logger->error("Size of input does not match network.");
    m_logger->error("Expecting: {}", m_inputTensor->getSize());
    m_logger->error("Got: {}", m_inputBuff.size());

    return false;
  }

  /* Copy the loaded input file contents into the networks input tensor.
    SNPE's ITensor supports C++ STL functions like std::copy() */
  std::copy(m_inputBuff.begin(), m_inputBuff.end(), m_inputTensor->begin());

  auto time_1 = std::chrono::high_resolution_clock::now();

  m_logger->debug("[Load Input]: \t{} ms", \
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  return true;
}


bool YOLOADAS::loadInput(cv::Mat &imgFrame)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  // Preprocessing
  if (!preProcessingMemory(imgFrame))
  {
    m_logger->error("Data Preprocessing Failed");

    return false;
  }

  auto time_0 = std::chrono::high_resolution_clock::now();

  if (m_inputTensor->getSize() != m_inputBuff.size())
  {
    m_logger->error("Size of input does not match network.");
    m_logger->error("Expecting: {}", m_inputTensor->getSize());
    m_logger->error("Got: {}", m_inputBuff.size());

    return false;
  }

  /* Copy the loaded input file contents into the networks input tensor.
    SNPE's ITensor supports C++ STL functions like std::copy() */
  std::copy(m_inputBuff.begin(), m_inputBuff.end(), m_inputTensor->begin());

  auto time_1 = std::chrono::high_resolution_clock::now();

  m_logger->debug("[Load Input]: \t{} ms", \
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  return true;
}


bool YOLOADAS::_loadImageFile(const std::string& inputFile)
{
  auto m_logger = spdlog::get("YOLO-ADAS");
  auto time_0 = std::chrono::high_resolution_clock::now();

  m_img = cv::imread(inputFile, -1);
  if (m_img.empty())
  {
    m_logger->error("image don't exist!");
    return false;
  }

  auto time_1 = std::chrono::high_resolution_clock::now();
  m_logger->debug("[Read Image]: \t{} ms", \
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  return true;
}

// ============================================
//               Pre Processing
// ============================================
bool YOLOADAS::preProcessingFile(std::string imgPath)
{
  _loadImageFile(imgPath);
  _imgPreprocessing();
  return true;
}


bool YOLOADAS::preProcessingMemory(cv::Mat &imgFrame)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  if (imgFrame.empty())
  {
    m_logger->error("Image don't exists!");
    return false;
  }
  else
  {
    m_img = imgFrame;
  }
  _imgPreprocessing();
  return true;
}


bool YOLOADAS::_imgPreprocessing()
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  int imageSize = m_inputChannel*m_inputHeight*m_inputWidth;
  cv::Size inputSize = cv::Size(m_inputWidth, m_inputHeight);
  cv::Mat imgResized;
  cv::Mat sample;
  cv::Mat sampleNorm;

  auto time_0 = std::chrono::high_resolution_clock::now();

  if (m_img.size() != inputSize)
  {
    cv::resize(m_img, imgResized, inputSize, cv::INTER_LINEAR);
  }
  else
  {
    imgResized = m_img;
  }
  m_imgResized = imgResized;

  // Calc brightness
  if (m_calcBrightnessCounter == 0)
  {
    m_brightness = imgUtil::calcBrightnessRatio(imgResized);
    m_calcBrightnessCounter += 1;
  }
  else if (m_calcBrightnessCounter >= 1)
  {
    m_calcBrightnessCounter = 0;
  }
  else
  {
    m_calcBrightnessCounter += 1;
  }

  // Image Enhancement
  imgUtil::brightnessEnhancement(m_brightness, m_imgResized);

  // BGR to RGB
  cv::cvtColor(imgResized, sample, cv::COLOR_BGR2RGB);

  // Normalize
  sample.convertTo(sampleNorm, CV_32F, 1.0 / 255, 0);

  std::memcpy(&m_inputBuff[0], sampleNorm.data, imageSize*sizeof(float));

  auto time_1 = std::chrono::high_resolution_clock::now();
  m_logger->debug("[Pre-Proc]: \t{}",\
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  return true;
}

// ============================================
//               Post Processing
// ============================================
bool YOLOADAS::postProcessing()
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  if (m_inference == true)
  {
    if(!_getOutputTensor())
    {
      m_logger->error("Unable to get output tensors!");
      return false;
    }
  }
  else
  {
    m_logger->error("Error while executing the network.");
    return false;
  }

  // STEP1: Semantic Segmentation
  _SEG_postProcessing();

  // STEP2: Object Detection
  _OD_postProcessing();

  return true;
}


void YOLOADAS::_SEG_postProcessing()
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  auto time_0 = std::chrono::high_resolution_clock::now();
  auto time_1 = std::chrono::high_resolution_clock::now();

  m_logger->debug("Starting object detection post-processing......");

  m_rawLine = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_32F, cv::Scalar::all(0));
  m_rawLane = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_32F, cv::Scalar::all(0));
  m_laneMask = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC1, cv::Scalar::all(0));
  m_lineMask = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC1, cv::Scalar::all(0));
  m_mainLaneMask = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC1, cv::Scalar::all(0));
  m_mainLineMask = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC1, cv::Scalar::all(0));
  m_horiLineMask = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC1, cv::Scalar::all(0));

  if (m_debugMode)
  {
    m_laneColor = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC3, cv::Scalar::all(0));
    m_lineColor = cv::Mat(cv::Size(SEG_WIDTH, SEG_HEIGHT), CV_8UC3, cv::Scalar::all(0));
  }

  if (!(m_laneBuff && m_lineBuff && m_detectionBoxBuff && m_detectionClsBuff && m_yoloOut))  // Missing output(s)
  {
    m_logger->error("Not all outputs of the network are available");
  }
  // Extract masks
  std::memcpy(m_rawLine.data, m_lineBuff, m_rawLine.rows*m_rawLine.cols*sizeof(float));
  std::memcpy(m_rawLane.data, m_laneBuff, m_rawLine.rows*m_rawLine.cols*sizeof(float));

  m_rawLine.convertTo(m_lineMask, CV_8UC1);
  m_rawLane.convertTo(m_laneMask, CV_8UC1);

  if (m_debugMode)
  {
    // Colorize segmentation outputs
    _colorize(m_laneMask, m_lineMask, m_laneColor, m_lineColor);
  }

  // Get maie lane and line
  _getMainLaneAndLine(m_laneMask, m_lineMask, m_mainLaneMask, m_mainLineMask);

  // Lane calibration
  _calibrateLaneMask();

  // Calculate lane information
  _calcLaneInfo(m_mainLaneMask, m_mainLineMask);

  if (m_estimateTime)
  {
    time_1 = std::chrono::high_resolution_clock::now();
    m_logger->debug("[Lane Calibration]: \t{}", \
      std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
  }

  // Do Line Calibration
  if (m_prevLaneMaskList.size() > 1)
  {
    m_laneLineCalib->run(
      m_mainLineMask, m_horiLineMask,
      m_midLinePointLists, m_yHead, m_yBottom, m_maxLaneWidth);
    // For fine tuning lane boundary
    m_laneLineCalib->getResult(
      m_currLeftX, m_currRightX,
      m_currLeftPointList, m_currRightPointList);

    // For line detection
    m_laneLineCalib->getCarhoodPoints(m_pLeftCarhood, m_pRightCarhood);
    m_laneLineCalib->getHorizontalLine(m_horiLineMask);
    m_laneLineCalib->getHorizontalLineArea(m_horiLineArea);

    // Get Calibration Information
    m_laneLineCalib->getLineCalibInfo(m_laneLineInfo.lineCalibInfo);
  }

  m_logger->debug("Finished semantic segmentation post-processing");

  if (m_estimateTime)
  {
    time_1 = std::chrono::high_resolution_clock::now();
    m_logger->debug("[_SEG_postProcessing]: \t{}", \
      std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
  }
}


void YOLOADAS::_OD_postProcessing()
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  auto time_0 = std::chrono::high_resolution_clock::now();
  auto time_1 = std::chrono::high_resolution_clock::now();

  // // Check
  // if (!(m_laneBuff && m_lineBuff && m_detectionBuff && m_yoloOut))  // Missing output(s)
  //   cerr << "Not all outputs of the network are available" << endl;

  if (!(m_laneBuff && m_lineBuff && m_detectionBoxBuff && m_detectionClsBuff && m_yoloOut))  // Missing output(s)
  {
    m_logger->error("Not all outputs of the network are available");
  }

  m_logger->debug("Starting object detection post-processing......");

  // m_numBox = m_decoder->decode((float *)m_detectionBuff , confidenceThreshold, iouThreshold, m_yoloOut);
  m_numBox = m_decoder->decode((
    float *)m_detectionBoxBuff, (float *)m_detectionConfBuff, (float *)m_detectionClsBuff, confidenceThreshold, iouThreshold, m_yoloOut);

  // _rescaleBoundingBox(
  //   m_numBox, m_yoloOut, m_scaledOut, m_inputWidth, m_inputHeight, m_img.cols, m_img.rows);

  m_logger->debug("[Post-Proc]: \t{}", \
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));

  m_logger->debug("=> GET # of raw BBOX(es): {}", m_numBox);
  for(int i=0; i< m_numBox; i++)
  {
    struct v8xyxy b = m_yoloOut[i];
    m_logger->debug("=> bbx {}: ({},{})-({},{}), c={}, conf={}", i, b.x1, b.y1, b.x2, b.y2, b.c, b.c_prob);
  }

  m_logger->debug("Finished object detection post-processing");

  if (m_estimateTime)
  {
    time_1 = std::chrono::high_resolution_clock::now();
    m_logger->debug("[_OD_postProcessing]: \t{} ms", \
      std::chrono::duration_cast<std::chrono::nanoseconds>(time_1 - time_0).count() / (1000.0 * 1000));
  }
}


void YOLOADAS::_rescaleBoundingBox(
    int bbx_num,
    struct v8xyxy *out,
    struct v8xyxy *scaledOut,
    int inputW, int inputH,
    int frameW, int frameH)
{
  float w_ratio = (float)inputW / (float)frameW;
  float h_ratio = (float)inputH / (float)frameH;

  for(int i=0; i<bbx_num; i++)
  {
    scaledOut[i].c = out[i].c;
    scaledOut[i].c_prob = out[i].c_prob;
    scaledOut[i].x1 = (int)((float)out[i].x1 / w_ratio);
    scaledOut[i].y1 = (int)((float)out[i].y1 / h_ratio);
    scaledOut[i].x2 = (int)((float)out[i].x2 / w_ratio);
    scaledOut[i].y2 = (int)((float)out[i].y2 / h_ratio);

    // Expand Bounding Box
    int w = scaledOut[i].x2 - scaledOut[i].x1;
    int h = scaledOut[i].y2 - scaledOut[i].y1;

    int c_x = scaledOut[i].x1 + (int)((float)w / 2.0);
    int c_y = scaledOut[i].y1 + (int)((float)h / 2.0);
    w = w * (m_bboxExpandRatio+0.15);
    h = h * m_bboxExpandRatio;

    scaledOut[i].x1 = c_x - (int)((float)w / 2.0);
    scaledOut[i].y1 = c_y - (int)((float)h / 2.0);
    scaledOut[i].x2 = c_x + (int)((float)w / 2.0);
    scaledOut[i].y2 = c_y + (int)((float)h / 2.0);

    if (scaledOut[i].x1 < 0)
      scaledOut[i].x1 = 0;
    if (scaledOut[i].x2 > frameW-1)
      scaledOut[i].x2 = frameW-1;
    if (scaledOut[i].y1 < 0)
      scaledOut[i].y1 = 0;
    if (scaledOut[i].y2 > frameH-1)
      scaledOut[i].y2 = frameH-1;
  }
}


float YOLOADAS::_getBboxOverlapRatio(BoundingBox &boxA, BoundingBox &boxB)
{
  int iouX = max(boxA.x1, boxB.x1);
  int iouY = max(boxA.y1, boxB.y1);
  int iouW = min(boxA.x2, boxB.x2) - iouX;
  int iouH = min(boxA.y2, boxB.y2) - iouY;
  iouW = max(iouW, 0);
  iouH = max(iouH, 0);

  if (boxA.getArea() == 0)
    return 0;

  float iouArea = iouW * iouH;
  float ratio = iouArea / (float)boxA.getArea();

  return ratio;
}


void YOLOADAS::_bboxMerging(BoundingBox &bboxA, BoundingBox &bboxB, int label, BoundingBox &bboxMerge)
{
  int newX1 = 0;
  int newY1 = 0;
  int newX2 = 0;
  int newY2 = 0;

  vector<BoundingBox> tmpBboxList;
  tmpBboxList.push_back(bboxA);
  tmpBboxList.push_back(bboxB);

  vector<vector<Point>> cornerPointList;
  for (int i=0; i<(int)tmpBboxList.size(); i++)
  {
    cornerPointList.push_back(tmpBboxList[i].getCornerPoint());
  }

  // 0:TL, 1:TR, 2:BL, 3:BR
  vector<int> x1List;
  vector<int> y1List;
  vector<int> x2List;
  vector<int> y2List;
  for (int i=0; i<(int)cornerPointList.size(); i++)
  {
    x1List.push_back(cornerPointList[i][0].x);
    y1List.push_back(cornerPointList[i][0].y);
    x2List.push_back(cornerPointList[i][3].x);
    y2List.push_back(cornerPointList[i][3].y);
  }

  vector<int>::iterator x1It;
  vector<int>::iterator x2It;
  vector<int>::iterator y1It;
  vector<int>::iterator y2It;

  x1It = std::min_element(x1List.begin(), x1List.end());
  x2It = std::max_element(x2List.begin(), x2List.end());

  y1It = std::min_element(y1List.begin(), y1List.end());
  y2It = std::min_element(y2List.begin(), y2List.end());

  // newX1 = int(((*x1It_min) + (*x1It_max)) * 0.5);
  // newX2 = int(((*x2It_min) + (*x2It_max)) * 0.5);
  newX1 = (*x1It);
  newX2 = (*x2It);

  newY1 = (*y1It);
  newY2 = (*y2It);

  bboxMerge.x1 = newX1;
  bboxMerge.y1 = newY1;
  bboxMerge.x2 = newX2;
  bboxMerge.y2 = newY2;
  bboxMerge.label = label;
}


void YOLOADAS::_getMainLaneAndLine(
  cv::Mat &inputLane, cv::Mat &inputLine,
  cv::Mat &outputLane, cv::Mat &outputLine)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  int rows = inputLane.size[0];
  int columns = inputLane.size[1];
  int yellowCount = 0;

  for (int i = 0; i < rows; i++)
  {
    for (int j = 1; j < columns-1; j++)
    {
      int laneCls = inputLane.at<uchar>(i, j);
      if (laneCls == 0) // Direct area
      {
        outputLane.at<uchar>(i, j) = 255;
        outputLine.at<uchar>(i, j) = 0;
      }
      else if (laneCls == 1) // Side area
      {
        outputLine.at<uchar>(i, j) = 0;
      }

      // YOLO-ADAS v0.3.7
      // 0: Background = Black
      // 1: Vertical double white = Aqua
      // 2: Vertical single white = Aquamarine 4
      // 3: Vertical Yellow = Banana
      // 4: Horizontal Single White = Dark Violet
      // 5: Road Curb = Brown

      // YOLO-ADAS v0.4.6
      // 0: Background = Black
      // 1: White Line = Aqua
      // 2: Cross Walk = Aquamarine 4
      // 3: Yellow Line = Banana
      // 4: Road Curb = Brown
      int prevLineCls = inputLine.at<uchar>(i-1, j);
      int lineCls = inputLine.at<uchar>(i, j);
      int nextLineCls = inputLine.at<uchar>(i+1, j);

      // if ((lineCls == 1 || lineCls == 2 || lineCls == 3 || lineCls == 5)) // YOLO-ADAS v0.3.7
      if ((lineCls == 1 || lineCls == 3 || lineCls == 4)) // YOLO-ADAS v0.4.6
      {
        /*
        if (lineCls == 5 && (prevLineCls == 2 || prevLineCls == 3))
        {
          for (int k=0; k<5; k++)
            outputLine.at<uchar>(i, j-k) = 0;
        }
        else if (lineCls == 5 && (nextLineCls == 2 || nextLineCls == 3))
        {
          outputLine.at<uchar>(i, j) = 0;
        }
        else
        {
          outputLine.at<uchar>(i, j) = 255;
        }
        */
        outputLine.at<uchar>(i, j) = 255;

        if (lineCls == 3)
        {
          yellowCount += 1;
        }
      }

      // if (lineCls == 4) // YOLO-ADAS v0.3.7

      if (lineCls == 2) // YOLO-ADAS v0.4.6
      {
        m_horiLineMask.at<uchar>(i, j) = 255;
      }
    }
  }
  float yDiff = (float)(m_yBottom - m_yHead);
  m_logger->debug("m_yBottom = {}", m_yBottom);
  m_logger->debug("m_yHead = {}", m_yHead);
  m_logger->debug("yDiff = {}", yDiff);

  // Save information
  m_laneLineInfo.laneMaskInfo.yLaneHead = m_yHead;
  m_laneLineInfo.laneMaskInfo.yLaneBottom = m_yBottom;

  // Calculate yellow line ratio
  float yellowRatio = (float)yellowCount / ((float)SEG_WIDTH * yDiff);

  if (yDiff == 0)
  {
    yellowRatio = 0;
  }

  // TODO: coulde optimize
  // yellow gird cases
  if (yellowRatio > 0.12)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < columns; j++)
      {
        int lineCls = inputLine.at<uchar>(i, j);

        if (lineCls == 3)
        {
          m_horiLineMask.at<uchar>(i, j) = 255;
        }
      }
    }
  }

  // Save information
  m_laneLineInfo.lineMaskInfo.yellowLineArea = yellowCount;
  m_laneLineInfo.lineMaskInfo.yellowLineAreaRatio = yellowRatio;

  m_logger->debug("yellowRatio = {}", yellowRatio);
}


void YOLOADAS::_updateYBottom(int yBottom)
{
  float ratio = (float)abs(m_yBottom - yBottom) / (float)SEG_HEIGHT;
  if (ratio < 0.1 && m_yBottom != 0)
  {
    utils::updateIntList(m_yBottomList, yBottom, m_yBottomListSize);
  }
  else if (m_yBottom == 0)
  {
    utils::updateIntList(m_yBottomList, yBottom, m_yBottomListSize);
  }

  if (m_yBottomList.size() >= m_yBottomListSize)
  {
    m_yBottom = utils::findMedian(m_yBottomList, m_yBottomListSize);
  }
  else
  {
    m_yBottom = yBottom;
  }
}


void YOLOADAS::_calcLaneInfo(cv::Mat &inputLane, cv::Mat &inputLine)
{
  int rows = inputLane.size[0];
  int columns = inputLane.size[1];
  m_midLinePointLists.clear();
  m_maxWidthPointList.clear();

  int yHead = rows;
  int yBottom = 0;
  int xStart = 0;
  int xEnd = columns;
  int prevWidth = 0;

  // Find the head and bottom of the lane
  for (int y=0; y<rows; y++)
  {
    vector<Point> lanePointList;
    for (int x=0; x<columns; x++)
    {
      if (inputLane.at<uchar>(y, x) == 255)
      {
        if (y < yHead)
        {
          yHead = y;
          if (lanePointList.size() > 1)
          {
            xStart = lanePointList.front().x;
            xEnd = lanePointList.back().x;
          }
        }
        lanePointList.push_back(Point(x, y));
      }
    }

    vector<Point> tmpPointList;
    if (lanePointList.size() > 0)
    {
      int midX = (int)((lanePointList.front().x + lanePointList.back().x) / 2);
      tmpPointList.push_back(Point(midX, y));

      int width = abs(lanePointList.front().x - lanePointList.back().x);
      if ((width >= prevWidth))
      {
        prevWidth = width;
        m_maxWidthPointList = lanePointList;
        m_maxLaneWidth = width;

        // Save Information
        m_laneLineInfo.laneMaskInfo.width = width;

        if (y > yBottom)
        {
          yBottom = y;
        }
      }
    }
    m_midLinePointLists.push_back(tmpPointList);
  }

  // float ratio = (float)abs(m_yBottom - yBottom) / (float)rows;
  // if (ratio < 0.08 && m_yBottom != 0)
  // {
  //   m_yBottom = yBottom;
  // }
  // else if (m_yBottom == 0)
  // {
  //   m_yBottom = yBottom;
  // }

  _updateYBottom(yBottom);

  //TODO: temporal: test-50

  m_yHead = yHead;

  // Remove the line mask value which is out of the lane
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      int laneCls = inputLane.at<uchar>(i, j);
      int lineCls = inputLine.at<uchar>(i, j);

      if (lineCls == 255 && (i < m_yHead) && (i > m_yBottom))
      {
        inputLine.at<uchar>(i, j) = 0;
      }
    }
  }
}

// ============================================
//              Lane Calibration
// ============================================
void YOLOADAS::_noiseRemoval(cv::Mat &laneMask)
{
  imgUtil::findMaxContour(laneMask, laneMask);

  if (m_debugMode)
    cv::imshow("lane mask orig (proc)", laneMask);
}


void YOLOADAS::_masksMerging(cv::Mat &laneMask)
{
  if (m_prevLaneMaskList.size() != 0)
  {
    for (int i=0; i<m_prevLaneMaskList.size(); i++)
      laneMask += m_prevLaneMaskList[i];
  }

  if (m_debugMode)
    cv::imshow("lane mask merge", laneMask);
}


void YOLOADAS::_calibrateLaneMask()
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  if (m_debugMode)
    cv::imshow("lane mask orig", m_mainLaneMask);

  _noiseRemoval(m_mainLaneMask);

  int currLaneArea = cv::countNonZero(m_mainLaneMask);

  float currAreaRatio = (float)currLaneArea / ((float)m_inputWidth*(float)(m_yBottom - m_yHead));

  if (abs(currLaneArea-m_prevLaneArea) == 0)
    currAreaRatio = 0;


  // Save Information
  m_laneLineInfo.laneMaskInfo.prevArea = m_prevLaneArea;
  m_laneLineInfo.laneMaskInfo.currArea = currLaneArea;
  m_laneLineInfo.laneMaskInfo.currAreaRatio = currAreaRatio;

  m_logger->debug("abs(currLaneArea-m_prevLaneArea) = {}", abs(currLaneArea-m_prevLaneArea));
  m_logger->debug("min(currLaneArea, m_prevLaneArea) = {}", min(currLaneArea, m_prevLaneArea));
  m_logger->debug("currLaneArea = {}", currLaneArea);
  m_logger->debug("m_prevLaneArea = {}", m_prevLaneArea);
  m_logger->debug("currAreaRatio = {}", currAreaRatio);

  // TODO: use counter will be better
  if (currAreaRatio > 0.3 && m_prevLaneArea > 10 && currLaneArea != 0)
  {
    m_laneLineInfo.laneMaskInfo.usePrevLaneMask = true;
    m_logger->debug("use prev lane mask");
    m_mainLaneMask = m_prevLaneMask.clone();
  }
  else
  {
    // STEP2: Merge previous masks
    cv::Mat laneMask = m_mainLaneMask.clone();
    _masksMerging(laneMask);

    // STEP3: Save current lane mask
    cv::Mat tmpMask = m_mainLaneMask.clone();
    if (m_prevLaneMaskList.size() >= m_maxLaneMaskListSize)
    {
      m_prevLaneMaskList.erase(m_prevLaneMaskList.begin());
      m_prevLaneMaskList.push_back(tmpMask);
    }
    else
    {
      m_prevLaneMaskList.push_back(tmpMask);
    }

    //
    imgUtil::findMaxContour(laneMask, laneMask);

    m_prevLaneArea = currLaneArea;
    m_mainLaneMask = laneMask;
    m_prevLaneMask = laneMask;
  }
}


// ============================================
//                  Outputs
// ============================================
bool YOLOADAS::getLaneMask(cv::Mat &mask)
{
  mask = m_laneColor;
  return true;
}


bool YOLOADAS::getLineMask(cv::Mat &mask)
{
  mask = m_lineColor;
  return true;
}


bool YOLOADAS::getMainLaneMask(cv::Mat &mask)
{
  mask = m_mainLaneMask;
  return true;
}


bool YOLOADAS::getMainLineMask(cv::Mat &mask)
{
  mask = m_mainLineMask;
  return true;
}


bool YOLOADAS::getHorizontalLineMask(cv::Mat &mask)
{
  mask = m_horiLineMask;
  return true;
}


void YOLOADAS::getLaneLineInfo(LaneLineInfo &laneLineInfo)
{
  laneLineInfo.pLeftCarhood = m_pLeftCarhood;
  laneLineInfo.pRightCarhood = m_pRightCarhood;
  laneLineInfo.leftLineMissCount = m_laneLineCalib->m_leftMissCount;
  laneLineInfo.rightLineMissCount = m_laneLineCalib->m_rightMissCount;
  laneLineInfo.yLaneHead = m_yHead;
  laneLineInfo.yLaneBottom = m_yBottom;
  laneLineInfo.xLaneAvgMid = m_laneLineCalib->m_laneAvgMidX;
  laneLineInfo.xLeftLaneAvgMid = m_laneLineCalib->m_leftLaneAvgMidX;
  laneLineInfo.xRightLaneAvgMid = m_laneLineCalib->m_rightLaneAvgMidX;
  laneLineInfo.maxLaneWidth = m_maxLaneWidth;

  //
  laneLineInfo.horiLineArea = m_horiLineArea;

  //
  laneLineInfo.leftLinePointList = m_currLeftPointList;
  laneLineInfo.rightLinePointList = m_currRightPointList;
}


bool YOLOADAS::getVehicleBoundingBox(
  vector<BoundingBox> &_outBboxList,
  float confidence,
  int videoWidth, int videoHeight,
  BoundingBox &fcwROI)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  // Clear previous bounding boxes
  _outBboxList.clear();

  m_logger->debug("Get vehicle box => m_numBox = {}", m_numBox);

  Point pROI_TL = fcwROI.getCornerPoint()[0];
  Point pROI_TR = fcwROI.getCornerPoint()[1];

  float wRatio = (float)m_inputWidth / (float)videoWidth;
  float hRatio = (float)m_inputHeight / (float)videoHeight;

  for(int i=0; i<m_numBox; i++)
  {
    v8xyxy box = m_yoloOut[i];
    if ((box.c == BIG_VEHICLE) && (box.c_prob >= confidence))
    {
      m_logger->debug("Get vehicle box [{}] : ({}, {}, {}, {}, {}, {})", \
        i, box.x1, box.y1, box.x2, box.y2, box.c, box.c_prob);

      BoundingBox bbox(box.x1, box.y1, box.x2, box.y2, box.c);

      float bboxWidth = (float)bbox.getWidth()/wRatio;
      float bboxHeight = (float)bbox.getHeight()/hRatio;

      // filter out outliers
      if ((bboxWidth > (float)videoWidth*0.8) || (bboxHeight > (float)videoHeight*0.8))
      {
        m_logger->debug("filter out outliers - (1)");
        m_logger->debug("bboxWidth > videoWidth*0.8) || (bboxHeight > videoHeight*0.8)");
        continue;
      }
      else if (bboxWidth < 15 || bboxHeight < 10)
      {
        m_logger->debug("filter out outliers - (2)");
        m_logger->debug("bboxWidth < 15 || bboxHeight < 10");
        continue;
      }
      else if (bbox.getAspectRatio() < 0.4)
      {
        m_logger->debug("filter out outliers - (3)");
        m_logger->debug("bboxA.getAspectRatio() < 0.4");
        continue;
      }

      // filter out vehicles that out of ROI
      Point cp = bbox.getCenterPoint();
      if (cp.x/wRatio < pROI_TL.x || cp.x/wRatio > pROI_TR.x)
      {
        m_logger->debug("cp = ({}, {})", cp.x, cp.y);
        m_logger->debug("pROI_TL = ({}, {})", pROI_TL.x, pROI_TL.y);
        m_logger->debug("pROI_TR = ({}, {})", pROI_TR.x, pROI_TR.y);
        m_logger->debug("filter out outliers - (3)");
        m_logger->debug("Out of FCW ROI");
        continue;
      }


      bbox.confidence = box.c_prob;
      _outBboxList.push_back(bbox);
    }
    else if (box.c_prob < confidence)
    {
      m_logger->debug("Get vehicle box [{}] : ({}, {}, {}, {}, {}, {})", \
        i, box.x1, box.y1, box.x2, box.y2, box.c, box.c_prob);
    }
  }

  return true;
}


bool YOLOADAS::getRiderBoundingBox(
  vector<BoundingBox> &_outBboxList,
  float confidence,
  int videoWidth, int videoHeight,
  BoundingBox &fcwROI)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  // Clear previous bounding boxes
  _outBboxList.clear();

  //
  m_logger->debug("Get rider box => m_numBox = {}", m_numBox);

  Point pROI_TL = fcwROI.getCornerPoint()[0];
  Point pROI_TR = fcwROI.getCornerPoint()[1];

  float wRatio = (float)m_inputWidth / (float)videoWidth;
  float hRatio = (float)m_inputHeight / (float)videoHeight;

  //
  vector<BoundingBox> tmpBboxList;

  for(int i=0; i<m_numBox; i++)
  {
    v8xyxy box = m_yoloOut[i];
    if ((box.c == SMALL_VEHICLE) && (box.c_prob >= confidence))  // Rider class
    {
      m_logger->debug("Get rider box [{}] : ({}, {}, {}, {}, {}, {})", \
        i, box.x1, box.y1, box.x2, box.y2, box.c, box.c_prob);

      BoundingBox bbox(box.x1, box.y1, box.x2, box.y2, box.c);

      // filter out vehicles that out of ROI
      Point cp = bbox.getCenterPoint();
      if (cp.x/wRatio < pROI_TL.x || cp.x/wRatio > pROI_TR.x)
      {
        m_logger->debug("filter out outliers - (1)");
        m_logger->debug("Out of FCW ROI");
        continue;
      }

      // TODO: Aspect Ratio

      //
      BoundingBox bboxRider(box.x1, box.y1, box.x2, box.y2, box.c);

      // Merge human box
      for (int j=0; j<m_numBox; j++)
      {
        if (j!=i)
        {
          v8xyxy boxB = m_yoloOut[j];
          BoundingBox bboxB(boxB.x1, boxB.y1, boxB.x2, boxB.y2, boxB.c);

          if (boxB.c == HUMAN)
          {
            int areaB = bboxB.getArea();
            float overlapRatio = _getBboxOverlapRatio(bbox, bboxB);

            if (overlapRatio > 0.1)
            {
              _bboxMerging(bbox, bboxB, 3, bboxRider); //TODO:
            }
          }
        }
      }
      bboxRider.confidence = box.c_prob;
      tmpBboxList.push_back(bboxRider);
    }
  }


  for(int i=0; i<tmpBboxList.size(); i++)
  {
    int areaA = tmpBboxList[i].getArea();

    bool keepThisRider = true;

    for (int j=0; j<tmpBboxList.size(); j++)
    {
      if (j!=i)
      {
        int areaB = tmpBboxList[j].getArea();

        float overlapRatio = _getBboxOverlapRatio(tmpBboxList[i], tmpBboxList[j]);

        if ((overlapRatio > 0.1) && (areaB < areaA))
          keepThisRider = false;
      }
    }

    if (keepThisRider)
      _outBboxList.push_back(tmpBboxList[i]);
  }

  return true;
}


bool YOLOADAS::getHumanBoundingBox(
  vector<BoundingBox> &_outBboxList,
  float confidence,
  int videoWidth, int videoHeight,
  BoundingBox &fcwROI)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  // Clear previous bounding boxes
  _outBboxList.clear();

  m_logger->debug("Get human box => m_numBox = {}", m_numBox);

  Point pROI_TL = fcwROI.getCornerPoint()[0];
  Point pROI_TR = fcwROI.getCornerPoint()[1];

  float wRatio = (float)m_inputWidth / (float)videoWidth;
  float hRatio = (float)m_inputHeight / (float)videoHeight;

  for(int i=0; i<m_numBox; i++)
  {
    v8xyxy box = m_yoloOut[i];
    if ((box.c == HUMAN) && (box.c_prob >= confidence))
    {
      m_logger->debug("Get human box [{}] : ({}, {}, {}, {}, {}, {})", \
        i, box.x1, box.y1, box.x2, box.y2, box.c, box.c_prob);

      BoundingBox bbox(box.x1, box.y1, box.x2, box.y2, box.c);

      // filter out vehicles that out of ROI
      Point cp = bbox.getCenterPoint();
      if (cp.x/wRatio < pROI_TL.x || cp.x/wRatio > pROI_TR.x)
      {
        m_logger->debug("filter out outliers - (1)");
        m_logger->debug("Out of FCW ROI");
        continue;
      }

      // Aspect Ratio TODO:

      bbox.confidence = box.c_prob;
      _outBboxList.push_back(bbox);
    }
  }
  return true;
}


bool YOLOADAS::getRoadSignBoundingBox(
  vector<BoundingBox> &_outBboxList,
  float confidence,
  int videoWidth,
  int videoHeight,
  BoundingBox &fcwROI)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  // Clear previous bounding boxes
  _outBboxList.clear();

  //
  m_logger->debug("Get road sign box => m_numBox = {}", m_numBox);

  Point pROI_TL = fcwROI.getCornerPoint()[0];
  Point pROI_TR = fcwROI.getCornerPoint()[1];

  float wRatio = (float)m_inputWidth / (float)videoWidth;
  float hRatio = (float)m_inputHeight / (float)videoHeight;

  //
  vector<BoundingBox> tmpBboxList;

  for(int i=0; i<m_numBox; i++)
  {
    v8xyxy box = m_yoloOut[i];
    if ((box.c == ROAD_SIGN) && (box.c_prob >= confidence))  // Rider class
    {
      m_logger->debug("Get road sign box [{}] : ({}, {}, {}, {}, {}, {})", \
        i, box.x1, box.y1, box.x2, box.y2, box.c, box.c_prob);

      BoundingBox bbox(box.x1, box.y1, box.x2, box.y2, box.c);

      // filter out vehicles that out of ROI
      Point cp = bbox.getCenterPoint();
      // if (cp.x < pROI_TL.x || cp.x > pROI_TR.x)
      // {
      //   DEBUG_PRINT("filter out outliers - (1)\n");
      //   DEBUG_PRINT("Out of FCW ROI\n");
      //   continue;
      // }

      // TODO: Aspect Ratio

      //
      BoundingBox bboxRoadSign(box.x1, box.y1, box.x2, box.y2, box.c);

      // Merge human box
      // for (int j=0; j<m_numBox; j++)
      // {
      //   if (j!=i)
      //   {
      //     v8xyxy boxB = m_yoloOut[j];
      //     BoundingBox bboxB(boxB.x1, boxB.y1, boxB.x2, boxB.y2, boxB.c);

      //     if (boxB.c == HUMAN)
      //     {
      //       int areaB = bboxB.getArea();
      //       float overlapRatio = _getBboxOverlapRatio(bbox, bboxB);

      //       if (overlapRatio > 0.1)
      //       {
      //         _bboxMerging(bbox, bboxB, 3, bboxRider); //TODO:
      //       }
      //     }
      //   }
      // }
      bboxRoadSign.confidence = box.c_prob;
      _outBboxList.push_back(bboxRoadSign);
    }
  }

  return true;
}


void YOLOADAS::genResultImage(cv::Mat &img, vector<BoundingBox> bboxList, int colorIdx)
{
  // cv::Mat filllMap = vehicalMap.clone();
  cv::Mat filllMap(cv::Size(img.cols, img.rows), CV_8UC3, cv::Scalar::all(0));
  // cv::Mat imgCar = img.clone();

  // cout << "=> Size of BBox List = " << bboxList.size() << endl;
  // draw the rescaled bbx to the matrix
  for(int i=0; i<bboxList.size(); i++){
      BoundingBox box = bboxList[i];
      cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), object_colors[colorIdx], 1.5/*thickness*/);
      cv::rectangle(filllMap, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), object_colors[colorIdx], -1/*fill*/);
      cv::addWeighted(img, 1.0, filllMap, 0.5, 0, img);

      // cout << "box = (" << box.x1 << "," << box.y1 << "," << box.x2 << "," << box.y2 << ")" << endl;


      // TODO: maybe detect wheel in the future?
      // BoundingBox halfCar = BoundingBox(box.x1, box.getCenterPoint().y, box.x2, box.y2, box.label);

      // cv::Mat car;
      // cv::Mat carBinary;
      // cropImages(imgCar, car, halfCar);
      // cv::resize(car, car, cv::Size(320, 160), cv::INTER_LINEAR);
      // cv::imshow("car gray"+to_string(i), car);

      // cv::cvtColor(car, carBinary, cv::COLOR_BGR2GRAY);
      // // cv::threshold(carBinary, carBinary, 60, 255, cv::THRESH_OTSU);


      // cv::imshow("car binary"+to_string(i), carBinary);

      // cv::Mat road;
      // BoundingBox roadROI = BoundingBox(0, 120, 320, 160, box.label);
      // cropImages(carBinary, road, roadROI);

      // cv::imshow("road binary"+to_string(i), road);
      // cv::waitKey(0);
  }

  // write the matrix back to a new result image file
  //sstring resultFileName = string("result_")+filename;
  // string resFile = "test.jpg";
  // resFile = resFile.insert(resFile.rfind("."), "_result");
}


void YOLOADAS::genTrackObjectImage(cv::Mat &imgFrame, vector<Object> objectList)
{
  auto m_logger = spdlog::get("YOLO-ADAS");

  cv::Mat img = imgFrame;
  if (img.empty())
  {
    m_logger->error(" error!  image don't exist!");
    exit(1);
  }

  // draw the rescaled bbx to the matrix
  for(int i=0; i<objectList.size(); i++)
  {
    if (objectList[i].aliveCounter < 10)
      continue;
    BoundingBox box = objectList[i].bbox;
    cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), object_colors[0], 2/*thickness*/);
  }
  // write the matrix back to a new result image file
  //sstring resultFileName = string("result_")+filename;
  string resFile = "test.jpg";
  resFile = resFile.insert(resFile.rfind("."), "_result");
  // cv::imwrite(resFile, img);

  cv::imshow("Detection Result", img);
}


// ============================================
//                  Others
// ============================================

void YOLOADAS::_colorize(cv::Mat &inputLane, cv::Mat &inputLine, cv::Mat &outputLane, cv::Mat &outputLine)
{
  // Colorize class mask to 3-channels color map for segmentation maps
  int rows = inputLane.size[0];
  int columns = inputLane.size[1];

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      // Lane
      int laneCls = inputLane.at<uchar>(i, j);
      outputLane.at<cv::Vec3b>(i, j)[0] = lane_colors[laneCls][0];
      outputLane.at<cv::Vec3b>(i, j)[1] = lane_colors[laneCls][1];
      outputLane.at<cv::Vec3b>(i, j)[2] = lane_colors[laneCls][2];

      // Line
      int lineCls = inputLine.at<uchar>(i, j);
      outputLine.at<cv::Vec3b>(i, j)[0] = line_colors[lineCls][0];
      outputLine.at<cv::Vec3b>(i, j)[1] = line_colors[lineCls][1];
      outputLine.at<cv::Vec3b>(i, j)[2] = line_colors[lineCls][2];
    }
  }
}


void YOLOADAS::debugON()
{
  m_debugMode = true;
}


void YOLOADAS::showProcTime()
{
  m_estimateTime = true;
}
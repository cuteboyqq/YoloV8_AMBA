/*******************************************************************************
 * test_yolov8.c
 *
 * History:
 *  2023/10/02  - [Alister Hsu] created
 *
 ******************************************************************************/
#include "yolov8_class.h"
#include "yolov8_utils/object.hpp"
#include "yolov8_utils/point.hpp"
#include "yolov8_utils/bounding_box.hpp"
using namespace std;


int main(int argc, char **argv)
{
	int rval = 0;
	//test_eazyai_params_t params;
	live_params_t params;
	// signal(SIGINT, sig_stop);
	// signal(SIGQUIT, sig_stop);
	// signal(SIGTERM, sig_stop);
	memset(&live_ctx, 0, sizeof(live_ctx_t));
	int sig_flag = 0;
	std::vector<BoundingBox> bboxList;
	// YoloV8_Class yolov8(argc, argv, &params, &live_ctx);
	YoloV8_Class yolov8;
	yolov8.test_yolov8_init(argc, argv, &params, &live_ctx);
	do {
		sig_flag = yolov8.test_yolov8_run_2(&live_ctx,&params); //RVAL_OK
		bboxList.clear();
		bboxList = yolov8.Get_yolov8_Bounding_Boxes(&live_ctx, &params, bboxList);
		yolov8.Draw_Yolov8_Bounding_Boxes(bboxList,&live_ctx, &params);
	} while (sig_flag==0);
	yolov8.test_yolov8_deinit(&live_ctx, &params);

	return rval;
}

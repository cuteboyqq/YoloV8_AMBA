/*******************************************************************************
 * test_yolov8.c
 *
 * History:
 *  2023/10/02  - [Alister Hsu] created
 *
 ******************************************************************************/
#include "yolov8_class.h"
using namespace std;


int main(int argc, char **argv)
{
	int rval = 0;
	//test_eazyai_params_t params;
	live_params_t params;

	signal(SIGINT, sig_stop);
	signal(SIGQUIT, sig_stop);
	signal(SIGTERM, sig_stop);

	memset(&live_ctx, 0, sizeof(live_ctx_t));

	YoloV8_Class yolov8;
	do {
		yolov8.init_param(argc, argv, &params); //RVAL_OK
		yolov8.live_init(&live_ctx, &params); //RVAL_OK
		yolov8.live_run_loop(&live_ctx, &params); //RVAL_OK
	} while (0);
	yolov8.live_deinit(&live_ctx, &params);

	return rval;
}

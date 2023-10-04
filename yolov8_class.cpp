/*******************************************************************************
 * test_yolov8.h
 *
 ******************************************************************************/
#include "yolov8_class.h"
#include "yolov8_utils/object.hpp"
#include "yolov8_utils/point.hpp"
#include "yolov8_utils/bounding_box.hpp"
using namespace cv;
YoloV8_Class::YoloV8_Class()
{

};

YoloV8_Class::YoloV8_Class(int argc, char **argv, live_params_t *params, live_ctx_t *live_ctx)
{
	int rval = 0;
	rval = YoloV8_Class::init_param(argc, argv, params);
	rval = YoloV8_Class::live_init(live_ctx, params);
}
YoloV8_Class::~YoloV8_Class()
{
	post_thread_deinit(&live_ctx->thread_ctx, &live_ctx->nn_cvflow);
	nn_cvflow_deinit(&live_ctx->nn_cvflow);
	cv_env_deinit(live_ctx);
	if (live_ctx->f_result > -1) {
		close(live_ctx->f_result);
		live_ctx->f_result = -1;
	}
}


int YoloV8_Class::test_yolov8_init(int argc, char **argv, live_params_t *params, live_ctx_t *live_ctx)
{
    int rval = 0;
	rval = YoloV8_Class::init_param(argc, argv, params);
	rval = YoloV8_Class::live_init(live_ctx,params);
	return rval;
}

int YoloV8_Class::init(int argc, char **argv, live_params_t *params, live_ctx_t *live_ctx)
{
	int rval = 0;
	rval = YoloV8_Class::init_param(argc, argv, params);
	rval = YoloV8_Class::live_init(live_ctx,params);
	return rval;
};


int YoloV8_Class::parse_param(int argc, char **argv, live_params_t *params)
{
	int rval = EA_SUCCESS;
	int ch;
	int option_index = 0;
	int value;

	do {
		opterr = 0;
		while ((ch = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
		switch (ch) {
			case 'm':
				params->mode = atoi(optarg);
				if (params->mode < RUN_LIVE_MODE || params->mode >= MAX_RUN_MODE) {
					EA_LOG_ERROR("mode is wrong, %d\n", params->mode);
					rval = EA_FAIL;
					break;
				}
				break;
			case 'v':
				params->log_level = atoi(optarg);
				if (params->log_level < EA_LOG_LEVEL_NONE || params->log_level >= EA_LOG_LEVEL_INVALID) {
					EA_LOG_ERROR("log_level is wrong, %d\n", params->log_level);
					rval = EA_FAIL;
					break;
				}
				break;
			case OPTION_OUTPUT_DIR:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("output_dir is empty\n");
					rval = EA_FAIL;
					break;
				}
				if (value > MAX_STR_LEN) {
					EA_LOG_ERROR("output_dir should be no more than %d characters, %s\n", MAX_STR_LEN, optarg);
					rval = EA_FAIL;
					break;
				}
				strncpy(params->output_dir, optarg, MAX_STR_LEN);
				if (optarg[value - 1] != '/') {
					strncat(params->output_dir, "/", MAX_STR_LEN - strlen(params->output_dir));
				}
				params->feature |= (OUT_TYPE_JPEG & (~OUT_TYPE_VOUT));
				break;
			case OPTION_EXTRA_INPUT:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("path for extra input is empty\n");
					rval = EA_FAIL;
					break;
				}
				if (value > MAX_STR_LEN) {
					EA_LOG_ERROR("path for extra input should be no more than %d characters, %s\n", MAX_STR_LEN, optarg);
					rval = EA_FAIL;
					break;
				}
				strncpy(params->extra_input, optarg, MAX_STR_LEN);
				break;
			case OPTION_SUPPORT_LIST:
				params->support = 1;
				break;
			case 'd':
				value = atoi(optarg);
				if (value < DRAW_BBOX_TEXTBOX || value > DRAW_256_COLORS_IMAGE) {
					EA_LOG_ERROR("draw mode is wrong, %d\n", params->draw_mode);
					rval = EA_FAIL;
					break;
				}
				params->draw_mode = value;

				break;
			case 's':
				params->stream_id = atoi(optarg);
				if (params->stream_id < 0) {
					EA_LOG_ERROR("stream ID is wrong, %d\n", params->stream_id);
					rval = EA_FAIL;
					break;
				}
				params->feature |= OUT_TYPE_STREAM;
				break;
			case 'c':
				params->canvas_id = atoi(optarg);
				if (params->canvas_id < 0) {
					EA_LOG_ERROR("canvas ID is wrong, %d\n", params->canvas_id);
					rval = EA_FAIL;
					break;
				}
				params->feature |= IN_TYPE_CANVAS_BUFFER;
				break;
			case 'p':
				value = sscanf(optarg, "%d,%d",
					&params->pyramid[0], &params->pyramid[1]);
				if (value != 2) {
					printf("pyramid parameters are wrong, %s\n", optarg);
					rval = EA_FAIL;
					break;
				}
				params->use_pyramid = IN_SRC_ON;
				params->feature |= IN_TYPE_PYRAMID_BUFFER;
				break;
			case OPTION_MULTI_IN:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("multi in is empyt\n");
					rval = EA_FAIL;
					break;
				}
				strncpy(params->multi_in_params[params->multi_in_num],
					optarg, MAX_STR_LEN);
				params->multi_in_num++;
				if (params->multi_in_num > NN_MAX_PORT_NUM) {
					EA_LOG_ERROR("network input number more than %d\n", NN_MAX_PORT_NUM);
					rval = EA_FAIL;
					break;
				}
				break;
			case OPTION_MODEL_PATH:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("model_path is empty\n");
					rval = EA_FAIL;
					break;
				}
				params->model_path = optarg;
				break;
			case OPTION_LABEL_PATH:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("label_path is empty\n");
					rval = EA_FAIL;
					break;
				}

				params->label_path = optarg;
				break;
			case 'n':
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("model name is empty\n");
					rval = EA_FAIL;
					break;
				}
				params->arm_nn_name = optarg;
				break;
			case 'r':
				params->rgb = RGB_PLANAR;
				break;
			case OPTION_LUA_FILE_PATH:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("lua file is empty\n");
					rval = EA_FAIL;
					break;
				}
				params->lua_file_path = optarg;
				break;
			case OPTION_QUEUE_SIZE:
				value = atoi(optarg);
				if (value <= 0) {
					EA_LOG_ERROR("queue is empty\n");
					rval = EA_FAIL;
					break;
				}
				params->queue_size = value;
				break;
			case OPTION_CLASS_NUM:
				value = atoi(optarg);
				if (value < 0) {
					EA_LOG_ERROR("class num should > 0\n");
					rval = EA_FAIL;
					break;
				}
				params->class_num = value;
				break;
			case OPTION_THREAD_NUM:
				value = atoi(optarg);
				if (value <= 0) {
					EA_LOG_ERROR("thread num should > 0, default is 1\n");
					rval = EA_FAIL;
					break;
				}
				params->thread_num = value;
				break;
			case OPTION_ADES_CMD_FILE:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("ades command file is empty\n");
					rval = EA_FAIL;
					break;
				}
				params->ades_cmd_file = optarg;
				break;
			case OPTION_ACINF_GPU_ID:
				value = atoi(optarg);
				if (value < -1)  {
					EA_LOG_ERROR("acinf_gpu id should be >= -1");
					rval = EA_FAIL;
					break;
				}
				params->acinf_gpu_id = value;
				break;
			case OPTION_OVERLAY_BUF_OFFSET:
				params->overlay_buffer_offset = atoi(optarg);
				break;
			case OPTION_YUV:
				params->yuv_flag = IN_SRC_ON;
				params->multi_in_num = 2;
				break;
			case OPTION_FSYNC_OFF:
				params->enable_fsync_flag = IN_SRC_OFF;
				break;
			case OPTION_HOLD_IMG:
				params->enable_hold_img_flag = IN_SRC_ON;
				break;
			case OPTION_IN_ROI:
				value = sscanf(optarg, "%d,%d,%d,%d",
					&(params->roi.x), &(params->roi.y),
					&(params->roi.h), &(params->roi.w));
				if (value != 4) {
					printf("roi parameters are wrong, %s\n", optarg);
					rval = EA_FAIL;
					break;
				}
				break;
			case OPTION_VOUT_ID:
				value = atoi(optarg);
				if (value < EA_DISPLAY_DIGITAL_VOUT ||
					value >= EA_DISPLAY_VOUT_NUM)  {
					EA_LOG_ERROR("vout id parameter is wrong, %d\n", value);
					rval = EA_FAIL;
					break;
				}
				params->vout_id = value;
				params->feature |= OUT_TYPE_VOUT;
				break;
			case OPTION_RESULT_TO_TXT:
				value = strlen(optarg);
				if (value == 0) {
					EA_LOG_ERROR("The path of txt file to save result is empty\n");
					rval = EA_FAIL;
					break;
				}
				params->result_f_path = optarg;
				break;
			default:
				EA_LOG_ERROR("unknown option found: %c\n", ch);
				rval = EA_FAIL;
				break;
			}
		}
	} while (0);

	return rval;
};

int YoloV8_Class::check_params(live_params_t *params)
{
	int rval = EA_SUCCESS;
	do {
		if (((params->feature & OUT_TYPE_LIVE) == OUT_TYPE_LIVE ||
			((params->feature & OUT_TYPE_LIVE) == 0 &&
			params->stream_id >= 0 && params->vout_id >= 0)) &&
			params->mode == RUN_LIVE_MODE) {
			EA_LOG_ERROR("Stream and Vout are set simultaneously. Only support one of them in live mode.\n");
			rval = EA_FAIL;
			break;
		}
		if ((params->feature & IN_TYPE_LIVE) == IN_TYPE_LIVE &&
			params->mode == RUN_LIVE_MODE) {
			EA_LOG_ERROR("Canvas and Pyramid buffer are set simultaneously. Only support one of them in live mode.\n");
			rval = EA_FAIL;
			break;
		}
	} while (0);

	return rval;
};

int YoloV8_Class::init_param(int argc, char **argv, live_params_t *params)
{
	int rval = EA_SUCCESS;

	memset(params, 0, sizeof(live_params_t));

	params->mode = RUN_LIVE_MODE;
	params->log_level = EA_LOG_LEVEL_NOTICE;
	params->draw_mode = DRAW_BBOX_TEXTBOX;
	params->rgb = BGR_PLANAR;

	params->yuv_flag = IN_SRC_OFF;
	params->use_pyramid = IN_SRC_OFF;
	params->enable_fsync_flag = IN_SRC_ON;

	params->canvas_id = DEFAULT_CANVAS_ID;
	params->vout_id = DEFAULT_VOUT_ID;
	params->stream_id = DEFAULT_STREAM_ID;

	params->queue_size = 1;
	params->thread_num = 1;
	params->acinf_gpu_id = -1;
	params->overlay_buffer_offset = -1;

	do {
		if (argc < 2) {
			usage();
			exit(0);
		} else {
			RVAL_OK(YoloV8_Class::parse_param(argc, argv, params));
		}
		if (!!params->support) {
			support_list();
			exit(0);
		}

		RVAL_OK(YoloV8_Class::check_params(params));
		if (params->mode == RUN_LIVE_MODE) {
			if (params->stream_id < 0) {
				params->feature |= OUT_TYPE_VOUT;
				if (params->vout_id < 0) {
					params->vout_id = EA_DISPLAY_ANALOG_VOUT;
				}
			} else if (params->stream_id >= 0 && params->vout_id < 0) {
				params->feature |= OUT_TYPE_STREAM;
			}

			if (params->use_pyramid == IN_SRC_OFF) {
				params->feature |= IN_TYPE_CANVAS_BUFFER;
				if (params->canvas_id < 0) {
					params->canvas_id = 1;
				}
			}
		}

		EA_LOG_SET_LOCAL(params->log_level);
		EA_LOG_NOTICE("live parameters:\n");
		EA_LOG_NOTICE("\tmode: %d\n", params->mode);
		EA_LOG_NOTICE("\toutput_dir path: %s\n", params->output_dir);
		EA_LOG_NOTICE("\tlog level: %d\n", params->log_level);
		EA_LOG_NOTICE("\tdraw mode: %d\n", params->draw_mode);
		EA_LOG_NOTICE("\tstream ID: %d\n", params->stream_id);
		EA_LOG_NOTICE("\tVOUT ID: %d\n", params->vout_id);
		EA_LOG_NOTICE("\tmodel path: %s\n", params->model_path);
		EA_LOG_NOTICE("\tlabel path: %s\n", params->label_path);
		EA_LOG_NOTICE("\tmodel name: %s\n", params->arm_nn_name);
		EA_LOG_NOTICE("\tqueue size: %d\n", params->queue_size);
		EA_LOG_NOTICE("\trgb type: %d\n", params->rgb);
		EA_LOG_NOTICE("\tcanvas id: %d\n", params->canvas_id);
		EA_LOG_NOTICE("\tfile name of saving result to txt: %s\n",
			params->result_f_path);

		if (params->draw_mode == DRAW_BBOX_TEXTBOX) {
			params->draw_mode = EA_DISPLAY_BBOX_TEXTBOX;
		} else if (params->draw_mode == DRAW_256_COLORS_IMAGE) {
			params->draw_mode = EA_DISPLAY_256_COLORS;
		}

	} while (0);

	return rval;
};

void YoloV8_Class::live_set_post_thread_params(live_params_t *params,
	post_thread_params_t *post_params)
{
	post_params->label_path = params->label_path;
	post_params->class_num = params->class_num;
	post_params->arm_nn_name = params->arm_nn_name;
	post_params->queue_size = params->queue_size;
	post_params->thread_num = params->thread_num;
	post_params->lua_file_path = params->lua_file_path;
	post_params->in_mode = params->mode;
	post_params->log_level = params->log_level;
	post_params->output_dir = params->output_dir;
	post_params->extra_input = params->extra_input;
	post_params->use_pyramid = params->use_pyramid;
	post_params->total_count = INTERVAL_PRINT_PROCESS_TIME;
	post_params->enable_fsync_flag = params->enable_fsync_flag;
	if (params->use_pyramid == IN_SRC_ON &&
		params->feature & LIVE_MODE_ALL) {
		post_params->img_src_id = params->pyramid[1];
	} else {
		post_params->img_src_id = 0;
	}
};

int YoloV8_Class::cv_env_init(live_ctx_t *live_ctx, live_params_t *params)
{
	int rval = EA_SUCCESS;
	nn_input_ops_type_t *ops = NULL;
	nn_input_context_type_t *ctx;
	int i;
	do {
		ctx = &live_ctx->nn_input_ctx;
		ctx->canvas_id = params->canvas_id;
		ctx->stream_id = params->stream_id;
		ctx->use_pyramid = params->use_pyramid;
		ctx->feature = params->feature;
		ctx->yuv_flag = params->yuv_flag;
		ctx->pyramid[0] = params->pyramid[0];
		ctx->pyramid[1] = params->pyramid[1];
		ctx->multi_in_num = params->multi_in_num;
		ctx->rgb = params->rgb;
		ctx->draw_mode = params->draw_mode;
		ctx->output_dir = params->output_dir;
		ctx->overlay_buffer_offset = params->overlay_buffer_offset;
		ctx->roi.x = params->roi.x;
		ctx->roi.y = params->roi.y;
		ctx->roi.h = params->roi.h;
		ctx->roi.w = params->roi.w;
		ctx->vout_id = params->vout_id;
		for (i = 0; i < params->multi_in_num; i++) {
			strncpy(ctx->multi_in_params[i],
				params->multi_in_params[i], MAX_STR_LEN);
		}
		ctx->ops = nn_intput_get_ops(run_mode_t(params->mode));
		RVAL_ASSERT(ctx->ops != NULL);
		ops = ctx->ops;
		RVAL_ASSERT(ops->nn_input_init != NULL);
		RVAL_OK(ops->nn_input_init(ctx));
	} while (0);
	return rval;
};



int YoloV8_Class::live_init(live_ctx_t *live_ctx, live_params_t *params)
{
	int rval = EA_SUCCESS;
	nn_cvflow_params_t net_params;
	nn_input_ops_type_t *ops = NULL;
	do {
		RVAL_ASSERT(live_ctx != NULL);
		RVAL_ASSERT(params != NULL);
		live_ctx->loop_count = INTERVAL_PRINT_PROCESS_TIME;
		live_ctx->f_result = -1;
		if (params->result_f_path &&
			params->mode == RUN_FILE_MODE) {
			live_ctx->f_result = open(params->result_f_path, O_CREAT | O_RDWR | O_TRUNC, 0644);
			RVAL_ASSERT(live_ctx->f_result != -1);
		}
		RVAL_OK(cv_env_init(live_ctx, params));
		ops = live_ctx->nn_input_ctx.ops;
		RVAL_ASSERT(params->model_path != NULL);
		memset(&net_params, 0, sizeof(nn_cvflow_params_t));
		net_params.log_level = params->log_level;
		net_params.model_path = params->model_path;
		net_params.ades_cmd_file = params->ades_cmd_file;
		net_params.acinf_gpu_id = params->acinf_gpu_id;
		RVAL_OK(nn_cvflow_init(&live_ctx->nn_cvflow, &net_params));
		live_ctx->nn_input_ctx.net = (live_ctx->nn_cvflow.net);
		RVAL_ASSERT(ops->nn_input_check_params != NULL);
		RVAL_OK(ops->nn_input_check_params(&live_ctx->nn_input_ctx));
		live_ctx->display = live_ctx->nn_input_ctx.display;
		if (params->mode != RUN_DUMMY_MODE) {
			YoloV8_Class::live_set_post_thread_params(params, &live_ctx->thread_ctx.params);
			live_ctx->thread_ctx.display = live_ctx->display;
			live_ctx->thread_ctx.input_ctx = &live_ctx->nn_input_ctx;
			live_ctx->thread_ctx.f_result = live_ctx->f_result;
			RVAL_OK(post_thread_init(&live_ctx->thread_ctx, &live_ctx->nn_cvflow));
			RVAL_OK(post_thread_set_notifier(&live_ctx->thread_ctx, notifier, &live_ctx->sig_flag));
		}
	} while (0);
	return rval;
}
int YoloV8_Class::test_yolov8_run_2(live_ctx_t *live_ctx, live_params_t *params)
{
	int rval = EA_SUCCESS;
	int sig_flag = 0;
	do {
		if (params->mode == RUN_DUMMY_MODE) {
			printf("In params->mode == RUN_DUMMY_MODE\n");
			sig_flag = YoloV8_Class::live_run_loop_dummy(live_ctx, params); //RVAL_OK
		} else {
			printf("In else \n");
			sig_flag = YoloV8_Class::live_run_loop_without_dummy(live_ctx, params); //RVAL_OK
		}
	} while (0);
	// return rval;
	return sig_flag;
}
int YoloV8_Class::test_yolov8_run()
{	
	printf("In test_yolov8_run\n");
    int rval = EA_SUCCESS;
	int sig_flag = 0;
	do {
		printf("In test_yolov8_run , start if (params->mode == RUN_DUMMY_MODE) \n");
		if (params->mode == RUN_DUMMY_MODE) {
			printf("Start live_run_loop_dummy\n");
			sig_flag = YoloV8_Class::live_run_loop_dummy(live_ctx, params); //RVAL_OK
		} else {
			printf("Start live_run_loop_without_dummy\n");
			sig_flag = YoloV8_Class::live_run_loop_without_dummy(live_ctx, params); //RVAL_OK
		}
	} while (0);
	// return rval;
	return sig_flag;
}
Object YoloV8_Class::test_yolov8_tracker(live_ctx_t *live_ctx, live_params_t *params)
{
	Object obj;
 	yolov8_result_t *yolov8_result = (yolov8_result_t *)live_ctx->thread_ctx.thread->nn_arm_ctx.result;
	int i = 0;
	printf("in tracker function~~~");
	for ( i = 0; i < yolov8_result->num; i++)
	{
		EA_LOG_DEBUG("num:%d, id:%d, x1:%f, y1:%f, x2:%f, y2:%f, score:%f, label:%s\n",
		yolov8_result->num,
		yolov8_result->bbox[i].id,
		yolov8_result->bbox[i].x_start,
		yolov8_result->bbox[i].y_start,
		yolov8_result->bbox[i].x_end,
		yolov8_result->bbox[i].y_end,
		yolov8_result->bbox[i].score,
		yolov8_result->bbox[i].label);
		
		printf("num:%d, id:%d, x1:%f, y1:%f, x2:%f, y2:%f, score:%f, label:%s\n",
		yolov8_result->num,
		yolov8_result->bbox[i].id,
		yolov8_result->bbox[i].x_start,
		yolov8_result->bbox[i].y_start,
		yolov8_result->bbox[i].x_end,
		yolov8_result->bbox[i].y_end,
		yolov8_result->bbox[i].score,
		yolov8_result->bbox[i].label);

		obj.bboxList.push_back(BoundingBox(yolov8_result->bbox[i].x_start,
								yolov8_result->bbox[i].y_start,
								yolov8_result->bbox[i].x_end,
								yolov8_result->bbox[i].y_end,
								yolov8_result->bbox[i].id));

		
		

	}
	for (int i=0;i<obj.bboxList.size();i++)
		{
			printf("%d",obj.bboxList[i].x1);
			printf("%d",obj.bboxList[i].y1);
			printf("%d",obj.bboxList[i].x2);
			printf("%d",obj.bboxList[i].y2);
		}
    return obj;
}
std::vector<BoundingBox> YoloV8_Class::Get_yolov8_Bounding_Boxes(live_ctx_t *live_ctx, live_params_t *params,std::vector<BoundingBox> bboxList)
{
    //Object obj;
	printf("Start initial yolov8_result_t~~~\n");
 	yolov8_result_t *yolov8_result = (yolov8_result_t *)live_ctx->thread_ctx.thread->nn_arm_ctx.result;
	printf("end initial yolov8_result_t~~~\n");
	int i = 0;
	printf("in tracker function~~~\n");
	for ( i = 0; i < yolov8_result->num; i++)
	{
		EA_LOG_DEBUG("num:%d, id:%d, x1:%f, y1:%f, x2:%f, y2:%f, score:%f, label:%s\n",
		yolov8_result->num,
		yolov8_result->bbox[i].id,
		yolov8_result->bbox[i].x_start,
		yolov8_result->bbox[i].y_start,
		yolov8_result->bbox[i].x_end,
		yolov8_result->bbox[i].y_end,
		yolov8_result->bbox[i].score,
		yolov8_result->bbox[i].label);
		
		printf("num:%d, id:%d, x1:%f, y1:%f, x2:%f, y2:%f, score:%f, label:%s\n",
		yolov8_result->num,
		yolov8_result->bbox[i].id,
		yolov8_result->bbox[i].x_start,
		yolov8_result->bbox[i].y_start,
		yolov8_result->bbox[i].x_end,
		yolov8_result->bbox[i].y_end,
		yolov8_result->bbox[i].score,
		yolov8_result->bbox[i].label);

		bboxList.push_back(BoundingBox(yolov8_result->bbox[i].x_start,
								yolov8_result->bbox[i].y_start,
								yolov8_result->bbox[i].x_end,
								yolov8_result->bbox[i].y_end,
								yolov8_result->bbox[i].id));

		
		

	}
	printf("Show bboxList ~~~~~~\n");
	for (int i=0;i<bboxList.size();i++)
		{	
			
			printf("%f, %f, %f, %f \n",bboxList[i].x1,
										bboxList[i].y1,
										bboxList[i].x2,
										bboxList[i].y2);
		
		}
    return bboxList;
}
std::vector<BoundingBox> YoloV8_Class::Get_yolov8_Bounding_Boxes(std::vector<BoundingBox> bboxList)
{
     //Object obj;
	 printf("start nitial yolov8_result~~~\n");
 	yolov8_result_t *yolov8_result = (yolov8_result_t *)live_ctx->thread_ctx.thread->nn_arm_ctx.result;
	printf("End nitial yolov8_result~~~\n");
	int i = 0;
	printf("in tracker function~~~");
	for ( i = 0; i < yolov8_result->num; i++)
	{
		EA_LOG_DEBUG("num:%d, id:%d, x1:%f, y1:%f, x2:%f, y2:%f, score:%f, label:%s\n",
		yolov8_result->num,
		yolov8_result->bbox[i].id,
		yolov8_result->bbox[i].x_start,
		yolov8_result->bbox[i].y_start,
		yolov8_result->bbox[i].x_end,
		yolov8_result->bbox[i].y_end,
		yolov8_result->bbox[i].score,
		yolov8_result->bbox[i].label);
		
		printf("num:%d, id:%d, x1:%f, y1:%f, x2:%f, y2:%f, score:%f, label:%s\n",
		yolov8_result->num,
		yolov8_result->bbox[i].id,
		yolov8_result->bbox[i].x_start,
		yolov8_result->bbox[i].y_start,
		yolov8_result->bbox[i].x_end,
		yolov8_result->bbox[i].y_end,
		yolov8_result->bbox[i].score,
		yolov8_result->bbox[i].label);

		bboxList.push_back(BoundingBox(yolov8_result->bbox[i].x_start,
								yolov8_result->bbox[i].y_start,
								yolov8_result->bbox[i].x_end,
								yolov8_result->bbox[i].y_end,
								yolov8_result->bbox[i].id));

		
		

	}
	printf("print BB~~~~~~~~~~~~~~~~~~");
	for (int i=0;i<int(bboxList.size());i++)
		{
			printf("%d",bboxList[i].x1);
			printf("%d",bboxList[i].y1);
			printf("%d",bboxList[i].x2);
			printf("%d",bboxList[i].y2);
		}
    return bboxList;
}

void YoloV8_Class::Draw_Yolov8_Bounding_Boxes(std::vector<BoundingBox> bboxList, live_ctx_t *live_ctx){
	printf("[Draw_Yolov8_Bounding_Boxes] Get dis_win_h and w\n");
	// int dis_win_h = live_ctx->thread_ctx.input_ctx->roi.h;
	// int dis_win_w = live_ctx->thread_ctx.input_ctx->roi.w;
	int dis_win_h = 1000;
	int dis_win_w = 500;
	printf("dis_win_h=%f,dis_win_w=%f\n ",dis_win_h,dis_win_w);
	printf("[Draw_Yolov8_Bounding_Boxes] start initial img \n");
	cv::Mat img(dis_win_h, dis_win_w, CV_8UC3,live_ctx->thread_ctx.input_queue );
	printf("[Draw_Yolov8_Bounding_Boxes] End initial img\n");
	for (int i=0;i<int(bboxList.size());i++)
		{
			int bbox_start_x = bboxList[i].x1 * dis_win_w;
			int bbox_start_y = bboxList[i].y1 * dis_win_h;
			int bbox_end_x = bboxList[i].x2 * dis_win_w;
			int bbox_end_y = bboxList[i].y2 * dis_win_h;

			//cv::rectangle()
			cv::Point pt1(bbox_start_x, bbox_start_y);
			cv::Point pt2(bbox_end_x, bbox_end_y );
			cv::rectangle(img, pt1, pt2, cv::Scalar(255,127,0), -1, cv::LINE_4);
			
		}
	printf("imshow");
	// EA_LOG_NOTICE("sdfsdfsdfsdfsdwerrwerwerwerwefdsdfsd");
	cv::imshow("test", img);
	
    // Wait for any keystroke
    cv::waitKey(0);
	// EA_LOG_NOTICE("sdfsdfsdfsdfsdfdsdfsd");
	printf("imwrite");
	cv::imwrite("./test_2023_10_04.jpg", img);


};

void YoloV8_Class::cv_env_deinit(live_ctx_t *live_ctx)
{
	nn_input_ops_type_t *ops = NULL;
	ops = live_ctx->nn_input_ctx.ops;
	if (ops) {
		EA_LOG_ASSERT(ops->nn_input_deinit != NULL);
		ops->nn_input_deinit(&live_ctx->nn_input_ctx);
	}
};

void YoloV8_Class::live_deinit(live_ctx_t *live_ctx, live_params_t *params)
{
	post_thread_deinit(&live_ctx->thread_ctx, &live_ctx->nn_cvflow);
	nn_cvflow_deinit(&live_ctx->nn_cvflow);
	cv_env_deinit(live_ctx);
	if (live_ctx->f_result > -1) {
		close(live_ctx->f_result);
		live_ctx->f_result = -1;
	}
}
void YoloV8_Class::test_yolov8_deinit(live_ctx_t *live_ctx, live_params_t *params){
	post_thread_deinit(&live_ctx->thread_ctx, &live_ctx->nn_cvflow);
	nn_cvflow_deinit(&live_ctx->nn_cvflow);
	cv_env_deinit(live_ctx);
	if (live_ctx->f_result > -1) {
		close(live_ctx->f_result);
		live_ctx->f_result = -1;
	}
};

int YoloV8_Class::live_update_net_output(live_ctx_t *live_ctx,
	vp_output_t **vp_output)
{
	int rval = EA_SUCCESS;
	ea_queue_t *queue = NULL;
	int i;
	vp_output_t *tmp;
	do {
		queue = post_thread_queue(&live_ctx->thread_ctx);
		*vp_output = (vp_output_t *)ea_queue_request_carrier(queue);
		RVAL_ASSERT(*vp_output != NULL);
		tmp = *vp_output;
		tmp->out_num = live_ctx->nn_cvflow.out_num;
		for (i = 0; i < tmp->out_num; i++) {
			ea_net_update_output(live_ctx->nn_cvflow.net, tmp->out[i].tensor_name,
				tmp->out[i].out);
		}
	} while (0);
	return rval;
};

int YoloV8_Class::live_run_loop_dummy(live_ctx_t *live_ctx, live_params_t *params)
{
	int rval = EA_SUCCESS;
	ea_calc_fps_ctx_t calc_fps_ctx;
	float fps;
	ea_img_resource_data_t data;
	memset(&calc_fps_ctx, 0, sizeof(ea_calc_fps_ctx_t));
	calc_fps_ctx.count_period = DEFAULT_FPS_COUNT_PERIOD;
	int i;
	nn_input_ops_type_t *ops = NULL;

	do {
		RVAL_ASSERT(live_ctx != NULL);
		ops = live_ctx->nn_input_ctx.ops;
		RVAL_ASSERT(ops->nn_input_hold_data != NULL);

		for (i = 0; i < live_ctx->nn_cvflow.in_num; i++) {
			RVAL_OK(ops->nn_input_hold_data(&live_ctx->nn_input_ctx,
				i, nn_cvflow_input(&live_ctx->nn_cvflow, i), &data));
		}

		do {
			EA_MEASURE_TIME_START();
			RVAL_OK(nn_cvflow_inference(&live_ctx->nn_cvflow));
			EA_MEASURE_TIME_END("network forward time: ");
			fps = ea_calc_fps(&calc_fps_ctx);
			if (fps > 0) {
				EA_LOG_NOTICE("fps %.1f\n", fps);
			}
		} while (live_ctx->sig_flag == 0);

		for (i = 0; i < live_ctx->nn_cvflow.in_num; i++) {
			RVAL_OK(ops->nn_input_release_data(&live_ctx->nn_input_ctx,
			&data, i));
		}
	} while (0);

	//return rval;
	return live_ctx->sig_flag;
};

int YoloV8_Class::live_convert_yuv_data_to_bgr_data_for_postprocess(live_params_t *params, img_set_t *img_set)
{
	int rval = EA_SUCCESS;
	size_t shape[EA_DIM] = {0};
	int img_src_id = 0;

	do {
		if (params->use_pyramid == IN_SRC_ON) {
			img_src_id = params->pyramid[1];
		}

		if (img_set->bgr == NULL) {
			shape[EA_N] = ea_tensor_shape(img_set->img[0].tensor_group[img_src_id])[EA_N];
			shape[EA_C] = 3;
			shape[EA_H] = ea_tensor_shape(img_set->img[0].tensor_group[img_src_id])[EA_H];
			shape[EA_W] = ea_tensor_shape(img_set->img[0].tensor_group[img_src_id])[EA_W];
			if (ea_tensor_related(img_set->img[0].tensor_group[img_src_id]) == NULL) {
				shape[EA_H] = shape[EA_H] * 2 / 3;
			}
			img_set->bgr = ea_tensor_new(EA_U8, shape, 0);
			RVAL_ASSERT(img_set->bgr != NULL);
		}
		RVAL_OK(ea_cvt_color_resize(img_set->img[0].tensor_group[img_src_id],
			img_set->bgr, EA_COLOR_YUV2BGR_NV12, EA_VP));
	} while (0);

	return rval;
};

int YoloV8_Class::live_run_loop_without_dummy(live_ctx_t *live_ctx, live_params_t *params)
{
	int rval = EA_SUCCESS;
	int i = 0;
	ea_calc_fps_ctx_t calc_fps_ctx;
	float fps;
	ea_queue_t *queue = NULL;
	vp_output_t *vp_output = NULL;
	calc_fps_ctx.count_period = DEFAULT_FPS_COUNT_PERIOD;
	img_set_t *img_set;
	nn_input_ops_type_t *ops = NULL;
	memset(&calc_fps_ctx, 0, sizeof(ea_calc_fps_ctx_t));
	int fps_notice_flag = 0;
	printf("In live_run_loop_without_dummy \n");
	do {
		RVAL_ASSERT(live_ctx != NULL);
		ops = live_ctx->nn_input_ctx.ops;
		RVAL_ASSERT(ops->nn_input_hold_data != NULL);
		printf("Start live_update_net_output\n");
		RVAL_OK(YoloV8_Class::live_update_net_output(live_ctx, &vp_output));
		printf("end live_update_net_output\n");
		printf("Start post_thread_get_img_set\n");
		img_set = post_thread_get_img_set(&live_ctx->thread_ctx, live_ctx->seq);
		printf("end post_thread_get_img_set\n");
		live_ctx->seq++;
		printf("start for  i < live_ctx->nn_cvflow.in_num\n");
		for (i = 0; i < live_ctx->nn_cvflow.in_num; i++) {
			RVAL_OK(ops->nn_input_hold_data(&live_ctx->nn_input_ctx,
				i, nn_cvflow_input(&live_ctx->nn_cvflow, i), &(img_set->img[i])));
			if (img_set->img[i].tensor_group == NULL) {
				EA_LOG_NOTICE("All files are handled\n");
				live_ctx->sig_flag = 1;
				break;
			}
		}
		printf("end for  i < live_ctx->nn_cvflow.in_num\n");
		RVAL_BREAK();
		printf("live_ctx->sig_flag = %d\n",live_ctx->sig_flag);
		if (live_ctx->sig_flag) {
			break;
		}
		printf("start live_convert_yuv_data_to_bgr_data_for_postprocess\n");
		if (params->mode == RUN_LIVE_MODE &&
			params->enable_hold_img_flag == IN_SRC_ON) {
			RVAL_OK(YoloV8_Class::live_convert_yuv_data_to_bgr_data_for_postprocess(params, img_set));
		}
		printf("end live_convert_yuv_data_to_bgr_data_for_postprocess\n");
		vp_output->arg = img_set;
		EA_MEASURE_TIME_START();
		RVAL_OK(nn_cvflow_inference(&live_ctx->nn_cvflow));
		live_ctx->loop_count--;
		if (live_ctx->loop_count == 0) {
			EA_MEASURE_TIME_END("network forward time: ");
			live_ctx->loop_count = INTERVAL_PRINT_PROCESS_TIME;
		}

		if (params->mode == RUN_LIVE_MODE) {
			fps = ea_calc_fps(&calc_fps_ctx);
			if (fps > 0) {
				if (fps_notice_flag == 0) {
					EA_LOG_NOTICE("!!! FPS based on frame query, preprocess, and inference.");
					fps_notice_flag = 1;
				}
				EA_LOG_NOTICE("fps %.1f\n", fps);
			}
		}
		for (i = 0; i < vp_output->out_num; i++) {
			RVAL_OK(ea_tensor_sync_cache(vp_output->out[i].out, EA_VP, EA_CPU));
		}
		RVAL_BREAK();
		printf("start post_thread_queue\n");
		queue = post_thread_queue(&live_ctx->thread_ctx);
		printf("end post_thread_queue = %d\n",queue);
		RVAL_OK(ea_queue_en(queue, vp_output));

	} while (0);
	printf("end while\n");
	// return rval;
	return live_ctx->sig_flag;
};

int YoloV8_Class::live_run_loop(live_ctx_t *live_ctx, live_params_t *params)
{
	int rval = EA_SUCCESS;
	int sig_flag = 0;
	do {
		if (params->mode == RUN_DUMMY_MODE) {
			sig_flag = YoloV8_Class::live_run_loop_dummy(live_ctx, params); //RVAL_OK
		} else {
			sig_flag = YoloV8_Class::live_run_loop_without_dummy(live_ctx, params); //RVAL_OK
		}
	} while (0);
	// return rval;
	return sig_flag;
};





#ifndef _YOLOV8_STRUCT_
#define _YOLOV8_STRUCT_
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <signal.h>
#include <getopt.h>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <eazyai.h>
#include <nn_arm.h>

#include "nn_cvflow.h"
#include "nn_input.h"
#include "post_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INTERVAL_PRINT_PROCESS_TIME 30
#define TO_FILE_POSTPROCESS_NAME "to_file"
#define DEFAULT_FPS_COUNT_PERIOD 100

#define CANVAS
#ifdef CANVAS
#define DEFAULT_CANVAS_ID 1
#else
#define DEFAULT_CANVAS_ID -1
#endif

#define VOUT
#ifdef VOUT
#define DEFAULT_VOUT_ID EA_DISPLAY_ANALOG_VOUT
#else
#define DEFAULT_VOUT_ID -1
#endif

#ifdef STREAM
#define DEFAULT_STREAM_ID 0
#else
#define DEFAULT_STREAM_ID -1
#endif


// Get from yolov8.cpp

#define YOLOV8_MAX_STR_LEN                (256)
#define YOLOV8_MAX_COLOR_NUM              (256)
#define YOLOV8_SEG_MAP_CLUT_NUM           (5)
#define YOLOV8_OUTPUT0_LOCAL_NUMBER       (4)

typedef struct yolov8_arm_cfg_s {
	float conf_threshold;
	int top_k;
	float nms_threshold;
	int class_num;
	int mask_num;
	int enable_seg;
	int log_level;
	int disable_fsync;

	char output_0[YOLOV8_MAX_STR_LEN];
	char output_1[YOLOV8_MAX_STR_LEN];
} yolov8_arm_cfg_t;

typedef struct yolov8_bbox_s {
	char label[YOLOV8_MAX_STR_LEN];
	int id;
	float score;
	float x_start; // normalized value
	float y_start;
	float x_end;
	float y_end;
} yolov8_bbox_t;

typedef struct yolov8_result_s {
	yolov8_bbox_t *bbox;
	int *mask_cls_id;
	int num;
} yolov8_result_t;

typedef struct yolov8_ctx_s {
	ea_tensor_t *output0_tensor;
	ea_tensor_t *output1_tensor;
	float *x1y1x2y2score;
	float *valid_x1y1x2y2score;
	int *index_array;
	int *valid_index;
	uint8_t *mask_map;

	int box_num;
	int input_w;
	int input_h;
	int mask_w;
	int mask_h;
} yolov8_ctx_t;

static yolov8_ctx_t yolov8_ctx;





static void notifier(void *arg)
{
	*((int *)arg) = 1;
}


typedef struct live_params_s {
	//Input parameters, include input from iav or jpg or raw file, run mode.
	int canvas_id;
	int pyramid[2];
	int mode;
	int multi_in_num;
	char multi_in_params[NN_MAX_PORT_NUM][MAX_STR_LEN + 1];
	int yuv_flag;
	int use_pyramid;
	int feature;

	//Preprocess parameters, include color conversion, roi.
	int rgb;
	ea_roi_t roi;

	//Inference parameters, include model path, label path, inference device, etc.
	const char *model_path;
	const char *ades_cmd_file;
	int acinf_gpu_id;

	//Model postprocessing parameters, include name, lua file path, etc.
	const char *arm_nn_name;
	int queue_size;
	const char *lua_file_path;
	int thread_num;
	char extra_input[MAX_STR_LEN + 1];
	const char *label_path;
	int class_num;
	int enable_hold_img_flag;

	//Showing results parameters, include showing results on stream or jpg file, saving result to file.
	int stream_id;
	int enable_fsync_flag;
	int overlay_buffer_offset;
	char *result_f_path;
	int vout_id;
	int draw_mode;
	char output_dir[MAX_STR_LEN + 1];

	//Miscellaneous
	int support;
	int log_level;

} live_params_t;

typedef struct live_ctx_s {
	nn_cvflow_t nn_cvflow;
	post_thread_ctx_t thread_ctx;
	ea_display_t *display;
	int sig_flag;
	unsigned int seq;
	nn_input_context_type_t nn_input_ctx;
	int loop_count;
	int f_result;
} live_ctx_t;

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();
static live_ctx_t live_ctx;

#define	NO_ARG	0
#define	HAS_ARG	1


static void sig_stop(int a)
{
	(void)a;
	live_ctx.sig_flag = 1;
}

typedef enum live_numeric_short_options_e {
	OPTION_MODEL_PATH,
	OPTION_LABEL_PATH,
	OPTION_OUTPUT_DIR,
	OPTION_LUA_FILE_PATH,
	OPTION_QUEUE_SIZE,
	OPTION_THREAD_NUM,
	OPTION_CLASS_NUM,
	OPTION_ADES_CMD_FILE,
	OPTION_ACINF_GPU_ID,
	OPTION_OVERLAY_BUF_OFFSET,
	OPTION_YUV,
	OPTION_MULTI_IN,
	OPTION_IN_ROI,
	OPTION_EXTRA_INPUT,
	OPTION_SUPPORT_LIST,
	OPTION_VOUT_ID,
	OPTION_FSYNC_OFF,
	OPTION_RESULT_TO_TXT,
	OPTION_HOLD_IMG,
} live_numeric_short_options_t;

#define INPUT_OPTIONS \
	{"canvas_id", HAS_ARG, 0, 'c'}, \
	{"pyramid_id", HAS_ARG, 0, 'p'}, \
	{"mode", HAS_ARG, 0, 'm'}, \
	{"isrc", HAS_ARG, 0, OPTION_MULTI_IN}, \
	{"yuv", NO_ARG, 0, OPTION_YUV}

#define PREPROCESS_OPTIONS \
	{"rgb", NO_ARG, 0, 'r'}, \
	{"roi", HAS_ARG, 0, OPTION_IN_ROI}

#define INFERENCE_OPTIONS \
	{"model_path", HAS_ARG, 0, OPTION_MODEL_PATH}, \
	{"ades_cmd_file", HAS_ARG, 0, OPTION_ADES_CMD_FILE}, \
	{"acinf_gpu_id", HAS_ARG, 0, OPTION_ACINF_GPU_ID}

#define POSTPROCESS_OPTIONS \
	{"nn_arm_name", HAS_ARG, 0, 'n'}, \
	{"queue_size", HAS_ARG, 0, OPTION_QUEUE_SIZE}, \
	{"lua_file", HAS_ARG, 0, OPTION_LUA_FILE_PATH}, \
	{"thread_num", HAS_ARG, 0, OPTION_THREAD_NUM}, \
	{"extra_input", HAS_ARG, 0, OPTION_EXTRA_INPUT}, \
	{"label_path", HAS_ARG, 0, OPTION_LABEL_PATH}, \
	{"class_num", HAS_ARG, 0, OPTION_CLASS_NUM}, \
	{"hold_img", NO_ARG, 0, OPTION_HOLD_IMG}

#define SHOW_RESULTS_OPTIONS \
	{"stream_id", HAS_ARG, 0, 's'}, \
	{"fsync_off", NO_ARG, 0, OPTION_FSYNC_OFF}, \
	{"stream_offset", HAS_ARG, 0, OPTION_OVERLAY_BUF_OFFSET}, \
	{"to_txt", HAS_ARG, 0, OPTION_RESULT_TO_TXT}, \
	{"vout_id", HAS_ARG, 0, OPTION_VOUT_ID}, \
	{"draw_mode", HAS_ARG, 0, 'd'}, \
	{"output_dir", HAS_ARG, 0, OPTION_OUTPUT_DIR}

#define MISCELLANEOUS_OPTIONS \
	{"support", NO_ARG, 0, OPTION_SUPPORT_LIST}, \
	{"log_level", HAS_ARG, 0, 'v'}, \
	{0, 0, 0, 0}

static struct option long_options[] = {
	INPUT_OPTIONS,
	PREPROCESS_OPTIONS,
	INFERENCE_OPTIONS,
	POSTPROCESS_OPTIONS,
	SHOW_RESULTS_OPTIONS,
	MISCELLANEOUS_OPTIONS,
};

static const char *short_options = "m:v:d:n:rs:c:p:";

struct hint_s {
	const char *arg;
	const char *str;
};

static const struct hint_s hint[] = {
	{"", "\t\tcanvas id. Default is 1"},
	{"", "\t\tpyramid id."},
	{"", "\t\trun mode"},
	{"", "\t\tmulti input, e.g. -isrc \"i:data=image|t:jpeg|c:rgb|r:0,0,0,0|d:cpu\". Only for file mode."},
	{"", "\t\tenable yuv input from iav, default is disable."},
	{"", "\t\tset color type to rgb_planar, default is bgr_planar. Only for live mode."},
	{"", "\t\troi of image, default is full image, order of roi parameters: x,y,h,w. Only for live mode."},
	{"", "\t\tpath of cavalry bin file."},
	{"", "\t\ades command file path. Run Ades if specified, otherwise run ACINF."},
	{"", "\tacinf gpu id, default is -1(CPU). Only for Acinference."},
	{"", "\tnn arm task name."},
	{"", "\t\tqueue size, default is 1."},
	{"", "\t\tlua file name."},
	{"", "\t\tthread number, default is 1."},
	{"", "\textra input for network."},
	{"", "\t\tpath of class names file."},
	{"", "\t\tclass number, for detection case should be set."},
	{"", "\t\tenable or disable to hold image data for postprocess, default is disable. Only for live mode."},
	{"", "\t\tstream ID to draw. Default is -1, means app doesn't use stream to draw."},
	{"", "\t\tenable or disable frame sync, default is enable."},
	{"", "\toverlay buffer offset for multi-stream display."},
	{"", "\t\tsave detection results to txt file. Only for file mode."},
	{"", "\t\tvout id for display, default is 1, means to use HDMI/CVBS."},
	{"", "\t\tdraw mode, 0=draw bbox, 1=draw img."},
	{"", "\t\tpath to contain output file."},
	{"", "\t\tshow support list."},
	{"", "\t\tlog level 0=None, 1=Error, 2=Notice, 3=Debug, 4=Verbose."},
};

static void sort(nn_task_name_t* array, int size)
{
	int j, i;
	char tmp[MAX_STR_LEN - 1];
	for (j = 0; j < (size - 1); j++) {
		for(i = 0; i < (size - j - 1); i++) {
			if(array[i].label[0] > array[i+1].label[0]) {
				strncpy(tmp, array[i].label, MAX_STR_LEN - 1);
				tmp[MAX_STR_LEN - 2] = '\0';
				strncpy(array[i].label, array[i+1].label, MAX_STR_LEN);
				array[i].label[MAX_STR_LEN - 1] = '\0';
				strncpy(array[i+1].label, tmp, MAX_STR_LEN - 1);
				array[i+1].label[MAX_STR_LEN - 1] = '\0';
			}
		}
	}
}

static void support_list()
{
	int nn_arm_task_num;
	nn_task_name_t *nn_task = NULL;
	int j;
	nn_arm_task_num = nn_arm_task_get_support_list(&nn_task);
	if (nn_arm_task_num > 1) {
		for (j = 0; j < nn_arm_task_num; j++) {
			if (strcmp(nn_task[j].label, TO_FILE_POSTPROCESS_NAME) == 0) {
				break;
			}
		}
		if (j != 0) {
			strncpy(nn_task[j].label, nn_task[0].label, MAX_STR_LEN);
			nn_task[j].label[MAX_STR_LEN - 1] = '\0';
			strncpy(nn_task[0].label, TO_FILE_POSTPROCESS_NAME, MAX_STR_LEN);
			nn_task[0].label[MAX_STR_LEN - 1] = '\0';
		}
	}
	if (nn_arm_task_num > 1) {
		sort(&nn_task[1], nn_arm_task_num - 1);
	}
	printf("support %d arm postprocess tasks:\n", nn_arm_task_num);
	for (j = 0; j < nn_arm_task_num; j++) {
		printf("\t%s\n", nn_task[j].label);
	}
	printf("\n");
	free(nn_task);
	nn_task = NULL;
}

static void usage(void)
{
	char mode[MAX_STR_LEN] = {0};
	nn_input_get_list(mode, MAX_STR_LEN);
	printf("test_eazyai usage:\n");
	for (size_t i = 0; i < sizeof(long_options) / sizeof(long_options[0]) - 1; i++) {
		if (isalpha(long_options[i].val)) {
			printf("-%c ", long_options[i].val);
		} else {
			printf("   ");
		}
		printf("--%s", long_options[i].name);
		if (hint[i].arg[0] != 0)
			printf(" [%s]", hint[i].arg);
		if (long_options[i].val == 'm') {
			printf("\t%s: %s.\n", hint[i].str, mode);
		} else {
			printf("\t%s\n", hint[i].str);
		}
	}
	support_list();
}



#ifdef __cplusplus
}
#endif


#endif
/*******************************************************************************
 * test_yolov8.h
 *
 ******************************************************************************/
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
#include "yolov8_struct.h"

class YoloV8_Class
{
    public:
        YoloV8_Class();
        // ~YoloV8_Class();

        int init_param(int argc, char **argv, 
                                live_params_t *params);

        int live_init(live_ctx_t *live_ctx, 
                            live_params_t *params);

        int live_run_loop(live_ctx_t *live_ctx, 
                                    live_params_t *params);

        void live_deinit(live_ctx_t *live_ctx, 
                                live_params_t *params);


	private:
        static int parse_param(int argc, char **argv, 
                                live_params_t *params);

        static int check_params(live_params_t *params);

        

        static void live_set_post_thread_params(live_params_t *params,
	                                            post_thread_params_t *post_params);
        
        
        static int cv_env_init(live_ctx_t *live_ctx, 
                                live_params_t *params);

        

        
        static void cv_env_deinit(live_ctx_t *live_ctx);


        

        static int live_update_net_output(live_ctx_t *live_ctx,
	                                    vp_output_t **vp_output);
        

        static int live_run_loop_dummy(live_ctx_t *live_ctx, 
                                        live_params_t *params);

        
        static int live_convert_yuv_data_to_bgr_data_for_postprocess(live_params_t *params, 
                                                                        img_set_t *img_set);

        static int live_run_loop_without_dummy(live_ctx_t *live_ctx, 
                                                live_params_t *params);

        
};
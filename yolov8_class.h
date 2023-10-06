/*******************************************************************************
 * test_yolov8.h
 *
 ******************************************************************************/
#include "yolov8_struct.h"
#include "yolov8_utils/object.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

class YoloV8_Class
{
    public:
        // live_params_t *params;
        // live_ctx_t *live_ctx;
        live_params_t *params;
        live_ctx_t *live_ctx;
        // static void sig_stop(int a)
        // {
        //         (void)a;
        //         live_ctx.sig_flag = 1;
        // };
        YoloV8_Class();

        ~YoloV8_Class();


        YoloV8_Class(int argc, 
                        char **argv);
                       
        YoloV8_Class(int argc, 
                        char **argv,
                        live_params_t *params,
                        live_ctx_t *live_ctx);

        int test_yolov8_init(int argc, 
                char **argv, 
                live_params_t *params,
                live_ctx_t *live_ctx);

        int init(int argc, 
                char **argv, 
                live_params_t *params,
                live_ctx_t *live_ctx);



        int init_param(int argc, 
                        char **argv, 
                        live_params_t *params);

        int live_init(live_ctx_t *live_ctx, 
                    live_params_t *params);


        int test_yolov8_run_2(live_ctx_t *live_ctx, 
                        live_params_t *params);

        int test_yolov8_run();


        Object test_yolov8_tracker(live_ctx_t *live_ctx, 
                        live_params_t *params);

        std::vector<BoundingBox> Get_yolov8_Bounding_Boxes(live_ctx_t *live_ctx, 
                        live_params_t *params,
                        std::vector<BoundingBox> bboxList);

        std::vector<BoundingBox> Get_Yolov8_Bounding_Boxes(std::vector<BoundingBox> bboxList);


        void Draw_Yolov8_Bounding_Boxes(std::vector<BoundingBox> bboxList,live_ctx_t *live_ctx, live_params_t *params);

        void Draw_Yolov8_Bounding_Boxes(std::vector<BoundingBox> bboxList);

        int live_run_loop(live_ctx_t *live_ctx, 
                        live_params_t *params);

        void live_deinit(live_ctx_t *live_ctx, 
                        live_params_t *params);

        void test_yolov8_deinit(live_ctx_t *live_ctx, 
                        live_params_t *params);

        void test_yolov8_deinit();


     private:
        int parse_param(int argc, char **argv, 
                        live_params_t *params);

        int check_params(live_params_t *params);
  
        void live_set_post_thread_params(live_params_t *params,
	                                post_thread_params_t *post_params);
        
        
        int cv_env_init(live_ctx_t *live_ctx, 
                        live_params_t *params);

        
        void cv_env_deinit(live_ctx_t *live_ctx);


        int live_update_net_output(live_ctx_t *live_ctx,
	                            vp_output_t **vp_output);
        
        int live_run_loop_dummy(live_ctx_t *live_ctx, 
                                live_params_t *params);

        int live_convert_yuv_data_to_bgr_data_for_postprocess(live_params_t *params, 
                                                            img_set_t *img_set);

        int live_run_loop_without_dummy(live_ctx_t *live_ctx, 
                                        live_params_t *params);

        
};
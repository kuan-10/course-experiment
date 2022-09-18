#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <math.h>

class Experiment2 {
public:
    Experiment2(std::vector<std::string> path){
        for(int i = 0; i < path.size(); i++){
            original_color_image.push_back(cv::imread(path[i]));
            original_gray_image.push_back(color2Gray(original_color_image[i]));
        }
    }
    // 0.1、彩色图像转灰度图像
    cv::Mat color2Gray(cv::Mat src_image){
        //创建与原图同类型和同大小的矩阵
	    cv::Mat gray_image(src_image.rows, src_image.cols, CV_8UC1);
        if(src_image.channels()!=1){
            for(int i = 0; i < src_image.rows; i++)
                for(int j = 0; j < src_image.cols; j++)
                    gray_image.at<uchar>(i, j) = (src_image.at<cv::Vec3b>(i, j)[0] + src_image.at<cv::Vec3b>(i, j)[1] + src_image.at<cv::Vec3b>(i, j)[2]) / 3;
        }
        else
            gray_image = src_image.clone();
        return gray_image;
    }
    // 2.1、计算灰度图像的归一化直方图
    void computerGrayHistogram(int pic_id, int opt=1){
        std::vector<float> hist(256,0);
        cv::Mat image = (opt > 0) ? original_gray_image[pic_id] : gray_image_equalization[pic_id];
        float nums = image.rows * image.cols;
        float max = 0;
        for(int i = 0; i < image.rows; i++)
            for(int j = 0; j < image.cols; j++)
                hist[image.at<uchar>(i, j)]++;
        for(int i=0; i < 256; i++){
            hist[i] = hist[i] / nums;
            if(hist[i] > max)
                max = hist[i];
        }
        if(opt)
            gray_hist.push_back(hist);
        else
            gray_hist_equalization.push_back(hist);
        drawGrayHist(max, pic_id, opt);
    }
    // 2.2、绘制归一化直方图 
    void drawGrayHist(float maxValue, int pic_id, int opt=1){
        int scale = 20000;
        int hist_h = scale * maxValue, hist_w = 257;
        std::vector<float> hist = (opt > 0) ? gray_hist[pic_id] : gray_hist_equalization[pic_id];
        // 画图底色
        cv::Mat image_hist = cv::Mat::zeros(hist_h, hist_w, CV_8U);
        // 画出直方图
        for (int i = 0; i < 256; i++){
            float binValue = hist[i];
            int realValue  = cv::saturate_cast<int>(binValue * scale);
            cv::rectangle(image_hist,
                          cv::Point(i, hist_h - realValue),
                          cv::Point(i + 1, hist_h),
                          cv::Scalar(255));
        }
        if(opt)
            gray_image_hist.push_back(image_hist);
        else
            gray_image_hist_equalization.push_back(image_hist);
    }
    // 3、灰度图像直方图均衡处理
    void grayEqualization(int pic_id){
        cv::Mat image_equalization = cv::Mat::zeros(original_gray_image[pic_id].size(), original_gray_image[pic_id].type());
        for(int i = 0; i < original_gray_image[pic_id].rows; i++)
            for(int j = 0; j < original_gray_image[pic_id].cols; j++){
                int k = original_gray_image[pic_id].at<uchar>(i, j);
                float sum = 0;
                for(int m = 0; m <= k; m++)
                    sum += gray_hist[pic_id][m];
                image_equalization.at<uchar>(i, j) = sum * 255 + 0.5;
            }
        gray_image_equalization.push_back(image_equalization);
    }
    // 4.1、彩色图像直方图均衡处理
    void colorEqualization(int pic_id){
        cv::Mat image_equalization = cv::Mat::zeros(original_color_image[pic_id].size(), original_color_image[pic_id].type());
        for(int i = 0; i < original_color_image[pic_id].rows; i++)
            for(int j = 0; j < original_color_image[pic_id].cols; j++)
                for(int n = 0; n < 3; n++){
                    int k = original_color_image[pic_id].at<cv::Vec3b>(i, j)[n];
                    float sum = 0;
                    for(int m = 0; m <= k; m++)
                        sum += color_hist[pic_id][n][m];
                    image_equalization.at<cv::Vec3b>(i, j)[n] = sum * 255 + 0.5;
                }
        color_image_equalization.push_back(image_equalization);
    }
    // 4.2、计算彩色图像的归一化直方图
    void computerColorHistogram(int pic_id){
        std::vector<std::vector<float> > hist(3, std::vector<float>(256)); 
        int nums = original_color_image[pic_id].rows * original_color_image[pic_id].cols;
        for(int i = 0; i < original_color_image[pic_id].rows; i++)
            for(int j = 0; j < original_color_image[pic_id].cols; j++){
                hist[0][original_color_image[pic_id].at<cv::Vec3b>(i, j)[0]]++;
                hist[1][original_color_image[pic_id].at<cv::Vec3b>(i, j)[1]]++;
                hist[2][original_color_image[pic_id].at<cv::Vec3b>(i, j)[2]]++;
            }
        for(int i=0; i < 256; i++){
            hist[0][i] = hist[0][i] / nums;
            hist[1][i] = hist[1][i] / nums;
            hist[2][i] = hist[2][i] / nums;
        }
        color_hist.push_back(hist);
    }
    // 5 显示 灰度均衡化 与 灰度直方图
    void test_GrayEqualizationAndHistogram(){
        for(int i = 0; i < original_gray_image.size(); i++){
            computerGrayHistogram(i);
            grayEqualization(i);
            std::cout<< "灰度图像: 当前已经均衡化第" << i + 1 << "张图像\n";
        }
        // 显示结果
        cv::imshow(" 原始灰度图像", original_gray_image[0]);
        cv::waitKey(0);
        for(int i = 0; i < original_gray_image.size(); i++){
            cv::imshow("原始图像的灰度直方图:", gray_image_hist[i]);
            cv::waitKey(0);
            cv::imshow("均衡化后的灰度图像:", gray_image_equalization[i]);
            cv::waitKey(0);
        }
        std::cout<< "\n";
    }
    // 6 显示 彩色均衡化 
    void test_ColorEqualizationAndHistogram(){
        for(int i = 0; i < original_color_image.size(); i++){
            computerColorHistogram(i);
            colorEqualization(i);
            std::cout<< "彩色图像: 当前已经均衡化第" << i + 1 << "张图像\n";
        }
        // 显示结果
        cv::imshow("原始图像", original_color_image[0]);
        cv::waitKey(0);
        for(int i = 0; i < original_color_image.size(); i++){
            cv::imshow("均衡化后---彩色图像:", color_image_equalization[i]);
            cv::waitKey(0);
        }
    }

private:
    std::vector<std::string> color_info;
    std::vector<std::string> gray_info;
    // 原始图像
    std::vector<cv::Mat> original_color_image;
    std::vector<cv::Mat> original_gray_image;
    // 均衡化后的图像
    std::vector<cv::Mat>  color_image_equalization;
    std::vector<cv::Mat>  gray_image_equalization;
    // 画出来的灰度直方图
    std::vector<cv::Mat> gray_image_hist;
    std::vector<std::vector<cv::Mat> > color_image_hist;
    std::vector<cv::Mat> gray_image_hist_equalization;
    std::vector<std::vector<cv::Mat> > color_image_hist_equalization;
    // 灰度图像的灰度分布 (图像名,每级灰度的个数)
    std::vector<std::vector<float> > gray_hist;
    std::vector<std::vector<float> > gray_hist_equalization;
    // 彩色图像的灰度分布(图像名,通道,每级灰度的个数)
    std::vector<std::vector<std::vector<float> > > color_hist;
    std::vector<std::vector<std::vector<float> > > color_hist_equalization;
};

int main(){
    std::vector<std::string> path;
    path.push_back("C:\\Users\\13906\\Desktop\\harden.webp");
    Experiment2 a(path);

    a.test_GrayEqualizationAndHistogram();
    a.test_ColorEqualizationAndHistogram();

    return 1;
}

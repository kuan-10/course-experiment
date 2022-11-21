#include <stdio.h>  
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/types_c.h>
#include <string>

class Experiment5 {
public:
    Experiment5(std::vector<std::string> path){
        filter_name.push_back("理想低通滤波器");
        filter_name.push_back("理想高通滤波器");
        filter_name.push_back("布特沃斯低通滤波器");
        filter_name.push_back("布特沃斯高通滤波器");

        for(int i = 0; i < path.size(); i++){
            original_color_image.push_back(cv::imread(path[i]));
            original_gray_image.push_back(color2Gray(original_color_image[i]));
            zeroPadding(i);
            gray_fourier.push_back(imageDFT(gray_image_padding[i]));
        }

        int row = gray_image_padding[0].rows, col = gray_image_padding[0].cols;
        makeIdealLowPassFilterKernel(row, col);
        makeIdealHighPassFilterKernel(row, col);
        makeButterworseLowPassFilterKernel(row, col);
        makeButterworseHighPassFilterKernel(row, col);
        moveFourier();
        std::cout<< "初始化结束" << "\n";
    }
    // 0 彩色图像转灰度图像
    cv::Mat color2Gray(cv::Mat& src){
        //创建与原图同类型和同大小的矩阵
	    cv::Mat gray_image = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
        if(src.channels()!=1){
            for(int i = 0; i < src.rows; i++)
                for(int j = 0; j < src.cols; j++)
                    gray_image.at<uchar>(i, j) = (src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2]) / 3;
        }
        else
            gray_image = src.clone();
        return gray_image;
    }
    // 0.1 零填充
    void zeroPadding(int pic_id){
        int centeri = original_gray_image[pic_id].rows, centerj = original_gray_image[pic_id].cols;
        cv::Mat image = cv::Mat::zeros(2*centeri, 2*centerj, CV_8UC1);
        image(cv::Rect(centeri/2, centerj/2, centerj, centeri)) += original_gray_image[pic_id];  
        gray_image_padding.push_back(image);
    }
    // 0.2 生成理想低通滤波器
    void makeIdealLowPassFilterKernel(int row, int col, double d0=80){
        int centeri =row/2, centerj = col/2;
        ideal_low_pass_filter = cv::Mat::zeros(row, col, CV_32F);
        for(int i=0; i < ideal_low_pass_filter.rows; i++)
            for(int j=0; j < ideal_low_pass_filter.cols; j++){
                double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
                if(d < d0)
                    ideal_low_pass_filter.at<float>(i,j) = 1;
            }
    }
    // 0.3 生成理想高通滤波器
    void makeIdealHighPassFilterKernel(int row, int col, double d0=12){
        int centeri =row/2, centerj = col/2;
        ideal_high_pass_filter = cv::Mat::zeros(row, col, CV_32F);
        for(int i=0; i < ideal_high_pass_filter.rows; i++)
            for(int j=0; j < ideal_high_pass_filter.cols; j++){
                double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
                if(d > d0)
                    ideal_high_pass_filter.at<float>(i,j) = 1;
            }
    }
    // 0.4 布特沃斯低通
    void makeButterworseLowPassFilterKernel(int row, int col, int n=2, double d0=80){
        int centeri =row/2, centerj = col/2;
        butter_low_pass_filter = cv::Mat::zeros(row, col, CV_32F);
        for(int i=0; i < butter_low_pass_filter.rows; i++)
            for(int j=0; j < butter_low_pass_filter.cols; j++){
                double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
                butter_low_pass_filter.at<float>(i,j) = 1/(1 + pow(d/d0,n));
            }
    }
    // 0.5 布特沃斯高通
    void makeButterworseHighPassFilterKernel(int row, int col, int n=2, double d0=10){
        int centeri =row/2, centerj = col/2;
        butter_high_pass_filter = cv::Mat::zeros(row, col, CV_32F);
        for(int i=0; i < butter_high_pass_filter.rows; i++)
            for(int j=0; j < butter_high_pass_filter.cols; j++){
                double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
                butter_high_pass_filter.at<float>(i,j) = 1/(1 + pow(d0/d,n));
            }
    }
    // 0.6 傅里叶变换
    cv::Mat imageDFT(cv::Mat& src_image){
        cv::Mat src, fourier;
        cv::Mat image = src_image;
        // 实部：图像 , 虚部：全部用0填充
        cv::Mat re_im[] = {cv::Mat_<float>(image), cv::Mat::zeros(image.size(), CV_32FC1)};
        // 将实部与虚部合并，形成一个复数
        cv::merge(re_im, 2, src);
        // 离散傅里叶变换
        cv::dft(src, fourier);
        return fourier;
    }
    // 0.7 移动傅里叶变换,将低频从四个角,移动到中心
    void moveFourier(){
        cv::Mat src = gray_image_padding[0];
        //int row = src.rows, col = src.cols;
        for(int i = 0; i < gray_fourier.size(); i++){
            cv::Mat fourier = gray_fourier[i];
            cv::Mat plane[]={cv::Mat_<float>(src), cv::Mat::zeros(src.size() , CV_32FC1)}; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
            cv::split(fourier, plane);
            cv::Mat tempu;
            // 获取未移动时的频谱
            cv::magnitude(plane[0],plane[1],tempu);
            tempu += cv::Scalar::all(1);
            cv::log(tempu, tempu);
            // 归一化操作
            cv::normalize(tempu, tempu, 1, 0, CV_MINMAX);
            gray_fourier_re.push_back(tempu);
            // 以下的操作是移动图像  (零频移到中心)
            shiftCenter(plane[0]);  // 实部
            shiftCenter(plane[1]);  // 虚部
            // 存入向量
            cv::Mat temp0 = plane[0].clone();
            cv::Mat temp1 = plane[1].clone();
            std::vector<cv::Mat> temp;
            temp.push_back(temp0);
            temp.push_back(temp1);
            gray_fourier_center.push_back(temp);
            // 获取原始图像的频谱图
            cv::magnitude(plane[0],plane[1],plane[0]);
            plane[0] += cv::Scalar::all(1);
            cv::log(plane[0],plane[0]);
            // 归一化操作便于显示
            cv::normalize(plane[0],plane[0],1,0,CV_MINMAX);
            gray_fourier_center_re.push_back(plane[0]);
        }
    }
    // 0.8 移动到中心
    void shiftCenter(cv::Mat& mat){
        int cx = mat.cols/2;
        int cy = mat.rows/2;
        //元素坐标表示为(cx,cy)
        cv::Mat part1(mat,cv::Rect(0,0,cx,cy));      
        cv::Mat part2(mat,cv::Rect(cx,0,cx,cy));
        cv::Mat part3(mat,cv::Rect(0,cy,cx,cy));
        cv::Mat part4(mat,cv::Rect(cx,cy,cx,cy));
        cv::Mat temp;
        // 位置交换
        part1.copyTo(temp); //左上与右下交换位置
        part4.copyTo(part1);
        temp.copyTo(part4);
        part2.copyTo(temp); //右上与左下交换位置
        part3.copyTo(part2);
        temp.copyTo(part3);
    }
    // 0.9 计算Mat类型的指数
    void MatPow(cv::Mat& src, double exp){
        for(int i = 0; i < src.rows; i++)
            for(int j = 0; j < src.cols; j++)
                src.at<float>(i, j) = pow(src.at<float>(i, j), exp);
    }
    // 2 频域滤波
    void frequencyDomainFilter(int select=0){
        cv::Mat filter_kernel;
        cv::Mat re, im;
        cv::Mat blur_r, blur_i, blur;
        for(int i = 0; i < gray_fourier_center.size(); i++){
            re = gray_fourier_center[i][0].clone();
            im = gray_fourier_center[i][1].clone();
            if(select == 0)
                filter_kernel = ideal_low_pass_filter;
            else if(select == 1)
                filter_kernel = ideal_high_pass_filter;
            else if(select == 2)
                filter_kernel = butter_low_pass_filter;
            else if(select == 3)
                filter_kernel = butter_high_pass_filter;
            // 相乘,滤波
            cv::multiply(re, filter_kernel, blur_r);
            cv::multiply(im, filter_kernel, blur_i);
            cv::Mat plane1[] = {blur_r, blur_i};
            // 实部与虚部合并
            cv::merge(plane1, 2, blur);
            imageIDFT(blur);
            moveImage(i);
        }
    }
    // 3 傅里叶逆变换
    cv::Mat imageIDFT(cv::Mat& fourier){
        cv::Mat invfourier;
        cv::idft(fourier, invfourier, 0);
        cv::Mat re_im[2];
        // 分离傅里叶变换的实部与虚部
        cv::split(invfourier, re_im);
        cv::normalize(re_im[0], re_im[0], 0, 1, CV_MINMAX); 
        gray_ifourier_center.push_back(re_im[0]);
        return re_im[0];
    }
    // 3.1 移动图像到中心位置
    void moveImage(int id){
        cv::Mat src_image = gray_ifourier_center[id];
        int row = original_gray_image[id].rows, col = original_gray_image[id].cols;
        cv::Mat dst_image = src_image(cv::Rect(row/2, col/2, col, row));
        gray_image_process.push_back(dst_image);
    }
    // 3.2 数据类型转换
    void Mat_convert2int(cv::Mat& src, cv::Mat& dst){
        double value;
        double max = findMatMax(src);
        for(int i = 0; i < src.rows; i++)
            for(int j = 0; j < src.cols; j++){
                value = 255 * src.at<float>(i, j) / max;
                if(value > 255)
                    value = 255;
                if(value < 0)
                    value = 0;
                dst.at<uchar>(i, j) = int(value);
            }
    }
    // 3.3 寻找矩阵最大值:
    double findMatMax(cv::Mat& src){
        double max = 0;
        for(int i = 0; i < src.rows; i++)
            for(int j = 0; j < src.cols; j++){
                if(src.at<float>(i, j) > max)
                    max = src.at<float>(i, j);
            }
        return max;
    }
    // 4 运行
    void test_filter(int id){
        frequencyDomainFilter(id);
        std::cout<<filter_name[id]<<": 处理完成"<< "\n";
    
        cv::imshow("原始图像:", original_gray_image[0]);
        cv::waitKey(0);
        cv::imshow("傅里叶频谱:", gray_fourier_re[0]);
        cv::waitKey(0);
        cv::imshow("中心化的傅里叶频谱:", gray_fourier_center_re[0]);
        cv::waitKey(0);
        cv::imshow(filter_name[id] + "-----处理之后:", gray_image_process[0]);
        cv::waitKey(0);
        gray_image_process.clear();
        gray_ifourier_center.clear();
        cv::destroyAllWindows();
    }

private:
    std::vector<std::string> filter_name;
    cv::Mat filter_kernel;

    std::vector<cv::Mat> original_color_image;
    std::vector<cv::Mat> original_gray_image;

    std::vector<cv::Mat> gray_image_padding;

    // 傅里叶变换
    std::vector<cv::Mat> gray_fourier;
    std::vector<cv::Mat> gray_fourier_re;
    // 中心化
    std::vector<std::vector<cv::Mat> > gray_fourier_center;
    std::vector<cv::Mat> gray_fourier_center_re;
    // 傅里叶逆变换
    std::vector<cv::Mat> gray_ifourier;
    // 中心化的傅里叶变换,直接逆变换:
    std::vector<cv::Mat> gray_ifourier_center;
    // 频域滤波后的图像
    std::vector<cv::Mat>  gray_image_process;
    // 滤波器在频域的表示:
    cv::Mat ideal_low_pass_filter;
    cv::Mat ideal_high_pass_filter;
    cv::Mat butter_low_pass_filter;
    cv::Mat butter_high_pass_filter;
};

int main(){
    std::vector<std::string> path;
    path.push_back("C:\\Users\\13906\\Desktop\\harden.webp");
    Experiment5 a(path);
    for(int i = 0; i < 4; i++)
        a.test_filter(i);
    return 1;
}
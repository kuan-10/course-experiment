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
        filter_name.push_back("�����ͨ�˲���");
        filter_name.push_back("�����ͨ�˲���");
        filter_name.push_back("������˹��ͨ�˲���");
        filter_name.push_back("������˹��ͨ�˲���");

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
        std::cout<< "��ʼ������" << "\n";
    }
    // 0 ��ɫͼ��ת�Ҷ�ͼ��
    cv::Mat color2Gray(cv::Mat& src){
        //������ԭͼͬ���ͺ�ͬ��С�ľ���
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
    // 0.1 �����
    void zeroPadding(int pic_id){
        int centeri = original_gray_image[pic_id].rows, centerj = original_gray_image[pic_id].cols;
        cv::Mat image = cv::Mat::zeros(2*centeri, 2*centerj, CV_8UC1);
        image(cv::Rect(centeri/2, centerj/2, centerj, centeri)) += original_gray_image[pic_id];  
        gray_image_padding.push_back(image);
    }
    // 0.2 ���������ͨ�˲���
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
    // 0.3 ���������ͨ�˲���
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
    // 0.4 ������˹��ͨ
    void makeButterworseLowPassFilterKernel(int row, int col, int n=2, double d0=80){
        int centeri =row/2, centerj = col/2;
        butter_low_pass_filter = cv::Mat::zeros(row, col, CV_32F);
        for(int i=0; i < butter_low_pass_filter.rows; i++)
            for(int j=0; j < butter_low_pass_filter.cols; j++){
                double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
                butter_low_pass_filter.at<float>(i,j) = 1/(1 + pow(d/d0,n));
            }
    }
    // 0.5 ������˹��ͨ
    void makeButterworseHighPassFilterKernel(int row, int col, int n=2, double d0=10){
        int centeri =row/2, centerj = col/2;
        butter_high_pass_filter = cv::Mat::zeros(row, col, CV_32F);
        for(int i=0; i < butter_high_pass_filter.rows; i++)
            for(int j=0; j < butter_high_pass_filter.cols; j++){
                double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
                butter_high_pass_filter.at<float>(i,j) = 1/(1 + pow(d0/d,n));
            }
    }
    // 0.6 ����Ҷ�任
    cv::Mat imageDFT(cv::Mat& src_image){
        cv::Mat src, fourier;
        cv::Mat image = src_image;
        // ʵ����ͼ�� , �鲿��ȫ����0���
        cv::Mat re_im[] = {cv::Mat_<float>(image), cv::Mat::zeros(image.size(), CV_32FC1)};
        // ��ʵ�����鲿�ϲ����γ�һ������
        cv::merge(re_im, 2, src);
        // ��ɢ����Ҷ�任
        cv::dft(src, fourier);
        return fourier;
    }
    // 0.7 �ƶ�����Ҷ�任,����Ƶ���ĸ���,�ƶ�������
    void moveFourier(){
        cv::Mat src = gray_image_padding[0];
        //int row = src.rows, col = src.cols;
        for(int i = 0; i < gray_fourier.size(); i++){
            cv::Mat fourier = gray_fourier[i];
            cv::Mat plane[]={cv::Mat_<float>(src), cv::Mat::zeros(src.size() , CV_32FC1)}; //����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
            cv::split(fourier, plane);
            cv::Mat tempu;
            // ��ȡδ�ƶ�ʱ��Ƶ��
            cv::magnitude(plane[0],plane[1],tempu);
            tempu += cv::Scalar::all(1);
            cv::log(tempu, tempu);
            // ��һ������
            cv::normalize(tempu, tempu, 1, 0, CV_MINMAX);
            gray_fourier_re.push_back(tempu);
            // ���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
            shiftCenter(plane[0]);  // ʵ��
            shiftCenter(plane[1]);  // �鲿
            // ��������
            cv::Mat temp0 = plane[0].clone();
            cv::Mat temp1 = plane[1].clone();
            std::vector<cv::Mat> temp;
            temp.push_back(temp0);
            temp.push_back(temp1);
            gray_fourier_center.push_back(temp);
            // ��ȡԭʼͼ���Ƶ��ͼ
            cv::magnitude(plane[0],plane[1],plane[0]);
            plane[0] += cv::Scalar::all(1);
            cv::log(plane[0],plane[0]);
            // ��һ������������ʾ
            cv::normalize(plane[0],plane[0],1,0,CV_MINMAX);
            gray_fourier_center_re.push_back(plane[0]);
        }
    }
    // 0.8 �ƶ�������
    void shiftCenter(cv::Mat& mat){
        int cx = mat.cols/2;
        int cy = mat.rows/2;
        //Ԫ�������ʾΪ(cx,cy)
        cv::Mat part1(mat,cv::Rect(0,0,cx,cy));      
        cv::Mat part2(mat,cv::Rect(cx,0,cx,cy));
        cv::Mat part3(mat,cv::Rect(0,cy,cx,cy));
        cv::Mat part4(mat,cv::Rect(cx,cy,cx,cy));
        cv::Mat temp;
        // λ�ý���
        part1.copyTo(temp); //���������½���λ��
        part4.copyTo(part1);
        temp.copyTo(part4);
        part2.copyTo(temp); //���������½���λ��
        part3.copyTo(part2);
        temp.copyTo(part3);
    }
    // 0.9 ����Mat���͵�ָ��
    void MatPow(cv::Mat& src, double exp){
        for(int i = 0; i < src.rows; i++)
            for(int j = 0; j < src.cols; j++)
                src.at<float>(i, j) = pow(src.at<float>(i, j), exp);
    }
    // 2 Ƶ���˲�
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
            // ���,�˲�
            cv::multiply(re, filter_kernel, blur_r);
            cv::multiply(im, filter_kernel, blur_i);
            cv::Mat plane1[] = {blur_r, blur_i};
            // ʵ�����鲿�ϲ�
            cv::merge(plane1, 2, blur);
            imageIDFT(blur);
            moveImage(i);
        }
    }
    // 3 ����Ҷ��任
    cv::Mat imageIDFT(cv::Mat& fourier){
        cv::Mat invfourier;
        cv::idft(fourier, invfourier, 0);
        cv::Mat re_im[2];
        // ���븵��Ҷ�任��ʵ�����鲿
        cv::split(invfourier, re_im);
        cv::normalize(re_im[0], re_im[0], 0, 1, CV_MINMAX); 
        gray_ifourier_center.push_back(re_im[0]);
        return re_im[0];
    }
    // 3.1 �ƶ�ͼ������λ��
    void moveImage(int id){
        cv::Mat src_image = gray_ifourier_center[id];
        int row = original_gray_image[id].rows, col = original_gray_image[id].cols;
        cv::Mat dst_image = src_image(cv::Rect(row/2, col/2, col, row));
        gray_image_process.push_back(dst_image);
    }
    // 3.2 ��������ת��
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
    // 3.3 Ѱ�Ҿ������ֵ:
    double findMatMax(cv::Mat& src){
        double max = 0;
        for(int i = 0; i < src.rows; i++)
            for(int j = 0; j < src.cols; j++){
                if(src.at<float>(i, j) > max)
                    max = src.at<float>(i, j);
            }
        return max;
    }
    // 4 ����
    void test_filter(int id){
        frequencyDomainFilter(id);
        std::cout<<filter_name[id]<<": �������"<< "\n";
    
        cv::imshow("ԭʼͼ��:", original_gray_image[0]);
        cv::waitKey(0);
        cv::imshow("����ҶƵ��:", gray_fourier_re[0]);
        cv::waitKey(0);
        cv::imshow("���Ļ��ĸ���ҶƵ��:", gray_fourier_center_re[0]);
        cv::waitKey(0);
        cv::imshow(filter_name[id] + "-----����֮��:", gray_image_process[0]);
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

    // ����Ҷ�任
    std::vector<cv::Mat> gray_fourier;
    std::vector<cv::Mat> gray_fourier_re;
    // ���Ļ�
    std::vector<std::vector<cv::Mat> > gray_fourier_center;
    std::vector<cv::Mat> gray_fourier_center_re;
    // ����Ҷ��任
    std::vector<cv::Mat> gray_ifourier;
    // ���Ļ��ĸ���Ҷ�任,ֱ����任:
    std::vector<cv::Mat> gray_ifourier_center;
    // Ƶ���˲����ͼ��
    std::vector<cv::Mat>  gray_image_process;
    // �˲�����Ƶ��ı�ʾ:
    cv::Mat ideal_low_pass_filter;
    cv::Mat ideal_high_pass_filter;
    cv::Mat butter_low_pass_filter;
    cv::Mat butter_high_pass_filter;
};

int main(){
    std::vector<std::string> path;
    path.push_back("C:\\Users\\13906\\Desktop\\frontier.png");
    Experiment5 a(path);
    for(int i = 0; i < 4; i++)
        a.test_filter(i);
    return 1;
}
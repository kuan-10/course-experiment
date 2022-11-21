#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include <string>
using namespace cv;
class Experiment1 {
public:
    // 0、彩色图像转灰度图像
    cv::Mat color2Gray(cv::Mat src_image){
        //创建与原图同类型和同大小的矩�?
	    cv::Mat gray_image=cv::Mat::zeros(src_image.rows, src_image.cols, CV_8UC1);
        if(src_image.channels()!=1){
            for(int i = 0; i < src_image.rows; i++)
                for(int j = 0; j < src_image.cols; j++)
                    gray_image.at<uchar>(i, j) = (src_image.at<cv::Vec3b>(i, j)[0] + src_image.at<cv::Vec3b>(i, j)[1] + src_image.at<cv::Vec3b>(i, j)[2]) / 3;
        }
        else
            gray_image = src_image.clone();
        return gray_image;
    }

    // 1、利�? OpenCV 读取图像
    cv::Mat readImage(std::string s = "C:\\Users\\13906\\Desktop\\harden.webp"){
        cv::Mat image = cv::imread(s);
        cv::namedWindow("Display Image");
        cv::imshow("Display Image", image);
        cv::waitKey(0);
        return image;
    }
    // 2、灰度图像二值化处理
    cv::Mat grayscaleBinarization(cv::Mat src_image, int threshold = 127){
        // 彩色图像转为灰度图像
        cv::Mat image = color2Gray(src_image);
        for(int i = 0; i < image.rows; i++){
            for(int j = 0; j < image.cols; j++){
                uchar pixel_value = image.at<uchar>(i, j);
                if(pixel_value > threshold)
                    image.at<uchar>(i, j) = 255;
                else
                    image.at<uchar>(i, j) = 0;          
            }
        }
        return image;
    }
    // 3、灰度图像的对数变换
    cv::Mat logarithmConversion(cv::Mat src_image, int c = 15){
        // 彩色图像转为灰度图像
        cv::Mat image = color2Gray(src_image);
        for(int i = 0; i < image.rows; i++)
            for(int j = 0; j < image.cols; j++)
                image.at<uchar>(i, j) = c * log(1 + image.at<uchar>(i, j));
        return image;
    }

    // 4、灰度图像的伽马变换（幂指数变换�?
    cv::Mat gammaConversion(cv::Mat src_image, int c = 1, double gamma = 2){
        // 彩色图像转为灰度图像
        cv::Mat image = color2Gray(src_image);
        for(int i = 0; i < image.rows; i++)
            for(int j = 0; j < image.cols; j++)
                image.at<uchar>(i, j) = c * pow(image.at<uchar>(i, j), gamma);
        return image;
    }

    // 5、彩色图像的反色变换
    cv::Mat antiConversion(cv::Mat src_image){
        cv::Mat image = src_image.clone();
        cv::Vec3b white = (255,255,255);
        for(int i = 0; i < image.rows; i++)
            for(int j = 0; j < image.cols; j++){
                //std::cout<<"before convert: "<<image.at<cv::Vec3b>(i, j)<<"\n";
                image.at<cv::Vec3b>(i, j)[0] = white[0] - image.at<cv::Vec3b>(i, j)[0];
                image.at<cv::Vec3b>(i, j)[1] = white[1] - image.at<cv::Vec3b>(i, j)[1];
                image.at<cv::Vec3b>(i, j)[2] = white[2] - image.at<cv::Vec3b>(i, j)[2];
                //std::cout<<"after convert: "<<image.at<cv::Vec3b>(i, j)<<"\n \n \n";
            }
        return image;
    }

    // 6 彩色图像的补色变�?
    cv::Mat complementConversion(cv::Mat src_image){
        cv::Mat image = src_image.clone();
        for(int i = 0; i < image.rows; i++)
            for(int j = 0; j < image.cols; j++){
                cv::Vec3b sum = (1,1,1);
                findmaxmin(image.at<cv::Vec3b>(i, j), sum);
                image.at<cv::Vec3b>(i, j) = sum - image.at<cv::Vec3b>(i, j);
            }
        return image;
    }
    // 7 找出极�?
    void findmaxmin(cv::Vec3b num,cv::Vec3b& sum){
        int num1 = num[0], num2 = num[0];
        for(int i = 0; i < 3; i++){
            if(num[i] > num1)
                num1 = num[i];
            if(num[i] < num2)
                num2 = num[i];
        }
        sum = (num1 + num2) * sum;
    }
    
};

int main(){
    std::string s = "C:\\Users\\13906\\Desktop\\harden.webp";
    Experiment1 image_convert;
    cv::Mat image1 = image_convert.readImage(s);
    cv::imshow("灰度图像", image_convert.color2Gray(image1));
    cv::waitKey(0);

    cv::Mat image2 = image_convert.grayscaleBinarization(image1);
    cv::imshow("二值化图像", image2);
    cv::waitKey(0);

    cv::Mat image3 = image_convert.logarithmConversion(image1);
    cv::imshow("��ɫ�任", image3);
    cv::waitKey(0);

    cv::Mat image4 = image_convert.gammaConversion(image1);
    cv::imshow("伽马变换", image4);
    cv::waitKey(0);

    cv::Mat image5 = image_convert.antiConversion(image1);
    cv::imshow("反色变换", image5);
    cv::waitKey(0);

    cv::Mat image6 = image_convert.complementConversion(image1);
    cv::imshow("补色变换", image6);
    cv::waitKey(0);

    return 1;
}
// void showPic(){
//      Mat img=imread("C:\\Users\\13906\\Desktop\\harden.webp");
//     cv::imshow("image",img);
//     cv::waitKey();
// }
//  Mat handleBinary(int gray, int max = 255,Mat img) {
//         Mat result;
//         //灰度处理，方式为二值化（THRESH_BINARY�?
//         threshold(img, result, gray, max, THRESH_BINARY);
//         return result;
//     }
// int main()
// {
//    // showPic();
//    handleBinary()
//    return 0;
// }


#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <math.h>

class Experiment3 {
public:
    // 初始化：彩色图片,灰度图片,模板(高提升滤波的模板除外)
    Experiment3(std::vector<std::string> path){
        filter_name.push_back("均值模板平滑");
        filter_name.push_back("高斯模板平滑");
        filter_name.push_back("Laplacian模板锐化");
        filter_name.push_back("Robert模板锐化");
        filter_name.push_back("Sobel模板锐化");
        filter_name.push_back("高提升滤波算法增强");

        pic_color.push_back("灰度");
        pic_color.push_back("彩色");

        filter_size_string.push_back("3 x 3");
        filter_size_string.push_back("5 x 5");
        filter_size_string.push_back("9 x 9");

        for(int i = 0; i < path.size(); i++){
            original_color_image.push_back(cv::imread(path[i]));
            original_gray_image.push_back(color2Gray(original_color_image[i]));
        }

        template_size = 3;
        int step = 2;
        for(int i = 0; i < 3; i++){
            filter_size.push_back(template_size);
            makemeanTemplate(template_size);
            makeGaussTemplate(template_size);
            template_size += step;
            step +=2;
        }
        makeLaplacianTemplate();
        makeRobertTemplate();
        makeSobelTemplate();
        computerGaussTemplateSum();
    }
    // 彩色图像转灰度图像
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
    //  生成均值模板
    void makemeanTemplate(int size){
        mean_template.push_back(cv::Mat::ones(size, size, CV_8UC1));
    }
    //  生成高斯模板
    void makeGaussTemplate(int size=3, int sigma=1){
        cv::Mat gaussTemplate = cv::Mat::zeros(size, size, CV_32F);
        
        int center=size/2;
        double min = g(center,center);
        for(int i=0; i < size; i++)
            for(int j=0; j < size; j++)
                gaussTemplate.at<float>(i, j) = g(i-center,j-center)/min;
        
        gaussTemplate.convertTo(gaussTemplate, CV_8U);
        gauss_template.push_back(gaussTemplate);
    }
    // 计算正态分布
    double g(double x, double y, double sigma=1){
        return exp(-(x*x + y*y)/(2*sigma*sigma));
    }
    //  计算高斯模板的和
    void computerGaussTemplateSum(){
        for(int k=0; k < 3; k++){
            int sum = 0;
            for(int i=0; i < gauss_template[k].rows; i++)
                for(int j=0; j < gauss_template[k].cols; j++)
                    sum += gauss_template[k].at<uchar>(i, j);
            gauss_template_sum.push_back(sum);
        }
    }
   
    //生成Laplacian
    void makeLaplacianTemplate(){
        laplacian_template = (cv::Mat_<float>(3,3) << 0,  1, 0, 
                                                      1, -4, 1, 
                                                      0,  1, 0);
    }
    
    //生成Robert
    void makeRobertTemplate(){
        robert_template.push_back((cv::Mat_<float>(3,3) << 0,  0,  0, 
                                                           0, -1,  0, 
                                                           0,  0,  1));

        robert_template.push_back((cv::Mat_<float>(3,3) << 0,  0,   0, 
                                                           0,  0,  -1, 
                                                           0,  1,   0));
    }
    // 生成Sobel模板
    //生成Sobel模板
    void makeSobelTemplate(){
        sobel_template.push_back((cv::Mat_<float>(3,3) <<  -1,   0,   1, 
                                                           -2,   0,   2, 
                                                           -1,   0,   1));

        sobel_template.push_back((cv::Mat_<float>(3,3) <<  -1,  -2,  -1, 
                                                            0,   0,   0, 
                                                            1,   2,   1));
    }
    // 生成灰度图像mask
    void makeGrayHighPromoteMask(int id){
        highPromote_gray_mask.push_back(original_gray_image[id] - gray_image_process[id * 5]);
    }
    // 生成彩色图像mask
    void makeColorHighPromoteMask(int id){
        highPromote_color_mask.push_back(original_color_image[id] - color_image_process[id * 5]);
    }
    // 灰度空域滤波
    void graySpatialFiltering(int pic_id, int size_id=0, int select=0){
        int size = filter_size[size_id];
        int m = size/2;
        
        cv::Mat image_process = cv::Mat::zeros(original_gray_image[pic_id].size(), original_gray_image[pic_id].type());
        cv::Mat image_grad = cv::Mat::zeros(original_gray_image[pic_id].size(), original_gray_image[pic_id].type());
        for(int i=m; i < original_gray_image[pic_id].rows - m; i++)
            for(int j=m; j < original_gray_image[pic_id].cols - m; j++){ 
                cv::Mat sub_matrix = original_gray_image[pic_id](cv::Rect(j - m, i - m, size, size));
                int grad, value;   
                if(select == 0)
                    image_process.at<uchar>(i, j) = computerMeanResult(sub_matrix, size_id);//计算灰度均值
                else if(select == 1)
                    image_process.at<uchar>(i, j) = computerGaussResult(sub_matrix, size_id);//计算高斯均值滤波
                else if(select == 2)
                    image_grad.at<uchar>(i, j) = computerLaplacianResult(sub_matrix);//计算Laplacian滤波
                else if(select == 3)
                    image_grad.at<uchar>(i, j) = computerRobertResult(sub_matrix);//计算Robert滤波
                else if(select == 4)
                    image_grad.at<uchar>(i, j) = computerSobelResult(sub_matrix);//计算Sobel滤波
                else if(select == 5)
                    grayHighPromote(pic_id);//计算高提升滤波
                
                if(select == 2 || select == 3 || select == 4){
                    grad = image_grad.at<uchar>(i, j);
                    value = sub_matrix.at<uchar>(m, m);
                    if(grad + value > 255)
                        image_process.at<uchar>(i, j) = 255;
                    else
                        image_process.at<uchar>(i, j) = grad + value;
                }
            }
        gray_image_process.push_back(image_process);
        if(select == 2)
            gray_image_grad_laplacian.push_back(image_grad);
        else if(select == 3)
            gray_image_grad_robert.push_back(image_grad);
        else if(select == 4)
            gray_image_grad_sobel.push_back(image_grad);
    }
    // 计算均值滤波
    int computerMeanResult(cv::Mat& image_block, int size_id){
        int sum = filter_size[size_id] * filter_size[size_id];
        return image_block.dot(mean_template[size_id])/sum;
    }
    // 计算高斯滤波
    int computerGaussResult(cv::Mat& image_block, int size_id){
        int sum = gauss_template_sum[size_id];
        return image_block.dot(gauss_template[size_id])/sum;
    }
    //计算Laplacian滤波
    int computerLaplacianResult(cv::Mat& image_block){
        float g = 0.0;
        for(int i=0; i < image_block.rows; i++)
            for(int j=0; j < image_block.cols; j++)
                g += float(image_block.at<uchar>(i, j)) * laplacian_template.at<float>(i, j);
        if(abs(g) > 255)
            return 255;
        else if(g < 0)
            return -g;
        else
            return g;
    }
    // 5 计算Robert滤波
    //计算Robert滤波
    int computerRobertResult(cv::Mat& image_block){
        float Gx = float(image_block.at<uchar>(2,2)) - float(image_block.at<uchar>(1,1));
        float Gy = float(image_block.at<uchar>(2,1)) - float(image_block.at<uchar>(1,2));
        float g =  abs(Gx) + abs(Gy);
        if(g > 255)
            return 255;
        if(g < 0)
            return 0;
        return g;
    }
    //  计算Sobel滤波
    //计算Sobel滤波
    int computerSobelResult(cv::Mat& image_block){
        float Gx = 0.0;
        float Gy = 0.0;
        for(int i=0; i < image_block.rows; i++)
            for(int j=0; j < image_block.cols; j++){
                Gx += float(image_block.at<uchar>(i, j)) * sobel_template[0].at<float>(i, j);
                Gy += float(image_block.at<uchar>(i, j)) * sobel_template[1].at<float>(i, j);
            }
        float g =  abs(Gx) + abs(Gy);
        if(g > 255)
            return 255;
        if(g < 0)
            return 0;
        return g;
    }
    //利用高提升滤波算法增强灰度图像
    void grayHighPromote(int id, double k = 2){
       gray_image_process.push_back(original_gray_image[id] + k * highPromote_gray_mask[id]);
    }
    // 利用高提升滤波算法增彩色度图像
    void colorHighPromote(int id, double k = 2){
       color_image_process.push_back(original_color_image[id] + k * highPromote_color_mask[id]);
    }
    // 彩色图像滤波
    void colorSpatialFiltering(int pic_id, int size_id=0, int select=0){
        int size = filter_size[size_id];
        int m = size/2;
        
        cv::Mat image_process = cv::Mat::zeros(original_color_image[pic_id].size(), original_color_image[pic_id].type());
        cv::Mat image_grad = cv::Mat::zeros(original_color_image[pic_id].size(), original_color_image[pic_id].type());
        std::vector<cv::Mat> channels;
        cv::split(original_color_image[pic_id], channels);

        for(int i=m; i < original_color_image[pic_id].rows - m; i++)
            for(int j=m; j < original_color_image[pic_id].cols - m; j++){ 
                for(int k=0; k < 3; k++){
                    cv::Mat sub_matrix = channels[k](cv::Rect(j - m, i - m, size, size));
                    int grad, value;   
                    if(select == 0)
                        image_process.at<cv::Vec3b>(i, j)[k] = computerMeanResult(sub_matrix, size_id);
                    else if(select == 1)
                        image_process.at<cv::Vec3b>(i, j)[k] = computerGaussResult(sub_matrix, size_id);
                    else if(select == 2)
                        image_grad.at<cv::Vec3b>(i, j)[k] = computerLaplacianResult(sub_matrix);
                    else if(select == 3)
                        image_grad.at<cv::Vec3b>(i, j)[k] = computerRobertResult(sub_matrix);
                    else if(select == 4)
                        image_grad.at<cv::Vec3b>(i, j)[k] = computerSobelResult(sub_matrix);
                    else if(select == 5)
                        colorHighPromote(pic_id);
                    
                    if(select == 2 || select == 3 || select == 4){
                        grad = image_grad.at<cv::Vec3b>(i, j)[k];
                        value = sub_matrix.at<uchar>(m, m);
                        if(grad + value > 255)
                            image_process.at<cv::Vec3b>(i, j)[k] = 255;
                        else
                            image_process.at<cv::Vec3b>(i, j)[k] = grad + value;
                    }
                }
            }
        color_image_process.push_back(image_process);
        if(select == 2)
            color_image_grad_laplacian.push_back(image_grad);
        else if(select == 3)
            color_image_grad_robert.push_back(image_grad);
        else if(select == 4)
            color_image_grad_sobel.push_back(image_grad);
    }
    //  测试灰度均值\高斯滤波器
    void test_MeanAndGaussGrayFilter(int filter_id){
        for(int i = 0; i < original_gray_image.size(); i++)
            for(int j = 0; j < filter_size.size(); j++){
                graySpatialFiltering(i, j, filter_id);
                std::cout<< pic_color[0] <<"-----"<<filter_name[filter_id]<<"-----尺寸大小: "<< filter_size[j] <<": 运行完毕!!!!  \n";
            }

        cv::imshow(filter_name[filter_id] + "----准备就绪,可以开始!", original_color_image[0]);
        cv::waitKey(0);

        //  显示灰度滤波结果
        int k = filter_id * original_gray_image.size() * filter_size.size();
        std::cout<<"gray_image_process nums: "<<gray_image_process.size()<<"\n";
        //int k = 0;
        for(int i = 0; i < original_gray_image.size(); i++){
            cv::imshow("原始图像", original_gray_image[i]);
            cv::waitKey(0);
            for(int j = 0; j < filter_size.size(); j++){
                cv::imshow(pic_color[0] + "-----" + filter_name[filter_id] + "-----" + filter_size_string[j], gray_image_process[k++]);
                cv::waitKey(0);
            }
        }
        std::cout<<"\n \n";
        cv::destroyAllWindows();
    }
    //  测试灰度Laplacian\Robert\Sobel滤波器
    void test_LaplacianRobertSobelGrayFilter(int filter_id){
        for(int i = 0; i < original_gray_image.size(); i++){
            graySpatialFiltering(i, 0, filter_id);
            std::cout<< pic_color[0] <<"-----"<<filter_name[filter_id]<<": 运行完毕!!!!  \n";
        }

        cv::imshow(filter_name[filter_id] + "----准备就绪,可以开始!", original_color_image[0]);
        cv::waitKey(0);

        //  显示灰度滤波结果
        int num_pre_pic_mean_gauss = original_gray_image.size() * filter_size.size() * 2;
        int k = num_pre_pic_mean_gauss + filter_id - 2;
        std::cout<<"gray_image_process nums: "<<gray_image_process.size()<<"\n";
        cv::imshow("原始图像", original_gray_image[0]);
        cv::waitKey(0);

        for(int i = 0; i < original_gray_image.size(); i++){
            cv::imshow(pic_color[0] + "-----" + filter_name[filter_id], gray_image_process[k++]);
            cv::waitKey(0);
            if(filter_id == 2)
                cv::imshow(pic_color[0] + "-----" + filter_name[filter_id] + "-----" + "灰度幅值: ", gray_image_grad_laplacian[i]);
            else if(filter_id == 3)
                cv::imshow(pic_color[0] + "-----" + filter_name[filter_id] + "-----" + "灰度幅值: ", gray_image_grad_robert[i]);
            else if(filter_id == 4)
                cv::imshow(pic_color[0] + "-----" + filter_name[filter_id] + "-----" + "灰度幅值: ", gray_image_grad_sobel[i]);
            cv::waitKey(0);
        }
        std::cout<<"\n \n";
        cv::destroyAllWindows();
    }
    //  测试灰度高提升滤波器
    void test_HighPromoteGrayFilter(){
        int k = 9;
        makeGrayHighPromoteMask(0);
        grayHighPromote(0);
        cv::imshow(pic_color[0] + "-----" + filter_name[5], gray_image_process[9]);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    //  测试彩色均值\高斯滤波器
    void test_MeanAndGaussColorFilter(int filter_id){
        for(int i = 0; i < original_color_image.size(); i++)
            for(int j = 0; j < filter_size.size(); j++){
                colorSpatialFiltering(i, j, filter_id);
                std::cout<< pic_color[1] <<"-----"<<filter_name[filter_id]<<"-----尺寸大小: "<< filter_size[j] <<": 运行完毕!!!!  \n";
            }

        cv::imshow(filter_name[filter_id] + "----准备就绪,可以开始!", original_color_image[0]);
        cv::waitKey(0);

        //  显示灰度滤波结果
        int k = filter_id * original_color_image.size() * filter_size.size();
        std::cout<<"color_image_process nums: "<<color_image_process.size()<<"\n";
        //int k = 0;
        for(int i = 0; i < original_color_image.size(); i++){
            cv::imshow("原始图像", original_color_image[i]);
            cv::waitKey(0);
            for(int j = 0; j < filter_size.size(); j++){
                cv::imshow(pic_color[1] + "-----" + filter_name[filter_id] + "-----" + filter_size_string[j], color_image_process[k++]);
                cv::waitKey(0);
            }
        }
        std::cout<<"\n \n";
        cv::destroyAllWindows();
    }
    //  测试彩色Laplacian\Robert\Sobel滤波器
    void test_LaplacianRobertSobelColorFilter(int filter_id){
        for(int i = 0; i < original_color_image.size(); i++){
            colorSpatialFiltering(i, 0, filter_id);
            std::cout<< pic_color[1] <<"-----"<<filter_name[filter_id]<<": 运行完毕!!!!  \n";
        }

        cv::imshow(filter_name[filter_id] + "----准备就绪,可以开始!", original_color_image[0]);
        cv::waitKey(0);

        //  显示灰度滤波结果
        int num_pre_pic_mean_gauss = original_color_image.size() * filter_size.size() * 2;
        int k = num_pre_pic_mean_gauss + filter_id - 2;
        std::cout<<"color_image_process nums: "<<color_image_process.size()<<"\n";
        cv::imshow("原始图像", original_color_image[0]);
        cv::waitKey(0);

        for(int i = 0; i < original_color_image.size(); i++){
            cv::imshow(pic_color[1] + "-----" + filter_name[filter_id], color_image_process[k++]);
            cv::waitKey(0);
            if(filter_id == 2)
                cv::imshow(pic_color[1] + "-----" + filter_name[filter_id] + "-----" + "灰度幅值: ", color_image_grad_laplacian[i]);
            else if(filter_id == 3)
                cv::imshow(pic_color[1] + "-----" + filter_name[filter_id] + "-----" + "灰度幅值: ", color_image_grad_robert[i]);
            else if(filter_id == 4)
                cv::imshow(pic_color[1] + "-----" + filter_name[filter_id] + "-----" + "灰度幅值: ", color_image_grad_sobel[i]);
            cv::waitKey(0);
        }
        std::cout<<"\n \n";
        cv::destroyAllWindows();
    }
    //  测试彩色高提升滤波器
    void test_HighPromoteColorFilter(){
        int k = 9;
        makeColorHighPromoteMask(0);
        colorHighPromote(0);
        cv::imshow(pic_color[0] + "-----" + filter_name[5], color_image_process[9]);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

private:
    std::vector<std::string> filter_name;
    std::vector<std::string> pic_color;

    std::vector<cv::Mat> original_color_image;
    std::vector<cv::Mat> original_gray_image;

    std::vector<cv::Mat>  color_image_process;
    std::vector<cv::Mat>  gray_image_process;

    // 梯度幅值图像
    std::vector<cv::Mat>  color_image_grad_robert;
    std::vector<cv::Mat>  gray_image_grad_robert;

    std::vector<cv::Mat>  color_image_grad_sobel;
    std::vector<cv::Mat>  gray_image_grad_sobel;

    std::vector<cv::Mat>  color_image_grad_laplacian;
    std::vector<cv::Mat>  gray_image_grad_laplacian;

    // 模板尺寸:3*3,5*5,9*9
    std::vector<cv::Mat> mean_template;
    std::vector<cv::Mat> gauss_template;
    std::vector<int> gauss_template_sum;

    // 以下模板尺寸固定
    cv::Mat laplacian_template;
    std::vector<cv::Mat> robert_template;
    std::vector<cv::Mat> sobel_template;

    // 不同图片的 mask
    std::vector<cv::Mat> highPromote_gray_mask;
    std::vector<cv::Mat> highPromote_color_mask;

    std::vector<int> filter_size;
    std::vector<std::string> filter_size_string;
    int template_size;
};
   
int main(){
    std::vector<std::string> path;
    path.push_back("C:\\Users\\13906\\Desktop\\harden.webp");
    Experiment3 a(path);

    for(int i=0; i < 2; i++)
        a.test_MeanAndGaussGrayFilter(i);//测试均值和高斯中值滤波器

    for(int i=2; i < 5; i++)
        a.test_LaplacianRobertSobelGrayFilter(i);//测试LaplacianRobertSobel滤波器
    a.test_HighPromoteGrayFilter();//测试高提升滤波器

    for(int i=0; i < 2; i++)
        a.test_MeanAndGaussColorFilter(i);//彩色图像

    for(int i=2; i < 5; i++)
        a.test_LaplacianRobertSobelColorFilter(i);//彩色图像
    a.test_HighPromoteColorFilter();//彩色图像
    return 1;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

const static std::string path = "C:\\Users\\13906\\Desktop\\";

//图片路径组合
string getFullPath(string name) {
    return path + name;
}

//打开图片显示窗口
void openWindows(string win_name, Mat img, int x = 500, int y = 200) {
    //窗口命名，指定大小，生成位置
    namedWindow(win_name, WINDOW_AUTOSIZE);
    moveWindow(win_name, x, y);
    //生成窗口显示图片
    imshow(win_name, img);
    //等待键入
    waitKey();
    //关闭窗口
    destroyWindow(win_name);
}

//统一数字输入函数
template<typename T>T inputNumber(string desc) {
    system("cls");
    T input;
    cout << desc;
    cin >> input;
    cout << endl;
    return input;
}

//统一图片打开函数，用于简化路径和处理打开图片错误
Mat openImage(string name, int type = 1) {
    //图片读取函数，返回图像存储类（包含存储方式、存储矩阵、矩阵大小等）
    Mat img = imread(getFullPath(name), type);
    if (img.empty()) {
        cout << "无效图片，读取失败" << endl;
        exit(-1);
    }
    return img;
}

//图像基类
class Image {
protected:
    Mat img;
public:
    Mat getImage() {
        return img.clone();
    }
};

//彩色图像处理类
class ColorImage :public Image {
public:
    ColorImage(Mat img,string name) {
        this->img = img.clone();
        openWindows(name, img);
    }
    //读取彩色图像并展示
    ColorImage(string path) {
        img = openImage(path, IMREAD_COLOR);
        openWindows("彩色图像", img);
    }
};

//灰度图像处理类
class GrayImage :public Image {
public:
    GrayImage(Mat img,string name) {
        this->img = img.clone();
        openWindows(name, img);
    }
    //仅读取灰度方式读取图像并展示（IMREAD_GRAYSCALE）
    GrayImage(string path) {
        img = openImage(path, IMREAD_GRAYSCALE);
        openWindows("灰度图像", img);
    }
};

//添加噪声处理类
class Noise {
    Mat src;
private:
    //防止图像溢出的图像加减处理，使用saturate_cast进行限制
    Mat handleImageAddition(Mat img, Mat add, bool ifadd = true) {
        Mat target = img.clone();
        vector<Mat> target_channels;
        vector<Mat> add_channels;
        split(target, target_channels);
        split(add, add_channels);
        for (int c = 0; c < target_channels.size(); c++) {
            Mat curr = target_channels[c];
            Mat cadd = add_channels[c];
            for (int i = 0; i < curr.rows; i++)
                for (int j = 0; j < curr.cols; j++)
                    curr.at<uchar>(i, j) = saturate_cast<uchar>(ifadd ? curr.at<uchar>(i, j) + cvRound(cadd.at<uchar>(i, j)) : curr.at<uchar>(i, j) - cvRound(cadd.at<uchar>(i, j)));
        }
        merge(target_channels, target);
        return target;
    }

    //添加指定数值噪声(椒：0，盐：255)
    Mat saltAndPepperNoise(const Mat& input, int n, int value) {
        int x, y;
        int row = input.rows, col = input.cols;
        bool color = input.channels() > 1;
        Mat target = input.clone();
        //随机取位置添加指定噪声
        for (int i = 0; i < n; i++) {
            x = rand() % row;
            y = rand() % col;
            if (color) {
                target.at<Vec3b>(x, y)[0] = value;
                target.at<Vec3b>(x, y)[1] = value;
                target.at<Vec3b>(x, y)[2] = value;
            }
            else {
                target.at<uchar>(x, y) = value;
            }
        }
        return target;
    }

    //获取随机高斯噪声图像
    Mat gaussianNoise(const Mat& input, int avg, int sd) {
        Mat target = Mat::zeros(input.size(), input.type());
        //Opencv随机数类
        RNG rng(rand());
        //高斯分布的平均数或均匀分布的最小值
        int a = avg;
        //高斯分布的标准差或均匀分布的最大值
        int b = sd;
        //类型，NORMAL为高斯分布，UNIFORM为均匀分布
        int distType = RNG::NORMAL;
        //产生一个符合高斯分布（或均匀分布）的图像
        rng.fill(target, distType, a, b);
        return target;
    }

public:
    Noise(Image img) {
        this->src = img.getImage();
    }

    //添加胡椒噪声
    Image addPepperNoise(int n) {
        Mat output = saltAndPepperNoise(src, n, 0);
        string name = "胡椒噪声";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }

    //添加盐噪声
    Image addSaltNoise(int n) {
        Mat output = saltAndPepperNoise(src, n, 255);
        string name = "盐噪声";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }

    //添加椒盐噪声
    Image addSaltAndPepperNoise(int n) {
        Mat output = saltAndPepperNoise(saltAndPepperNoise(src, n / 2, 0), n / 2, 255);
        string name = "椒盐噪声";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }

    //添加高斯分布
    Image addGaussianNoise(int avg = 15, int sd = 30) {
        //高斯噪声图像
        Mat noise = gaussianNoise(src, avg, sd);
        //噪声图像与原图像叠加
        Mat output = handleImageAddition(src, noise);
        string name = "高斯噪声";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }
};

class Filter {
protected:
    Mat img;
    bool color;

    //对图像进行滤波处理
    Mat filterProcess(const Mat& img, uchar kernel(const Mat&, int, int, int, int), int size, int q = 1) {
        Mat target = img.clone();
        vector<Mat> channels;
        int m = size / 2;
        split(target, channels);
        for (Mat ch : channels) {
            Mat src = ch.clone();
            for (int i = src.rows - size; i >= 0; i--)
                for (int j = src.cols - size; j >= 0; j--) {
                    ch.at<uchar>(i + m, j + m) = kernel(src, size, i, j, q);
                }
        }
        merge(channels, target);
        return target;
    }

public:
    Filter(Image img) {
        this->img = img.getImage();
        this->color = this->img.channels() != 1;
    }    

};

class MeanFilter :public Filter {
private:
    //算术均值计算
    static uchar arithmeticMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res = 0;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res += img.at<uchar>(r, c);
            }
        }
        return (uchar)(res / ((double)size * size));
    }
    //几何均值计算
    static uchar geometricMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res = 1;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res *= img.at<uchar>(r, c);
            }
        }
        return (uchar)(pow(res, 1.0 / ((double)size * size)));
    }
    //谐波平均值计算
    static uchar harmonicMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res = 0;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res += 1.0 / ((double)img.at<uchar>(r, c) + 1);
            }
        }
        return (uchar)(((double)size * size) / res - 1);
    }
    //反谐波平均值计算
    static uchar antiHarmonicMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res1 = 0, res2 = 0;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res1 += pow(img.at<uchar>(r, c), q + 1);
                res2 += pow(img.at<uchar>(r, c), q);
            }
        }
        return (uchar)(res1 / res2);
    }

public:
    MeanFilter(Image img) :Filter(img) {

    }
    //算术平均滤波
    void handleArithmeticMean(int size, string type = "") {
        string name = type + "算术平均";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, arithmeticMeanKernel, size));
    }
    //几何平均滤波
    void handleGeometricMean(int size, string type = "") {
        string name = type + "几何平均";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, geometricMeanKernel, size));
    }
    //谐波平均滤波
    void handleHarmonicMean(int size, string type = "") {
        string name = type + "谐波平均";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, harmonicMeanKernel, size));
    }
    //反谐波平均滤波
    void handleAntiHrithmeticMean(int size, int q, string type = "", bool  twice = false) {
        string name = type + "反谐波平均";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        Mat target = filterProcess(img, antiHarmonicMeanKernel, size, q);
        if (twice)
            target = filterProcess(target, antiHarmonicMeanKernel, size, -q);
        openWindows(name, target);
    }

    static void show(string type, Image img, int size, int q = 1, bool twice = false) {
        MeanFilter mf(img);
        mf.handleArithmeticMean(size, type);
        mf.handleGeometricMean(size, type);
        mf.handleHarmonicMean(size, type);
        mf.handleAntiHrithmeticMean(size, q, type, twice);
    }
};

class MedianFilter :public Filter {
protected:
    //快速选择中值
    static int quickSelect(vector<uchar>& vec, int left, int right, int target) {
        int t = vec[left], l = left, r = right;
        bool isLeft = true;
        while (l < r) {
            if (isLeft) {
                while (l < r && vec[r] >= t)
                    r--;
                vec[l] = vec[r];
            }
            else {
                while (l < r && vec[l] < t)
                    l++;
                vec[r] = vec[l];
            }
            isLeft = !isLeft;
        }
        vec[l] = t;
        if (l < target)
            return quickSelect(vec, l + 1, right, target);
        else if (l > target)
            return quickSelect(vec, left, l - 1, target);
        return vec[l];
    }

private:
    //中值计算
    static uchar medianKernel(const Mat& img, int size, int i, int j, int q) {
        vector<uchar> values;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                values.push_back(img.at<uchar>(r, c));
            }
        }
        int cnt = size * size;
        return quickSelect(values, 0, cnt - 1, cnt / 2);
    }

public:
    MedianFilter(Image img) :Filter(img) {

    }
    void handleMedian(int size, string type = "") {
        string name = type + "中值滤波";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, medianKernel, size));
    }
    static void show(string type, Image img, int size1, int size2) {
        MedianFilter mf(img);
        mf.handleMedian(size1, type);
        mf.handleMedian(size2, type);
    }
};

class AdaptiveMeanFilter :public MeanFilter {
private:
    static uchar adaptiveMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double avg = 0, ds = 0;
        uchar u = img.at<uchar>(i + size / 2, j + size / 2);
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                avg += img.at<uchar>(r, c);
            }
        }
        avg /= size*size;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                ds += pow(avg - img.at<uchar>(r, c), 2);
            }
        }
        double rate = q / ds;
        if (rate > 1.0)
            rate = 1.0;
        return (int)(u - rate * (u - avg));
    }
public:
    AdaptiveMeanFilter(Image img) :MeanFilter(img) {

    }
    //自适应均值
    void handleAdaptiveMeanKernel(int size, int q, string type = "") {
        string name = type + "自适应均值";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        Mat res = filterProcess(img, adaptiveMeanKernel, size, q);
        openWindows(name, res);
    }
    static void show(string type, Image img, int size, int q) {
        AdaptiveMeanFilter mf(img);
        mf.handleAdaptiveMeanKernel(size, q, type);
    }
};

class AdaptiveMedianFilter :MedianFilter {
private:
    //自适应中值计算
    static uchar adaptiveMedianKernel(const Mat& img, int size, int i, int j, int max_size) {
        vector<uchar> values;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                values.push_back(img.at<uchar>(r, c));
            }
        }
        int cnt = size * size;
        int mid = quickSelect(values, 0, cnt - 1, cnt / 2);
        int min = quickSelect(values, 0, cnt - 1, 0);
        int max = quickSelect(values, 0, cnt - 1, cnt - 1);
        if (mid < max && min < mid) {
            uchar u = img.at<uchar>(i + size / 2, j + size / 2);
            return (min < u&& u < max) ? u : mid;
        }
        if (size == max_size || i == 0 || j == 0 || i + size + 1 >= img.rows || j + size + 1 >= img.cols)
            return mid;
        return adaptiveMedianKernel(img, size + 2, i - 1, j - 1, max_size);
    }
public:
    AdaptiveMedianFilter(Image img) :MedianFilter(img) {
        
    }
    void handleAdaptiveMedianKernel(int size, int max_size, string type = "") {
        string name = type + "自适应中值";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        Mat res = filterProcess(img, adaptiveMedianKernel, size, max_size);
        openWindows(name, res);
    }
    static void show(string type, Image img, int size, int max_size) {
        AdaptiveMedianFilter mf(img);
        mf.handleAdaptiveMedianKernel(size, max_size, type);
    }
};

int main() {

    string name = "harden.webp";
    //灰度图像 
    GrayImage gray_img(name);
    //灰度图噪声生成器
    Noise noise(gray_img);
    //高斯噪声
    Image gaussianNoise = noise.addGaussianNoise(0, 30);
    //胡椒噪声
    Image pepperNoise = noise.addPepperNoise(5000);
    //盐噪声
    Image saltNoise = noise.addSaltNoise(5000);
    //椒盐噪声
    Image saltAndPepperNoise = noise.addSaltAndPepperNoise(5000);

    //均值滤波
    MeanFilter::show("高斯-", gaussianNoise, 5, 2);
    MeanFilter::show("胡椒-", pepperNoise, 5, 2);
    MeanFilter::show("盐-", saltNoise, 5, -2);
    MeanFilter::show("椒盐-", saltAndPepperNoise, 5, 2, true);

    //中值滤波
    MedianFilter::show("高斯-", pepperNoise, 5, 9);
    MedianFilter::show("胡椒-", pepperNoise, 5, 9);
    MedianFilter::show("盐-", saltNoise, 5, 9);
    MedianFilter::show("椒盐-", saltAndPepperNoise, 5, 9);

    //自适应均值滤波
    AdaptiveMeanFilter::show("高斯-", gaussianNoise, 7, 10000);

    //自适应中值滤波
    AdaptiveMedianFilter::show("高斯-", pepperNoise, 3, 7);
    AdaptiveMedianFilter::show("胡椒-", pepperNoise, 3, 7);
    AdaptiveMedianFilter::show("盐-", saltNoise, 3, 7);
    AdaptiveMedianFilter::show("椒盐-", saltAndPepperNoise, 7, 7);

    //彩色图像
    ColorImage color_img(name);

    //彩色图噪声生成器
    Noise c_noise(color_img);
    //高斯噪声
    Image c_gaussianNoise = c_noise.addGaussianNoise(0, 30);
    //胡椒噪声
    Image c_pepperNoise = c_noise.addPepperNoise(5000);
    //盐噪声
    Image c_saltNoise = c_noise.addSaltNoise(5000);
    //椒盐噪声
    Image c_saltAndPepperNoise = c_noise.addSaltAndPepperNoise(5000);

    //均值滤波
    MeanFilter::show("高斯-", c_gaussianNoise, 5, 2);
    MeanFilter::show("胡椒-", c_pepperNoise, 5, 2);
    MeanFilter::show("盐-", c_saltNoise, 5, -2);
    MeanFilter::show("椒盐-", c_saltAndPepperNoise, 5, 2, true);
    return 0;
}
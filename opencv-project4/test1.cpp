#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

const static std::string path = "C:\\Users\\13906\\Desktop\\";

//ͼƬ·�����
string getFullPath(string name) {
    return path + name;
}

//��ͼƬ��ʾ����
void openWindows(string win_name, Mat img, int x = 500, int y = 200) {
    //����������ָ����С������λ��
    namedWindow(win_name, WINDOW_AUTOSIZE);
    moveWindow(win_name, x, y);
    //���ɴ�����ʾͼƬ
    imshow(win_name, img);
    //�ȴ�����
    waitKey();
    //�رմ���
    destroyWindow(win_name);
}

//ͳһ�������뺯��
template<typename T>T inputNumber(string desc) {
    system("cls");
    T input;
    cout << desc;
    cin >> input;
    cout << endl;
    return input;
}

//ͳһͼƬ�򿪺��������ڼ�·���ʹ����ͼƬ����
Mat openImage(string name, int type = 1) {
    //ͼƬ��ȡ����������ͼ��洢�ࣨ�����洢��ʽ���洢���󡢾����С�ȣ�
    Mat img = imread(getFullPath(name), type);
    if (img.empty()) {
        cout << "��ЧͼƬ����ȡʧ��" << endl;
        exit(-1);
    }
    return img;
}

//ͼ�����
class Image {
protected:
    Mat img;
public:
    Mat getImage() {
        return img.clone();
    }
};

//��ɫͼ������
class ColorImage :public Image {
public:
    ColorImage(Mat img,string name) {
        this->img = img.clone();
        openWindows(name, img);
    }
    //��ȡ��ɫͼ��չʾ
    ColorImage(string path) {
        img = openImage(path, IMREAD_COLOR);
        openWindows("��ɫͼ��", img);
    }
};

//�Ҷ�ͼ������
class GrayImage :public Image {
public:
    GrayImage(Mat img,string name) {
        this->img = img.clone();
        openWindows(name, img);
    }
    //����ȡ�Ҷȷ�ʽ��ȡͼ��չʾ��IMREAD_GRAYSCALE��
    GrayImage(string path) {
        img = openImage(path, IMREAD_GRAYSCALE);
        openWindows("�Ҷ�ͼ��", img);
    }
};

//�������������
class Noise {
    Mat src;
private:
    //��ֹͼ�������ͼ��Ӽ�����ʹ��saturate_cast��������
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

    //���ָ����ֵ����(����0���Σ�255)
    Mat saltAndPepperNoise(const Mat& input, int n, int value) {
        int x, y;
        int row = input.rows, col = input.cols;
        bool color = input.channels() > 1;
        Mat target = input.clone();
        //���ȡλ�����ָ������
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

    //��ȡ�����˹����ͼ��
    Mat gaussianNoise(const Mat& input, int avg, int sd) {
        Mat target = Mat::zeros(input.size(), input.type());
        //Opencv�������
        RNG rng(rand());
        //��˹�ֲ���ƽ��������ȷֲ�����Сֵ
        int a = avg;
        //��˹�ֲ��ı�׼�����ȷֲ������ֵ
        int b = sd;
        //���ͣ�NORMALΪ��˹�ֲ���UNIFORMΪ���ȷֲ�
        int distType = RNG::NORMAL;
        //����һ�����ϸ�˹�ֲ�������ȷֲ�����ͼ��
        rng.fill(target, distType, a, b);
        return target;
    }

public:
    Noise(Image img) {
        this->src = img.getImage();
    }

    //��Ӻ�������
    Image addPepperNoise(int n) {
        Mat output = saltAndPepperNoise(src, n, 0);
        string name = "��������";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }

    //���������
    Image addSaltNoise(int n) {
        Mat output = saltAndPepperNoise(src, n, 255);
        string name = "������";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }

    //��ӽ�������
    Image addSaltAndPepperNoise(int n) {
        Mat output = saltAndPepperNoise(saltAndPepperNoise(src, n / 2, 0), n / 2, 255);
        string name = "��������";
        if (src.channels() == 1)
            return GrayImage(output, name);
        else
            return  ColorImage(output, name);
    }

    //��Ӹ�˹�ֲ�
    Image addGaussianNoise(int avg = 15, int sd = 30) {
        //��˹����ͼ��
        Mat noise = gaussianNoise(src, avg, sd);
        //����ͼ����ԭͼ�����
        Mat output = handleImageAddition(src, noise);
        string name = "��˹����";
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

    //��ͼ������˲�����
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
    //������ֵ����
    static uchar arithmeticMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res = 0;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res += img.at<uchar>(r, c);
            }
        }
        return (uchar)(res / ((double)size * size));
    }
    //���ξ�ֵ����
    static uchar geometricMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res = 1;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res *= img.at<uchar>(r, c);
            }
        }
        return (uchar)(pow(res, 1.0 / ((double)size * size)));
    }
    //г��ƽ��ֵ����
    static uchar harmonicMeanKernel(const Mat& img, int size, int i, int j, int q) {
        double res = 0;
        for (int r = i + size - 1; r >= i; r--) {
            for (int c = j + size - 1; c >= j; c--) {
                res += 1.0 / ((double)img.at<uchar>(r, c) + 1);
            }
        }
        return (uchar)(((double)size * size) / res - 1);
    }
    //��г��ƽ��ֵ����
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
    //����ƽ���˲�
    void handleArithmeticMean(int size, string type = "") {
        string name = type + "����ƽ��";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, arithmeticMeanKernel, size));
    }
    //����ƽ���˲�
    void handleGeometricMean(int size, string type = "") {
        string name = type + "����ƽ��";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, geometricMeanKernel, size));
    }
    //г��ƽ���˲�
    void handleHarmonicMean(int size, string type = "") {
        string name = type + "г��ƽ��";
        name += (char)(size + '0');
        name += '*';
        name += (char)(size + '0');
        openWindows(name, filterProcess(img, harmonicMeanKernel, size));
    }
    //��г��ƽ���˲�
    void handleAntiHrithmeticMean(int size, int q, string type = "", bool  twice = false) {
        string name = type + "��г��ƽ��";
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
    //����ѡ����ֵ
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
    //��ֵ����
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
        string name = type + "��ֵ�˲�";
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
    //����Ӧ��ֵ
    void handleAdaptiveMeanKernel(int size, int q, string type = "") {
        string name = type + "����Ӧ��ֵ";
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
    //����Ӧ��ֵ����
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
        string name = type + "����Ӧ��ֵ";
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
    //�Ҷ�ͼ�� 
    GrayImage gray_img(name);
    //�Ҷ�ͼ����������
    Noise noise(gray_img);
    //��˹����
    Image gaussianNoise = noise.addGaussianNoise(0, 30);
    //��������
    Image pepperNoise = noise.addPepperNoise(5000);
    //������
    Image saltNoise = noise.addSaltNoise(5000);
    //��������
    Image saltAndPepperNoise = noise.addSaltAndPepperNoise(5000);

    //��ֵ�˲�
    MeanFilter::show("��˹-", gaussianNoise, 5, 2);
    MeanFilter::show("����-", pepperNoise, 5, 2);
    MeanFilter::show("��-", saltNoise, 5, -2);
    MeanFilter::show("����-", saltAndPepperNoise, 5, 2, true);

    //��ֵ�˲�
    MedianFilter::show("��˹-", pepperNoise, 5, 9);
    MedianFilter::show("����-", pepperNoise, 5, 9);
    MedianFilter::show("��-", saltNoise, 5, 9);
    MedianFilter::show("����-", saltAndPepperNoise, 5, 9);

    //����Ӧ��ֵ�˲�
    AdaptiveMeanFilter::show("��˹-", gaussianNoise, 7, 10000);
    AdaptiveMeanFilter::show("����-", pepperNoise, 7, 10000);
    AdaptiveMeanFilter::show("��-", saltNoise, 7, 10000);
    AdaptiveMeanFilter::show("����-", saltAndPepperNoise, 7, 10000);

    //����Ӧ��ֵ�˲�
    AdaptiveMedianFilter::show("��˹-", pepperNoise, 7, 7);
    AdaptiveMedianFilter::show("����-", pepperNoise, 7, 7);
    AdaptiveMedianFilter::show("��-", saltNoise, 7, 7);
    AdaptiveMedianFilter::show("����-", saltAndPepperNoise, 7, 7);

    //��ɫͼ��
    ColorImage color_img(name);

    //��ɫͼ����������
    Noise c_noise(color_img);
    //��˹����
    Image c_gaussianNoise = c_noise.addGaussianNoise(0, 30);
    //��������
    Image c_pepperNoise = c_noise.addPepperNoise(5000);
    //������
    Image c_saltNoise = c_noise.addSaltNoise(5000);
    //��������
    Image c_saltAndPepperNoise = c_noise.addSaltAndPepperNoise(5000);

    //��ֵ�˲�
    MeanFilter::show("��˹-", c_gaussianNoise, 5, 2);
    MeanFilter::show("����-", c_pepperNoise, 5, 2);
    MeanFilter::show("��-", c_saltNoise, 5, -2);
    MeanFilter::show("����-", c_saltAndPepperNoise, 5, 2, true);
    return 0;
}
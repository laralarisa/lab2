#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using std::vector;

uchar calculatePixel(const Mat& image, const vector<vector<double> >& kernel,
                     int i, int j) {
    int newPixel = 0;
    int border = static_cast<int>(kernel.size() / 2);

    //высчитываем значение нового пикселя по заданному ядру
    for (unsigned k = 0; k < kernel.size(); ++k) {
      for (unsigned l = 0; l < kernel.size(); ++l) {
        newPixel += image.at<uchar>(i + static_cast<int>(k) - border,
                                  j + static_cast<int>(l) - border) *
                    kernel[k][l];
      }
    }

    return saturate_cast<uchar>(newPixel);
  }

void customFilter(const Mat& image, Mat& result,
                  const vector<vector<double> >& kernel) {
    int border = static_cast<int>(kernel.size() / 2);
    //если изображение черно-белое
    if (image.type() == 0) {
      for (int i = border; i < image.cols - border + 1; ++i) {
        for (int j = border; j < image.rows - border + 1; ++j) {
          result.at<uchar>(i, j) = calculatePixel(image, kernel, i, j);
        }
      }
    } else {
    // иначе разделяем на отдельные цвета
      vector<Mat> channels;
      split(image, channels);
      vector<Mat> results = channels;

      // работает с каждым цветом отдельно
      for (int i = border; i < image.cols - border + 1; ++i) {
        for (int j = border; j < image.rows - border + 1; ++j) {
          results[0].at<uchar>(i, j) = calculatePixel(channels[0], kernel, i, j);
          results[1].at<uchar>(i, j) = calculatePixel(channels[1], kernel, i, j);
          results[2].at<uchar>(i, j) = calculatePixel(channels[2], kernel, i, j);
        }
      }

      //соединяем обратно
      merge(results, result);
    }
}

void customSmoothing(const Mat& image, Mat& result, unsigned size) {
    image.copyTo(result);

    //матрица для размытия изображения
    vector<vector<double> > kernel(size,

                                   vector<double>(size, 1. / (size * size)));
    customFilter(image, result, kernel);
}

void customGradient(const Mat& image, Mat& result) {
    image.copyTo(result);

    //матрица для градиента, направление - север
    vector<vector<double> > kernel = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};

    customFilter(image, result, kernel);
}



int main(int argc, char** argv) {
  if (argc != 2) return -1;

  Mat image, imageRes;
  image = imread(argv[1], 1);

  GaussianBlur(image, imageRes, Size(15, 15), 0, 0);
  imshow("Gaussian", imageRes);

  customSmoothing(image, imageRes, 5);
  imshow("Custom smoothing", imageRes);

  cvtColor(image, image, COLOR_BGR2GRAY);
  customGradient(image, imageRes);
  imshow("Gradient", imageRes);

  waitKey(0);
  return 0;
}

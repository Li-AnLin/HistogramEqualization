#include <limits>
#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include <opencv2\imgproc\imgproc.hpp>

int* Equalize(int arr[256], int cols, int rows)
{
	int min = INT_MAX;
	//find min
	for (int i = 0; i < 256; i++)
	{
		if (arr[i] <= 0)
			continue;
		if (arr[i] < min)
			min = arr[i];
	}

	int hist[256];
	//equalize
	for (int i = 0; i < 256; i++)
	{
		if (arr[i] <= 0)
			continue;
		hist[i] = round(((arr[i] - (float)min) / (rows * cols - (float)min)) * 255.0);
	}

	return hist;
}

void CalHist(cv::Mat img, cv::Mat& output)
{
	//calculate count
	int count_b[256] = { 0 }, count_g[256] = { 0 }, count_r[256] = { 0 };
	for (int col = 0; col < img.cols; col++)
	{
		for (int row = 0; row < img.rows; row++)
		{		
			//b
			count_b[img.at<cv::Vec3b>(row, col)[0]]++;
			//g
			count_g[img.at<cv::Vec3b>(row, col)[1]]++;
			//r
			if (img.channels() == 3)
			{
				count_r[img.at<cv::Vec3b>(row, col)[2]]++;
			}
		}
	}

	//turn to cdf
	int cdf_b[256] = { 0 }, cdf_g[256] = { 0 }, cdf_r[256] = { 0 };
	cdf_b[0] = count_b[0], cdf_g[0] = count_g[0], cdf_r[0] = count_r[0];	//inital

	for (int i = 1; i < 256; i++)
	{
		cdf_b[i] = cdf_b[i - 1] + count_b[i];
		cdf_g[i] = cdf_g[i - 1] + count_g[i];
		cdf_r[i] = cdf_r[i - 1] + count_r[i];
	}

	//equalize
	int* hist_b = Equalize(cdf_b, img.cols, img.rows);
	int* hist_g = Equalize(cdf_g, img.cols, img.rows);
	int* hist_r = Equalize(cdf_r, img.cols, img.rows);

	//write into output
	for (int col = 0; col < img.cols; col++)
	{
		for (int row = 0; row < img.rows; row++)
		{
			//b
			output.at<cv::Vec3b>(row, col)[0] = hist_b[img.at<cv::Vec3b>(row, col)[0]];
			//g
			output.at<cv::Vec3b>(row, col)[1] = hist_g[img.at<cv::Vec3b>(row, col)[1]];
			//r
			if (img.channels() == 3)
				output.at<cv::Vec3b>(row, col)[2] = hist_r[img.at<cv::Vec3b>(row, col)[2]];

			if (output.at<cv::Vec3b>(row, col)[0] > 255)
				output.at<cv::Vec3b>(row, col)[0] = 255;
			if (output.at<cv::Vec3b>(row, col)[1] > 255)
				output.at<cv::Vec3b>(row, col)[1] = 255;
			if (output.at<cv::Vec3b>(row, col)[2] > 255)
				output.at<cv::Vec3b>(row, col)[2] = 255;

			if (output.at<cv::Vec3b>(row, col)[0] < 0)
				output.at<cv::Vec3b>(row, col)[0] = 0;
			if (output.at<cv::Vec3b>(row, col)[1] < 0)
				output.at<cv::Vec3b>(row, col)[1] = 0;
			if (output.at<cv::Vec3b>(row, col)[2] < 0)
				output.at<cv::Vec3b>(row, col)[2] = 0;
		}
	}

}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		std::cout << "please input one file";
		exit(EXIT_FAILURE);
	}
	
	cv::Mat src = cv::imread(argv[1]);
	cv::Mat output = cv::Mat(src.rows, src.cols, src.type());

	CalHist(src, output);

	cv::imshow("original", src);
	cv::imshow("output", output);

	cv::waitKey(0);
	
	return 0;
}
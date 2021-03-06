#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace cv::dnn;
using namespace std;

const char* classNames[] = { "background","aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair","cow", "diningtable", "dog", "horse","motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor" };

int main()
{
	Scalar colors[21];
	srand((unsigned)time(NULL));
	for (int i = 0; i < 21; i++) {
		colors[i] = Scalar(rand() % 256, rand() % 256, rand() % 256);
	}

	String modelTxt = "deploy.prototxt";
	String modelBin = "VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel";
	String imageFile = "bali-crop.jpg";

	Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	Mat img = cv::imread(imageFile);

	Mat inputBlob = blobFromImage(img, 1, Size(512, 512));   //Convert Mat to batch of images
	Mat prob;

	cv::TickMeter t;
	net.setInput(inputBlob, "data");        //set the network input
	t.start();
	prob = net.forward("detection_out");                          //compute output
	t.stop();
	cout << "Runtime: " << (double)t.getTimeMilli() << " ms " << endl;

	Mat detectionMat(prob.size[2], prob.size[3], CV_32F, prob.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > 0.4)
		{
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

			Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

			rectangle(img, object, colors[objectClass],2);
			String label = String(classNames[objectClass]) + ": " + to_string(confidence);
			std::cout << label << endl;

			int baseLine = 0;
			Size labelSize = getTextSize(label, FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
			rectangle(img, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
				Size(labelSize.width, labelSize.height + baseLine)), colors[objectClass], CV_FILLED);
			putText(img, label, Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(0,0,0));
		}
	}

	cv::imshow("image", img);
	cv::waitKey(0);
	return 0;
}
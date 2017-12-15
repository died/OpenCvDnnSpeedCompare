# OpenCV DNN Speed Test in Python / C# / C++
This is a simple test using OpenCv DNN module load [SSD](https://github.com/weiliu89/caffe/tree/ssd/) model running in different language, compare their running speed.

## Test environment
##### Python
* Python 3.6.3 
* OpenCv 3.3.0.10 ([opencv-python](https://pypi.python.org/pypi/opencv-python))
##### C# 
* .NET Framework 4.7
* EmguCV 3.3.0.2824
* OpenCvSharp 3.3.1.20171117 :new: 
##### C++
* VC14
* OpenCv 3.3.1
#### System
* Windows 10 Pro 64bit
#### Hardware
* Intel Core i7-7820HQ @ 2.90GHz  
*GPU didn't list because OpenCV DNN module can't enable GPU right now*
#### Model
* SSD / PASCAL VOC models : 07++12+COCO: SSD512 from [here](https://github.com/weiliu89/caffe/tree/ssd/#models)

## Result
![result chart](https://i.imgur.com/fjbCs3z.png "OpenCV DNN Speed Test Result")
#### RunTime
* Python with OpenCV: 1510 ms
* C# with OpenCvSharp: 1917 ms :new:
* C# with EmguCV: 4041 ms
* C++ with OpenCV: 9306 ms

---

More detail please check [here](http://www.died.tw/2017/11/opencv-dnn-speed-compare-in-python-c-c.html).  

---
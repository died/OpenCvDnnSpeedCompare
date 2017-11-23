using System;
using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;

namespace emguCvSsd
{
    class Program
    {
        private static readonly string[] Labels = {"background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair","cow", "diningtable", "dog", "horse", "motorbike","person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
        private static readonly MCvScalar[] Colors = new MCvScalar[21];

        static void Main()
        {
            //set random color
            Random rnd = new Random();
            for (int i = 0; i < 21; i++)
            {
                Colors[i] = new Rgb(rnd.Next(0,256), rnd.Next(0, 256), rnd.Next(0, 256)).MCvScalar;
            }
            //get image and set model
            //Mat img = CvInvoke.Imread("bali-crop.jpg");
            Mat img = CvInvoke.Imread("fish-bike.jpg");
            var blob = DnnInvoke.BlobFromImage(img, 1, new Size(512, 515));
            var prototxt = "deploy.prototxt";
            var model = "VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel";
            var net = new Net();
            var import = Importer.CreateCaffeImporter(prototxt, model);
            import.PopulateNet(net);
            net.SetInput(blob, "data");
            
            Stopwatch sw = new Stopwatch();
            sw.Start();
            //forward model
            var prob = net.Forward("detection_out");
            sw.Stop();
            Console.WriteLine($"Runtime:{sw.ElapsedMilliseconds} ms");

            //copy result to byte due to egmucv can't access Mat pixel.
            byte[] data = new byte[5600];
            prob.CopyTo(data);

            //draw result
            for (int i = 0; i < prob.SizeOfDimemsion[2]; i++)
            {
                var d = BitConverter.ToSingle(data, i * 28 + 8);
                if (d > 0.4)
                {
                    var idx = (int)BitConverter.ToSingle(data, i * 28 + 4);
                    var w1 = (int) (BitConverter.ToSingle(data, i * 28 + 12) * img.Width);
                    var h1 = (int)(BitConverter.ToSingle(data, i * 28 + 16) * img.Height);
                    var w2 = (int)(BitConverter.ToSingle(data, i * 28 + 20) * img.Width);
                    var h2 = (int)(BitConverter.ToSingle(data, i * 28 + 24) * img.Height);

                    var label = $"{Labels[idx]} {d * 100:0.00}%";
                    Console.WriteLine(label);
                    CvInvoke.Rectangle(img,new Rectangle(w1,h1,w2-w1,h2-h1),Colors[idx],2);
                    int baseline = 0;
                    var textSize = CvInvoke.GetTextSize(label, FontFace.HersheyTriplex, 0.5, 1, ref baseline);
                    var y = h1 - textSize.Height < 0 ? h1 + textSize.Height : h1;
                    CvInvoke.Rectangle(img,new Rectangle(w1,y-textSize.Height,textSize.Width, textSize.Height),Colors[idx],-1);
                    CvInvoke.PutText(img, label, new Point(w1, y), FontFace.HersheyTriplex, 0.5, new Bgr(0, 0, 0).MCvScalar);
                }
            }

            //Show the image
            CvInvoke.Imshow("image", img); 
            CvInvoke.WaitKey();  
            CvInvoke.DestroyAllWindows();
        }
    }
}

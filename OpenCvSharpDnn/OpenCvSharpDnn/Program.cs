using System;
using System.Diagnostics;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenCvSharpDnn
{
    class Program
    {
        private static readonly string[] Labels = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

        static void Main()
        {
            var file = "bali.jpg";
            var prototxt = "deploy.prototxt";
            var model = "VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel";
            var colors = Enumerable.Repeat(false, 21).Select(x=> Scalar.RandomColor()).ToArray();
            //get image
            var org = Cv2.ImRead(file);
            var blob = CvDnn.BlobFromImage(org,1,new Size(512,512));
            //setup model
            var net = CvDnn.ReadNetFromCaffe(prototxt, model);
            net.SetInput(blob, "data");

            Stopwatch sw = new Stopwatch();
            sw.Start();
            //forward model
            var prob = net.Forward("detection_out");
            sw.Stop();
            Console.WriteLine($"Runtime:{sw.ElapsedMilliseconds} ms");

            //reshape from [1,1,200,7] to [200,7]
            var p = prob.Reshape(1, prob.Size(2));

            for (int i = 0; i < prob.Size(2); i++)
            {
                var confidence = p.At<float>(i, 2);
                if (confidence > 0.4)
                {
                    //get value what we need
                    var idx = (int)p.At<float>(i, 1);
                    var w1 = (int)(org.Width * p.At<float>(i, 3));
                    var h1 = (int)(org.Width * p.At<float>(i, 4));
                    var w2 = (int)(org.Width * p.At<float>(i, 5));
                    var h2 = (int)(org.Width * p.At<float>(i, 6));
                    var label = $"{Labels[idx]} {confidence * 100:0.00}%";
                    Console.WriteLine($"{label}");
                    //draw result
                    Cv2.Rectangle(org, new Rect(w1, h1, w2 - w1, h2 - h1), colors[idx], 2);
                    var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);
                    Cv2.Rectangle(org,new Rect(new Point(w1, h1 - textSize.Height),
                            new Size(textSize.Width, textSize.Height + baseline)), colors[idx], Cv2.FILLED);
                    Cv2.PutText(org, label, new Point(w1, h1), HersheyFonts.HersheyTriplex, 0.5, Scalar.Black);
                }
            }

            using(new Window("image", org))
            {
                Cv2.WaitKey();
            }
        }
    }
}

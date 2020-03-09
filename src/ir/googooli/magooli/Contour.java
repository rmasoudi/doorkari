package ir.googooli.magooli;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.LINE_8;

public class Contour {
    public static void main(String[] args) {
        Random rng=new Random();
        double threshold=50;
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        File[] files = new File("D:\\rasoul\\images\\").listFiles();
        assert files != null;
        for (File file : files) {
            Mat image = Imgcodecs.imread(file.getAbsolutePath());
            Mat gray = new Mat();
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
//            gray.convertTo(gray,CV_32SC1);
            Imgproc.blur(gray, gray, new Size(3, 3));
            Mat cannyOutput = new Mat();
            Imgproc.Canny(gray, cannyOutput, threshold, threshold * 2);
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
//            Mat drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3);
            for (int i = 0; i < contours.size(); i++) {
                Scalar color = new Scalar(255, 0, 0);
                Imgproc.drawContours(image, contours, i, color, 1, LINE_8, hierarchy, 0, new Point());
            }
            Imgcodecs.imwrite("D:\\rasoul\\out\\" + file.getName(), image);
        }
    }
}

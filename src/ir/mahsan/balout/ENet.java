package ir.mahsan.balout;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

public class ENet {

    private static Net net;
    static String[] classes = {
            "Background", "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
            "TrafficLight", "TrafficSign", "Vegetation", "Terrain", "Sky", "Person",
            "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
    };

    public static void main(String[] args) {
        System.out.println(Core.VERSION);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Dnn.readNetFromTorch("D:\\rasoul\\model\\shelfnet\\ShelfNet18_realtime.pth");
        net = Dnn.readNetFromTorch("D:\\rasoul\\model\\enet\\enet.net");
        File[] files = new File("D:\\rasoul\\images\\").listFiles();
        assert files != null;
        for (File file : files) {
            Mat image = Imgcodecs.imread(file.getAbsolutePath());

            Mat segmented = segment(image);

            Imgcodecs.imwrite("D:\\rasoul\\out\\" + file.getName(), segmented);
        }

    }

    private static Mat segment(Mat image) {
        int mainWidth = image.width();
        int mainHeight = image.height();
        long l = System.currentTimeMillis();
        Imgproc.resize(image, image, new Size(1024, 512));
        Mat blob = Dnn.blobFromImage(image, 1.0 / 255, new Size(1024, 512), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        Mat score = net.forward();
        score = score.reshape(1, 20);
        System.out.println(System.currentTimeMillis() - l);
        Mat segmented = new Mat(image.rows(), image.cols(), image.type());
        for (int i = 0; i < score.cols(); i++) {
            double maxVal = -1000;
            int maxCateg = 0;
            for (int categ = 0; categ < score.rows(); categ++) {
                double v = score.get(categ, i)[0];
                if (v > maxVal) {
                    maxVal = v;
                    maxCateg = categ;
                }
            }
            int rowIndex = i / image.width();
            int colIndex = i - rowIndex * image.width();
            segmented.put(rowIndex, colIndex, getColor(maxCateg));
        }
        Imgproc.resize(segmented, segmented, new Size(mainWidth, mainHeight));
        Imgproc.resize(image, image, new Size(mainWidth, mainHeight));
        Core.addWeighted(image, .5, segmented, .5, 0.0, image);
        return image;
    }

    private static double[] getColor(int labelIndex) {

        switch (labelIndex + 1) {
            case 15:
                return new double[]{255, 0, 0};
            case 16:
                return new double[]{0, 255, 0};
            case 17:
                return new double[]{0, 0, 255};
            case 18:
                return new double[]{255, 255, 0};
            case 19:
                return new double[]{0, 255, 255};
            case 20:
                return new double[]{255, 0, 255};
        }
        return new double[]{255, 255, 255};
    }
}

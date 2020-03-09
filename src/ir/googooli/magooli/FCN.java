package ir.googooli.magooli;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

public class FCN {

    private static Net net;

    public static void main(String[] args) {
        System.out.println(Core.VERSION);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        net = Dnn.readNetFromCaffe("D:\\rasoul\\model\\fcn8s-heavy-pascal.prototxt.txt", "D:\\rasoul\\model\\fcn8s-heavy-pascal.caffemodel");
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
        Imgproc.resize(image, image, new Size(500, 500));
        long l = System.currentTimeMillis();
        Mat blob = Dnn.blobFromImage(image);
        net.setInput(blob, "data");
        Mat score = net.forward("score");
        score = score.reshape(1, 21);
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
            int rowIndex = i / image.height();
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
            case 7:
                return new double[]{255, 0, 0};
            case 8:
                return new double[]{0, 255, 0};
            case 15:
                return new double[]{0, 0, 255};
            case 20:
                return new double[]{255, 255, 0};
        }
        return new double[]{255, 255, 255};
    }
}

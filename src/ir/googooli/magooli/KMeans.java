package ir.googooli.magooli;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;

import static org.opencv.core.CvType.CV_32F;

public class KMeans {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        File[] files = new File("D:\\rasoul\\images\\").listFiles();
        assert files != null;
        for (File file : files) {
            Mat image = Imgcodecs.imread(file.getAbsolutePath());
            Mat data = image.reshape(1, image.width() * image.height());
            data.convertTo(data, CV_32F);
            Mat labels = new Mat();
            Mat centers = new Mat();
            Core.kmeans(data, 2, labels, new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 100, 0.1), 10, Core.KMEANS_PP_CENTERS, centers);
            labels = labels.reshape(1, image.height());
            for (int i = 0; i < labels.rows(); i++) {
                for (int j = 0; j < labels.cols(); j++) {
                    try {
                        int clusterIndex = (int) labels.get(i, j)[0];
                        double b = centers.get(clusterIndex, 0)[0];
                        double g = centers.get(clusterIndex, 1)[0];
                        double r = centers.get(clusterIndex, 2)[0];
                        image.put(i,j,b,g,r);
                        System.out.println();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            Imgcodecs.imwrite("D:\\rasoul\\out\\" + file.getName(), image);
        }
    }
}

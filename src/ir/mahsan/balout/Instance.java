package ir.mahsan.balout;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Instance {
    private static Net net;

    public static void main(String[] args) {

        System.out.println(Core.VERSION);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        net = Dnn.readNetFromTensorflow("D:\\rasoul\\model\\instance\\frozen_inference_graph.pb", "D:\\rasoul\\model\\instance\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
        File[] files = new File("D:\\rasoul\\images\\").listFiles();
        assert files != null;
        for (File file : files) {
            Mat image = Imgcodecs.imread(file.getAbsolutePath());

            Mat segmented = segment(image);

//            Imgcodecs.imwrite("D:\\rasoul\\out\\" + file.getName(), segmented);
        }
    }

    private static Mat segment(Mat image) {
        Mat blob = Dnn.blobFromImage(image, 1.0, new Size(), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        List<Mat> results = new ArrayList<>();
        long l = System.currentTimeMillis();
        net.forward(results, Arrays.asList("detection_out_final", "detection_masks"));
        System.out.println(System.currentTimeMillis() - l);
        return null;
    }
}

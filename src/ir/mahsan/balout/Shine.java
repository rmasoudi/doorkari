package ir.mahsan.balout;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;

public class Shine {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        File[] files = new File("D:\\rasoul\\images\\").listFiles();
        assert files != null;
        for (File file : files) {
            Mat image = Imgcodecs.imread(file.getAbsolutePath());
            Mat submat = image.submat(new Rect((int) (.05 * image.width()), (int) (image.height() * .3), (int) (.9 * image.width()), (int) (image.height() * .6)));
            Imgcodecs.imwrite("D:\\rasoul\\out\\" + file.getName(), submat);
        }
    }
}

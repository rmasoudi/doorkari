package ir.googooli.magooli;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.util.Map;

public class Capture {
    public static void main(String[] args) {
        System.out.println(Core.VERSION);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String userPath = System.getProperty("user.home");
        if (!userPath.endsWith(File.separator)) {
            userPath = userPath + File.separator;
        }
        userPath += "nia" + File.separator;
        new File(userPath).mkdirs();
        VideoCapture c = new VideoCapture(0);
        Mat image = new Mat();
        Mat prev = null;
        c.read(image);
        long lastSend = System.currentTimeMillis();
        Cloudinary cloudinary = new Cloudinary(ObjectUtils.asMap(
                "cloud_name", "dr4eclxx1",
                "api_key", "169382814821652",
                "api_secret", "Zq7MK4ARDH6lEbuUW2cBok6XB0o"));
        HttpClient httpclient = HttpClients.createDefault();
        while (c.isOpened()) {
            if (c.read(image)) {
                try {
                    Mat gray = new Mat();
                    Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
                    if (prev != null) {
                        Mat diff = new Mat();
                        Core.absdiff(gray, prev, diff);
                        Imgproc.threshold(diff, diff, 25, 255, Imgproc.THRESH_BINARY);
                        double count = Core.countNonZero(diff);
                        double ratio = count * 100 / (diff.width() * diff.height());
                        if (ratio > 1 && System.currentTimeMillis() - lastSend > 2000) {
                            MatOfByte matOfByte = new MatOfByte();
                            Imgproc.resize(image, image, new Size(300, 300));
                            Imgcodecs.imencode(".jpg", image, matOfByte);
                            byte[] bytes = matOfByte.toArray();
                            Map upload = cloudinary.uploader().upload(bytes, ObjectUtils.asMap(
                                    "public_id", "niayesh_e_man"));
                            Integer version = (Integer) upload.get("version");
                            HttpGet httpGet=new HttpGet("http://azinvista.com/version?version="+version);
                            HttpResponse response = httpclient.execute(httpGet);
                            System.out.println(EntityUtils.toString(response.getEntity()));
                        }
                    }
                    prev = gray.clone();
                    gray.release();
                    image.release();
                } catch (Throwable e) {
                    e.printStackTrace();
                }
            }

        }
        System.exit(0);
    }
}

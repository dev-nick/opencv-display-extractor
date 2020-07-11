package com.mkyong.hashing;

import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import org.opencv.core.Point; 

public class App 
{
    public Mat warp(MatOfPoint points, Mat grayImage) {

		int i=0;
		int[][] pts = new int[4][2];
		for(Point p: points.toArray()) {
			pts[i][0] = (int)p.x;
			pts[i][1] = (int)p.y;
			System.out.println(pts[i][0] + ","  +pts[i][1]);
			i++;
		}

		Arrays.sort(pts, Comparator.comparing((int[] arr) -> arr[0]));
		int[][] l={{pts[0][0], pts[0][1]},{pts[1][0], pts[1][1]}};
		int[][] r={{pts[2][0], pts[2][1]},{pts[3][0], pts[3][1]}};
		Arrays.sort(l, Comparator.comparing((int[] arr) -> arr[1]));
		Point tl = new Point(l[0][0], l[0][1]), bl = new Point(l[1][0], l[1][1]), br,tr;

		double dist1 = Math.hypot(Math.abs(r[0][1]-tl.y), Math.abs(r[0][0]-tl.x));
		double dist2 = Math.hypot(Math.abs(r[1][1]-tl.y), Math.abs(r[1][0]-tl.x));
		if (dist1 > dist2) {
			br = new Point(r[0][0], r[0][1]);
			tr = new Point(r[1][0], r[1][1]);
		} else {
			br = new Point(r[1][0], r[1][1]);
			tr = new Point(r[0][0], r[0][1]);
		}

		double widthA = Math.sqrt(Math.pow((br.x - bl.x),2) + Math.pow((br.y - bl.y),2));
		double widthB = Math.sqrt(Math.pow((tr.x - tl.x),2) + Math.pow((tr.y - tl.y),2));
		double heightA = Math.sqrt((Math.pow((tr.x - br.x),2)) + (Math.pow((tr.y - br.y),2)));
		double heightB = Math.sqrt((Math.pow((tl.x - bl.x),2)) + (Math.pow((tl.y - bl.y),2)));
		int maxWidth = Math.max((int)widthA, (int)widthB);
		int maxHeight = Math.max((int)heightA, (int)heightB);

		Mat srcw = new MatOfPoint2f(tl,tr,br,bl);
		Mat dstw = new MatOfPoint2f(
			new Point(0, 0),
			new Point(maxWidth - 1, 0),
			new Point(maxWidth - 1, maxHeight - 1),
			new Point(0, maxHeight - 1));
		Mat transform = Imgproc.getPerspectiveTransform(srcw, dstw);
		Mat destImage=new Mat();// = new Mat(new Size(maxWidth, maxHeight), src.type());
		Imgproc.warpPerspective(grayImage, destImage, transform, new Size(maxWidth, maxHeight));

		points.release();
		srcw.release();
		dstw.release();
		return destImage;
    }

    public static void main( String[] args )
    {
    	OpenCV.loadShared();
		int height = 500; // required height of image
		Mat src = Imgcodecs.imread("/home/qaz/Downloads/recognize-digits/" + args[0]);
		Mat grayImage=new Mat(), hsvImage=new Mat(), hierarchy=new Mat();
		MatOfPoint points=null;

		double scaleratio = src.size().height/500;
		double width = src.size().width;
		Size scaleSize = new Size((int)(width/scaleratio) ,height);
		Imgproc.resize(src, src, scaleSize , 0, 0, Imgproc.INTER_AREA);
		Imgproc.cvtColor(src, grayImage, Imgproc.COLOR_BGR2GRAY);
		Imgproc.cvtColor(src, hsvImage, Imgproc.COLOR_BGR2HSV, 3);

		Scalar lower = new Scalar(17,14,50), upper = new Scalar(58,160,222);
		Scalar lower2 = new Scalar(84,27,72), upper2 = new Scalar(111,130,255);
		Scalar lower3 = new Scalar(0,0,48), upper3 = new Scalar(0,0,160);

        Mat skinMask = new Mat(hsvImage.rows(), hsvImage.cols(), CvType.CV_8U, new Scalar(3));
        Core.inRange(hsvImage, lower, upper, skinMask);
        Mat skinMask2 = new Mat(hsvImage.rows(), hsvImage.cols(), CvType.CV_8U, new Scalar(3));
        Core.inRange(hsvImage, lower2, upper2, skinMask2);
		Core.add(skinMask, skinMask2, skinMask);
        Core.inRange(hsvImage, lower3, upper3, skinMask2);
		Core.add(skinMask, skinMask2, skinMask);
        Mat skin = new Mat(skinMask.rows(), skinMask.cols(), CvType.CV_8U, new Scalar(3));
        Core.bitwise_or(src, src, skin, skinMask);


		ArrayList<Mat> channels = new ArrayList<Mat>(3);
		Core.split(skin, channels);
		Mat saturation = channels.get(1);

		ArrayList<MatOfPoint> contours = new ArrayList<>();
		Imgproc.findContours(saturation, contours, hierarchy, Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_NONE);//simple
		
		skinMask.release(); // release memory of frames
		skinMask2.release();
		hierarchy.release();
		hsvImage.release();
		channels.get(0).release();
		channels.get(2).release();
		src.release();

	    double approxDistance = 0;
		Collections.sort(contours, Collections.reverseOrder(Comparator.comparing(Imgproc::contourArea)));
	    MatOfPoint2f approxCurve = new MatOfPoint2f(), contour2f = null;
		for(int c = 0; c < contours.size(); c++)
		{
		    contour2f = new MatOfPoint2f(contours.get(c).toArray());
		    approxDistance = Imgproc.arcLength(contour2f,  true)*0.02;
		    Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
			contour2f.release();
		    if (approxCurve.toArray().length==4){
		        points = new MatOfPoint( approxCurve.toArray() );
		        break;
		    }
			approxCurve.release();
		}

		App obj = new App();
		Mat destImage = obj.warp(points, grayImage);
		
		HighGui.imshow("finalop", destImage);
		HighGui.waitKey(0);
    }
}
package pers.zylo117.nextgen.explorer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImageMatching {
	/**
	 * 
	 * @param scene
	 *            参考图
	 * @param object
	 *            目标需要调整图
	 */
	static void tiaozheng2(Mat scene, Mat object) {

		FeatureDetector detector = FeatureDetector.create(FeatureDetector.FAST);
		DescriptorExtractor descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		// DETECTION

		// first image
		Mat descriptors_scene = new Mat();
		MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();
		detector.detect(scene, keypoints_scene);
		descriptor.compute(scene, keypoints_scene, descriptors_scene);

		// second image
		Mat descriptors_object = new Mat();
		MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
		detector.detect(object, keypoints_object);
		descriptor.compute(object, keypoints_object, descriptors_object);
		// MATCHING
		// match these two keypoints sets
		List<MatOfDMatch> matches = new ArrayList<MatOfDMatch>();
		matcher.knnMatch(descriptors_object, descriptors_scene, matches, 5);
		/////////////////////////////////////////////////////////////
		// ratio test
		LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
		for (Iterator<MatOfDMatch> iterator = matches.iterator(); iterator.hasNext();) {
			MatOfDMatch matOfDMatch = (MatOfDMatch) iterator.next();
			if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.9) {
				good_matches.add(matOfDMatch.toArray()[0]);
			}
		}
		// get keypoint coordinates of good matches to find homography and remove
		// outliers using ransac
		List<Point> pts_object = new ArrayList<Point>();
		List<Point> pts_scene = new ArrayList<Point>();
		for (int i = 0; i < good_matches.size(); i++) {
			pts_object.add(keypoints_object.toList().get(good_matches.get(i).queryIdx).pt);
			pts_scene.add(keypoints_scene.toList().get(good_matches.get(i).trainIdx).pt);
		}
		// convertion of data types - there is maybe a more beautiful way
		Mat outputMask = new Mat();
		MatOfPoint2f pts_objectMat = new MatOfPoint2f();
		pts_objectMat.fromList(pts_object);
		MatOfPoint2f pts_sceneMat = new MatOfPoint2f();
		pts_sceneMat.fromList(pts_scene);

		Mat Homog = Calib3d.findHomography(pts_objectMat, pts_sceneMat, Calib3d.RANSAC, 10, outputMask, 2000, 0.995);
		Mat resultMat = new Mat(new Size(object.cols(), object.rows()), object.type());
		Imgproc.warpPerspective(object, object, Homog, resultMat.size());

		// // outputMask contains zeros and ones indicating which matches are filtered
		LinkedList<DMatch> better_matches = new LinkedList<DMatch>();
		for (int i = 0; i < good_matches.size(); i++) {
			// System.out.println(outputMask.get(i, 0)[0]);
			if (outputMask.get(i, 0)[0] != 0.0) {
				better_matches.add(good_matches.get(i));
			}
		}
		// System.out.println(better_matches.toString());
		/////////////////////////////////////////////////////////////
		// // DRAWING OUTPUT
		Mat outputImg = new Mat();
		// this will draw all matches, works fine
		MatOfDMatch better_matches_mat = new MatOfDMatch();
		better_matches_mat.fromList(better_matches);
		// System.out.println(better_matches_mat.toString());
		Features2d.drawMatches(object, keypoints_object, scene, keypoints_scene, better_matches_mat, outputImg);
		Mat outputImg2 = new Mat();
		Features2d.drawMatches2(object, keypoints_object, scene, keypoints_scene, matches, outputImg2);
		Imgcodecs.imwrite("result.jpg", outputImg);
		Imgcodecs.imwrite("result2.jpg", outputImg2);
	}

	/**
	 * 
	 * @param scene
	 *            参考图
	 * @param object
	 *            目标需要调整图
	 */
	static void tiaozheng(Mat scene, Mat object) {

		FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);
		MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
		MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();

		detector.detect(object, keypoints_object);
		detector.detect(scene, keypoints_scene);

		Mat descriptors_object = new Mat();
		Mat descriptors_scene = new Mat();

		DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		extractor.compute(object, keypoints_object, descriptors_object);
		extractor.compute(scene, keypoints_scene, descriptors_scene);
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

		MatOfDMatch matches = new MatOfDMatch();
		matcher.match(descriptors_object, descriptors_scene, matches);
		DMatch[] tmmpDMatchs = matches.toArray();

		double max_dist = 0;
		double min_dist = 400;
		for (int i = 0; i < descriptors_object.rows(); i++) {
			double dist = tmmpDMatchs[i].distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
		for (int i = 0; i < descriptors_object.rows(); ++i) {
			double dist = tmmpDMatchs[i].distance;
			if (dist < 3 * min_dist) {
				good_matches.add(tmmpDMatchs[i]);
			}
		}

		List<Point> pts_scene = new ArrayList<Point>();
		List<Point> pts_object = new ArrayList<Point>();//
		for (int i = 0; i < good_matches.size(); i++) {
			pts_object.add(keypoints_object.toList().get(good_matches.get(i).queryIdx).pt);
			pts_scene.add(keypoints_scene.toList().get(good_matches.get(i).trainIdx).pt);
		}

		Mat outputMask = new Mat();
		MatOfPoint2f pts_sceneMat = new MatOfPoint2f();
		pts_sceneMat.fromList(pts_scene);
		MatOfPoint2f pts_objectMat = new MatOfPoint2f();
		pts_objectMat.fromList(pts_object);

		Mat Homog = Calib3d.findHomography(pts_objectMat, pts_sceneMat, Calib3d.RANSAC, 10, outputMask, 2000, 0.995);
		Mat resultMat = new Mat(new Size(scene.cols(), scene.rows()), scene.type());
		Imgproc.warpPerspective(object, object, Homog, resultMat.size());
	}

	public static void main(String args[]) {
		System.loadLibrary("opencv_java330_64");

		Mat scene = Imgcodecs.imread("00.jpg");// 参考
		Mat object = Imgcodecs.imread("11.jpg");// 调整目标
		object.convertTo(object, CvType.CV_8UC3);// 目标图
		scene.convertTo(scene, CvType.CV_8UC3);// 参考图
		Mat img2 = object.submat(new Rect(0, 0, object.width(), object.height() * 9 / 10));
		Mat img1 = scene.submat(new Rect(0, 0, object.width(), object.height() * 9 / 10));
//		Imgcodecs.imwrite("scene.jpg", img1);// 调整目标
//		Imgcodecs.imwrite("object.jpg", img2);// 调整目标
		// tiaozheng(img1,img2);

//		MatView.imshow(scene, "ori");
		
		tiaozheng2(img1, img2);
//		Imgcodecs.imwrite("object2.jpg", img2);
	}
}

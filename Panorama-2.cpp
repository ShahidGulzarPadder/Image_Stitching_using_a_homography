#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;

#define SQRT2 1.41


void normalizePoints(
    const vector<Point2f>& input_source_points_,
    const vector<Point2f>& input_destination_points_,
    vector<Point2f>& output_source_points_,
    vector<Point2f>& output_destination_points_,
    Mat& T1_,
    Mat& T2_)
{
    int ptsNum = input_source_points_.size();

    //Here calculation of means takes place (they will be the center of coordinate systems)
    float mean1x = 0.0, mean1y = 0.0, mean2x = 0.0, mean2y = 0.0;
    for (int i = 0; i < ptsNum; i++) {
        mean1x += input_source_points_[i].x;
        mean1y += input_source_points_[i].y;
        mean2x += input_destination_points_[i].x;
        mean2y += input_destination_points_[i].y;
    }
    mean1x /= ptsNum;
    mean1y /= ptsNum;
    mean2x /= ptsNum;
    mean2y /= ptsNum;

    float spread1x = 0.0, spread1y = 0.0, spread2x = 0.0, spread2y = 0.0;

    for (int i = 0; i < ptsNum; i++) {
        spread1x += (input_source_points_[i].x - mean1x) * (input_source_points_[i].x - mean1x);
        spread1y += (input_source_points_[i].y - mean1y) * (input_source_points_[i].y - mean1y);
        spread2x += (input_destination_points_[i].x - mean2x) * (input_destination_points_[i].x - mean2x);
        spread2y += (input_destination_points_[i].y - mean2y) * (input_destination_points_[i].y - mean2y);
    }

    spread1x /= ptsNum;
    spread1y /= ptsNum;
    spread2x /= ptsNum;
    spread2y /= ptsNum;

    Mat offs1 = Mat::eye(3, 3, CV_32F);
    Mat offs2 = Mat::eye(3, 3, CV_32F);
    Mat scale1 = Mat::eye(3, 3, CV_32F);
    Mat scale2 = Mat::eye(3, 3, CV_32F);

    offs1.at<float>(0, 2) = -mean1x;
    offs1.at<float>(1, 2) = -mean1y;

    offs2.at<float>(0, 2) = -mean2x;
    offs2.at<float>(1, 2) = -mean2y;

    scale1.at<float>(0, 0) = SQRT2 / sqrt(spread1x);
    scale1.at<float>(1, 1) = SQRT2 / sqrt(spread1y);

    scale2.at<float>(0, 0) = SQRT2 / sqrt(spread2x);
    scale2.at<float>(1, 1) = SQRT2 / sqrt(spread2y);

    T1_ = scale1 * offs1;
    T2_ = scale2 * offs2;

    for (int i = 0; i < ptsNum; i++) {
        Point2f p1D;
        Point2f p2D;

        p1D.x = SQRT2 * (input_source_points_[i].x - mean1x) / sqrt(spread1x);
        p1D.y = SQRT2 * (input_source_points_[i].y - mean1y) / sqrt(spread1y);

        p2D.x = SQRT2 * (input_destination_points_[i].x - mean2x) / sqrt(spread2x);
        p2D.y = SQRT2 * (input_destination_points_[i].y - mean2y) / sqrt(spread2y);

        output_source_points_.push_back(p1D);
        output_destination_points_.push_back(p2D);
    }
}


Mat calcHomography(vector<pair<Point2f, Point2f> > pointPairs) {
    const int ptsNum = pointPairs.size();
    Mat A(2 * ptsNum, 9, CV_32F);
    for (int i = 0; i < ptsNum; i++) {
        float u1 = pointPairs[i].first.x;
        float v1 = pointPairs[i].first.y;

        float u2 = pointPairs[i].second.x;
        float v2 = pointPairs[i].second.y;

        A.at<float>(2 * i, 0) = u1;
        A.at<float>(2 * i, 1) = v1;
        A.at<float>(2 * i, 2) = 1.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = 0.0f;
        A.at<float>(2 * i, 5) = 0.0f;
        A.at<float>(2 * i, 6) = -u2 * u1;
        A.at<float>(2 * i, 7) = -u2 * v1;
        A.at<float>(2 * i, 8) = -u2;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = 0.0f;
        A.at<float>(2 * i + 1, 3) = u1;
        A.at<float>(2 * i + 1, 4) = v1;
        A.at<float>(2 * i + 1, 5) = 1.0f;
        A.at<float>(2 * i + 1, 6) = -v2 * u1;
        A.at<float>(2 * i + 1, 7) = -v2 * v1;
        A.at<float>(2 * i + 1, 8) = -v2;

    }

    Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
    cout << A << endl;
    eigen(A.t() * A, eVals, eVecs);

    cout << eVals << endl;
    cout << eVecs << endl;


    Mat H(3, 3, CV_32F);
    for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

    cout << H << endl;

    //Normalize:
    H = H * (1.0 / H.at<float>(2, 2));
    cout << H << endl;

    return H;
}






//Tranformation of images

void transformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective) {
    Mat invTr = tr.inv();
    const int WIDTH = origImg.cols;
    const int HEIGHT = origImg.rows;

    const int newWIDTH = newImage.cols;
    const int newHEIGHT = newImage.rows;



    for (int x = 0; x < newWIDTH; x++) for (int y = 0; y < newHEIGHT; y++) {
        Mat pt(3, 1, CV_32F);
        pt.at<float>(0, 0) = x;
        pt.at<float>(1, 0) = y;
        pt.at<float>(2, 0) = 1.0;

        Mat ptTransformed = invTr * pt;
        //        cout <<pt <<endl;
        //        cout <<invTr <<endl;
        //        cout <<ptTransformed <<endl;
        if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

        int newX = round(ptTransformed.at<float>(0, 0));
        int newY = round(ptTransformed.at<float>(1, 0));

        if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT)) newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);

        //        printf("x:%d y:%d newX:%d newY:%d\n",x,y,newY,newY);
    }
}

int getIterationNumber(int point_number_,
    int inlier_number_,
    int sample_size_,
    double confidence_)
{
    const double inlier_ratio = static_cast<float>(inlier_number_) / point_number_;

    static const double log1 = log(1.0 - confidence_);
    const double log2 = log(1.0 - pow(inlier_ratio, sample_size_));

    const int k = log1 / log2;
    if (k < 0)
        return INT_MAX;
    return k;
}

Mat ransacHMatrix(
    const vector<Point2f>& normalized_input_src_points_,
    const vector<Point2f>& normalized_input_destination_points_,
    vector<size_t>& inliers_,
    double confidence_,
    double threshold_)
{
    srand(time(NULL));
    // The so-far-the-best H
    Mat best_H(3, 3, CV_32F);
    // The number of correspondences
    const size_t point_number = normalized_input_src_points_.size();
    float prev_error = numeric_limits<double>::max();
    // Initializing the index pool from which the minimal samples are selected
    vector<size_t> index_pool(point_number);
    for (size_t i = 0; i < point_number; ++i)
        index_pool[i] = i;

    // The size of a minimal sample
    constexpr size_t sample_size = 4;
    // The minimal sample
    size_t* mss = new size_t[sample_size];

    size_t maximum_iterations = numeric_limits<int>::max(), // The maximum number of iterations set adaptively when a new best model is found
        iteration_limit = 5000, // A strict iteration limit which mustn't be exceeded
        iteration = 0; // The current iteration number

    vector<Point2f> source_points(sample_size),
        destination_points(sample_size);

    while (iteration++ < MIN(iteration_limit, maximum_iterations))
    {
        vector<pair<Point2f, Point2f>> pointPairs;
        for (auto sample_idx = 0; sample_idx < sample_size; ++sample_idx)
        {
            // Select a random index from the pool
            const size_t idx = round((rand() / (double)RAND_MAX) * (index_pool.size() - 1));
            mss[sample_idx] = index_pool[idx];
            index_pool.erase(index_pool.begin() + idx);

            // Put the selected correspondences into the point containers
            const size_t point_idx = mss[sample_idx];
            source_points[sample_idx] = normalized_input_src_points_[point_idx];
            destination_points[sample_idx] = normalized_input_destination_points_[point_idx];

            pair<Point2f, Point2f> Point;
            Point.first = source_points[sample_idx];
            Point.second = destination_points[sample_idx];
            pointPairs.push_back(Point);
        }

        // Estimate H matrix
        Mat H_(3, 3, CV_32F);
        H_ = calcHomography(pointPairs);

        vector<size_t> inliers;
        for (auto i = 0; i < point_number; ++i) {
            Mat pt1(3, 1, CV_32F);
            pt1.at<float>(0, 0) = normalized_input_src_points_[i].x;
            pt1.at<float>(1, 0) = normalized_input_src_points_[i].y;
            pt1.at<float>(2, 0) = 1;
            Mat pt2(3, 1, CV_32F);
            pt2.at<float>(0, 0) = normalized_input_destination_points_[i].x;
            pt2.at<float>(1, 0) = normalized_input_destination_points_[i].y;
            pt2.at<float>(2, 0) = 1;
            double error1 = pow(norm(H_ * pt1, pt2), 2);
            double error2 = pow(norm(pt1, H_.inv() * pt2), 2);
            double average_error = (error1 + error2) * 0.5;
            if (average_error < threshold_)
                inliers.push_back(i);
        }
        // Update if the new model is better than the previous so-far-the-best.
        if (inliers_.size() < inliers.size())
        {
            cout << "Inlier size: " << inliers.size();
            // Update the set of inliers
            inliers_.swap(inliers);
            inliers.clear();
            inliers.resize(0);
            // Update the model parameters
            best_H = H_;
            // Update the iteration number
            maximum_iterations = getIterationNumber(point_number,
                inliers_.size(),
                sample_size,
                confidence_);
            cout << " Max iterations: " << maximum_iterations << endl;
        }
        // Put back the selected points to the pool
        for (size_t i = 0; i < sample_size; ++i)
            index_pool.push_back(mss[i]);
    }

    return best_H;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cout << " Usage: point_file img1 img2" << endl;
        return -1;
    }

    /*

    with manual features we use this hidden code and just replace input to normalize points from src_points, dest_points to image1_points, image2_points

    MatrixReaderWriter mtxrw(argv[1]);

    if ((mtxrw.rowNum != 4) || (mtxrw.columnNum == 0))
    {
        cout << "Point file format error" << std::endl;
        return -1;
    }


    int r = mtxrw.rowNum;
    int c = mtxrw.columnNum;

    //Convert the coordinates:
    vector<pair<Point2f, Point2f> > pointPairs;
    vector<Point2f> image1_points;
    vector<Point2f> image2_points;
    for (int i = 0; i < mtxrw.columnNum; i++) {
        pair<Point2f, Point2f> currPts;
        currPts.first = Point2f((float)mtxrw.data[i], (float)mtxrw.data[c + i]);
        image1_points.push_back(currPts.first);
        currPts.second = Point2f((float)mtxrw.data[2 * c + i], (float)mtxrw.data[3 * c + i]);
        image2_points.push_back(currPts.second);
        pointPairs.push_back(currPts);
    }

    Mat H = calcHomography(pointPairs);

    */
    std::vector<cv::Point2f> src_points;
    std::vector<cv::Point2f> dest_points;

    std::ifstream file(argv[1]);

    int n;
    file >> n;

    for (int i = 0; i < n; i++) {
        double u1, v1, u2, v2;
        file >> u1 >> v1 >> u2 >> v2;
        src_points.push_back(cv::Point2d(u1, v1));
        dest_points.push_back(cv::Point2d(u2, v2));
    }

    file.close();

    Mat image1;
    image1 = imread(argv[2]);   // Read the file

    if (!image1.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << argv[2] << std::endl;
        return -1;
    }

    Mat image2;
    image2 = imread(argv[3]);   // Read the file

    if (!image2.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << argv[3] << std::endl;
        return -1;
    }


    //    cout <<tr.inv() <<endl;



    Mat T1(3, 3, CV_32F), T2(3, 3, CV_32F); // Normalizing transformations
    vector<Point2f> normalized_source_points, normalized_destination_points; // Normalized point correspondences
    normalizePoints(src_points, // Points in the first image 
        dest_points,  // Points in the second image
        normalized_source_points,  // Normalized points in the first image
        normalized_destination_points, // Normalized points in the second image
        T1, // Normalizing transformation in the first image
        T2); // Normalizing transformation in the second image


    Mat best_H(3, 3, CV_32F);
    vector<size_t> inliers;
    best_H = ransacHMatrix(normalized_source_points,   // Normalized points in the first image 
        normalized_destination_points,   // Normalized points in the second image
        inliers, // The inliers of the fundamental matrix
        0.99, // The required confidence in the results 
        0.1); // The inlier-outlier threshold

    best_H = T2.inv() * best_H * T1; // Denormalize the H matrix
    best_H = best_H * (1.0 / best_H.at<float>(2, 2));
    cout << best_H << endl;


    Mat transformedImage = Mat::zeros(1.5 * image1.size().height, 2.0 * image1.size().width, image1.type());
    transformImage(image2, transformedImage, Mat::eye(3, 3, CV_32F), true);

    transformImage(image1, transformedImage, best_H, true);

    imwrite("ASIFT_Res1.png", transformedImage);

    namedWindow("Result", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Result", transformedImage);                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
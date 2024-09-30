#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp> 

const int MAX_FEATURE_POINTS = 700;

std::vector<double> original_angles, original_x, original_y;
std::vector<double> stabilized_angles, stabilized_x, stabilized_y;

using namespace std;
using namespace cv;
using namespace cv::cuda;

double old_dx, old_dy, old_da;
double new_dx, new_dy, new_da;

double stab_old_dx, stab_old_dy, stab_old_da;
double stab_new_dx, stab_new_dy, stab_new_da;

//Parameters for Kalman Filter
#define Q1 0.001
#define R1 0.1
double errscaleX = 1;
double errscaleY = 1;
double errthetha = 1;
double errtransX = 1;
double errtransY = 1;

double Q_scaleX = Q1;
double Q_scaleY = Q1;
double Q_thetha = Q1;
double Q_transX = Q1;
double Q_transY = Q1;

double R_scaleX = R1;
double R_scaleY = R1;
double R_thetha = R1;
double R_transX = R1;
double R_transY = R1;

double sum_transX = 0;
double sum_transY = 0;
double sum_thetha = 0;
double sum_scaleX = 0;
double sum_scaleY = 0;
int k = 1;
bool cnt = true;
double scaleX = 0;
double scaleY = 0;
double thetha = 0;
double transX = 0;
double transY = 0;


static Ptr<cv::cuda::ORB> d_orb = cv::cuda::ORB::create(MAX_FEATURE_POINTS, 1.2f, 8);


static Ptr<cv::cuda::DescriptorMatcher> d_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);


static cv::cuda::Stream stream;


Mat gray1, gray2;
cv::cuda::GpuMat d_img1, d_img2, d_kpts1, d_kpts2, d_descriptors1, d_descriptors2;

cv::cuda::GpuMat d_matches;

cuda::GpuMat d_stabilized_frame;

Mat resizeAndCropFrame(Mat& frame, double scaleFactor) {
   int frameWidth = frame.cols;
   int frameHeight = frame.rows;

   
   Size newSize(frameWidth * scaleFactor, frameHeight * scaleFactor);
   Mat resizedFrame;
   cv::resize(frame, resizedFrame, newSize, 0, 0, INTER_LANCZOS4);

   
   Rect cropRegion((resizedFrame.cols - frameWidth) / 2, (resizedFrame.rows - frameHeight) / 2, frameWidth, frameHeight);
   Mat croppedFrame = resizedFrame(cropRegion);

   return croppedFrame;
}

void ORB_GPU(Mat& img1, Mat& img2, std::vector<KeyPoint>& keypoints1,
    std::vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2,
    std::vector<DMatch>& matches) {

    if (cnt) {
        
        cv::cvtColor(img1, gray1, COLOR_BGR2GRAY);
        cv::cvtColor(img2, gray2, COLOR_BGR2GRAY);

        cv::convertScaleAbs(gray1, gray1);
        cv::convertScaleAbs(gray2, gray2);
        cv::equalizeHist(gray1, gray1);
        cv::equalizeHist(gray2, gray2);

       
        d_img1.upload(gray1, stream);
        d_img2.upload(gray2, stream);

        
        auto start = std::chrono::high_resolution_clock::now();
        try {
            d_orb->setBlurForDescriptor(false);
            d_orb->detectAndComputeAsync(d_img1, cv::cuda::GpuMat(), d_kpts1, d_descriptors1, false, stream);
            d_orb->detectAndComputeAsync(d_img2, cv::cuda::GpuMat(), d_kpts2, d_descriptors2, false, stream);
        }
        catch (const cv::Exception& e) {
            cerr << "OpenCV Error: " << e.what() << endl;
            return;
        }
        //stream.waitForCompletion();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        cout << "Thoi gian phat hien diem dac trung va tinh toan vector (GPU): " << elapsed.count() << " ms" << endl;

        // Tải keypoints và descriptors từ GPU xuống CPU
        d_orb->convert(d_kpts1, keypoints1);
        d_orb->convert(d_kpts2, keypoints2);
        d_descriptors1.download(descriptors1, stream);
        d_descriptors2.download(descriptors2, stream);
        stream.waitForCompletion();
    }
    else {
        
        cv::cvtColor(img2, gray2, COLOR_BGR2GRAY);

        cv::convertScaleAbs(gray2, gray2);
        cv::equalizeHist(gray2, gray2);

        
        d_img2.upload(gray2, stream);

       
        auto start = std::chrono::high_resolution_clock::now();
        try {
            d_orb->setBlurForDescriptor(false);
            d_orb->detectAndComputeAsync(d_img2, cv::cuda::GpuMat(), d_kpts2, d_descriptors2, true, stream);
        }
        catch (const cv::Exception& e) {
            cerr << "OpenCV Error: " << e.what() << endl;
            return;
        }
        //stream.waitForCompletion();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        cout << "Thoi gian phat hien diem dac trung va tinh toan vector (GPU): " << elapsed.count() << " ms" << endl;

        
        d_descriptors1.upload(descriptors1, stream);
        d_orb->convert(d_kpts2, keypoints2);
        d_descriptors2.download(descriptors2, stream);
        stream.waitForCompletion();
    }

   
    auto start2 = std::chrono::high_resolution_clock::now();
    d_matcher->matchAsync(d_descriptors1, d_descriptors2, d_matches, noArray(), stream);
    d_matcher->matchConvert(d_matches, matches);
    stream.waitForCompletion();

    // Kiểm tra nếu tìm thấy matches
    if (matches.empty()) {
        cerr << "No matches found." << endl;
        return;
    }

   
    std::sort(matches.begin(), matches.end(), [](DMatch& a, DMatch& b) {
        return a.distance < b.distance;
        });

    
    matches.resize(static_cast<size_t>(matches.size() * 1));
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    cout << "Thoi gian matching (GPU): " << elapsed2.count() << " ms" << endl;

    cnt = false; // Cập nhật cnt cho lần tiếp theo
}

const int windowSize = 5;


std::deque<double> scaleXHistory, scaleYHistory, thethaHistory, transXHistory, transYHistory;

double movingAverage(std::deque<double>& history, double newValue) {
   if (history.size() >= windowSize) {
       history.pop_front();
   }
   history.push_back(newValue);

   double sum = 0;
   for (double value : history) {
       sum += value;
   }

   return sum / history.size();
}


double calculateStandardDeviation(const std::deque<double>& history, double mean) {
   double sum = 0;
   for (double value : history) {
       sum += (value - mean) * (value - mean);
   }
   return sqrt(sum / history.size());
}

void Kalman_Filter(double* scaleX, double* scaleY, double* thetha, double* transX, double* transY)
{
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *thetha = frame_1_thetha + gain_thetha * (sum_thetha - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = (1 - gain_scaleX) * frame_1_errscaleX;
    errscaleY = (1 - gain_scaleY) * frame_1_errscaleX;
    errthetha = (1 - gain_thetha) * frame_1_errthetha;
    errtransX = (1 - gain_transX) * frame_1_errtransX;
    errtransY = (1 - gain_transY) * frame_1_errtransY;

    double meanScaleX = movingAverage(scaleXHistory, *scaleX);
    double meanScaleY = movingAverage(scaleYHistory, *scaleY);
    double meanThetha = movingAverage(thethaHistory, *thetha);
    double meanTransX = movingAverage(transXHistory, *transX);
    double meanTransY = movingAverage(transYHistory, *transY);

    double stdDevScaleX = calculateStandardDeviation(scaleXHistory, meanScaleX);
    double stdDevScaleY = calculateStandardDeviation(scaleYHistory, meanScaleY);
    double stdDevThetha = calculateStandardDeviation(thethaHistory, meanThetha);
    double stdDevTransX = calculateStandardDeviation(transXHistory, meanTransX);
    double stdDevTransY = calculateStandardDeviation(transYHistory, meanTransY);

    // Điều chỉnh Q và R dựa trên độ lệch chuẩn
    Q_scaleX = stdDevScaleX * 0.01;
    Q_scaleY = stdDevScaleY * 0.01;
    Q_thetha = stdDevThetha * 0.01;
    Q_transX = stdDevTransX * 0.01;
    Q_transY = stdDevTransY * 0.01;

    R_scaleX = stdDevScaleX * 0.1;
    R_scaleY = stdDevScaleY * 0.1;
    R_thetha = stdDevThetha * 0.1;
    R_transX = stdDevTransX * 0.1;
    R_transY = stdDevTransY * 0.1;
}

 
void computeAndApplyAffineTransformation( Mat& img1,  Mat& img2,  std::vector<KeyPoint>& keypoints1,
     std::vector<KeyPoint>& keypoints2,  std::vector<DMatch>& matches, Mat &stabilized_frame){
    
    std::vector<Point2f> pts1, pts2;
    std::vector<DMatch> good_matches;


    for ( auto& match : matches) {
        pts1.push_back(keypoints1[match.queryIdx].pt);
        pts2.push_back(keypoints2[match.trainIdx].pt);
    }
    vector <uchar> status;
    vector <float> err;
    vector <Point2f> prev_corner2, cur_corner2;
    //Mat gray1, gray2;
    //cv::cvtColor(img1, gray1, COLOR_BGR2GRAY);
    //cv::cvtColor(img2, gray2, COLOR_BGR2GRAY);

    calcOpticalFlowPyrLK(gray1, gray2, pts1, pts2, status, err);

    // weed out bad matches
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            prev_corner2.push_back(pts1[i]);
            cur_corner2.push_back(pts2[i]);
        }
    }

    // translation + rotation only
    auto start2 = std::chrono::high_resolution_clock::now();
    //Mat affine_transform = estimateRigidTransform(prev_corner2, cur_corner2, false);
    Mat affine_transform = cv::estimateAffine2D(prev_corner2, cur_corner2);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed2 = end2 - start2;
    cout << "Thoi gian uoc luong ma tran (CPU): " << elapsed2.count() << " ms" << endl;

    //Mat affine_transform = cv::estimateAffinePartial2D(prev_corner2, cur_corner2);

    //cout <<"Du doan: " << affine_transform << endl;
    double dx = affine_transform.at<double>(0, 2);
    double dy = affine_transform.at<double>(1, 2);
    double da = atan2(affine_transform.at<double>(1, 0), affine_transform.at<double>(0, 0));
    double ds_x = affine_transform.at<double>(0, 0) / cos(da);
    double ds_y = affine_transform.at<double>(1, 1) / cos(da);

    new_dx = dx; new_dy = dy; new_da = da;

    if (original_x.size() < 300) {
        original_x.push_back(new_dx - old_dx);
        old_dx = new_dx;
        original_y.push_back(new_dy - old_dy);
        old_dy = new_dy;
        original_angles.push_back(new_da - old_da);
        old_da = new_da;
    }

    
    ds_x = std::max(0.5, std::min(ds_x, 1.0));
    ds_y = std::max(0.5, std::min(ds_y, 1.0));
    dx = std::max(-50.0, std::min(dx, 50.0));
    dy = std::max(-50.0, std::min(dy, 50.0));

    double sx = ds_x;
    double sy = ds_y;

    sum_transX += dx;
    sum_transY += dy;
    sum_thetha += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;

    if (k == 1){
        k++;
    }
    else{
        Kalman_Filter(&scaleX, &scaleY, &thetha, &transX, &transY);

    }

    double diff_scaleX = scaleX - sum_scaleX;
    double diff_scaleY = scaleY - sum_scaleY;
    double diff_transX = transX - sum_transX;
    double diff_transY = transY - sum_transY;
    double diff_thetha = thetha - sum_thetha;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_thetha;
    // Giới hạn các giá trị đã làm mịn
    ds_x = std::max(0.5, std::min(ds_x, 1.0));
    ds_y = std::max(0.5, std::min(ds_y, 1.0));
    dx = std::max(-50.0, std::min(dx, 50.0));
    dy = std::max(-50.0, std::min(dy, 50.0));

    stab_new_dx = dx; stab_new_dy = dy; stab_new_da = da;

    // Log the stabilized rotation angle
    if (stabilized_x.size() < 300) {
        stabilized_x.push_back(stab_new_dx - stab_old_dx);
        stab_old_dx = stab_new_dx;
        stabilized_y.push_back(stab_new_dy - stab_old_dy);
        stab_old_dy = stab_new_dy;
        stabilized_angles.push_back(stab_new_da - stab_new_da);
        stab_old_da = stab_new_da;
    }

    //Creating the smoothed parameters matrix
    affine_transform.at<double>(0, 0) = sx * cos(da);
    affine_transform.at<double>(0, 1) = sx * -sin(da);
    affine_transform.at<double>(1, 0) = sy * sin(da);
    affine_transform.at<double>(1, 1) = sy * cos(da);
    affine_transform.at<double>(0, 2) = dx;
    affine_transform.at<double>(1, 2) = dy;

    //cout << "Phan hoi" << affine_transform << endl;

    
    
    //Rect roi1(0, 0, img1.cols, img1.rows);
    d_img1.upload(img1);

    // Áp dụng ma trận affine bằng CUDA
    Size dsize(img2.cols, img2.rows);
    auto start = std::chrono::high_resolution_clock::now();

    try {
        cv::cuda::warpAffine(d_img1, d_stabilized_frame, affine_transform, dsize, INTER_LINEAR, BORDER_CONSTANT, Scalar(), stream);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error applying warpAffine on GPU: " << e.what() << std::endl;
        return;
    }
    //stream.waitForCompletion();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    cout << "Thoi gian apply affine matrix (GPU): " << elapsed.count() << " ms" << endl;

    d_stabilized_frame.download(stabilized_frame, stream);
    //stabilized_frame = resizeAndCropFrame(stabilized_frame, 1.2);
    stream.waitForCompletion();
}

// Function to process video frames using ORB and affine transformation
void processVideoFrames(VideoCapture& capture) {

    Mat frame1, frame2;
    Mat croppedFrame1, croppedFrame2;

    if (!capture.read(frame1)) {
        cerr << "Failed to read the first frame." << endl;
        return;
    }
    //cv::flip(frame1, frame1, 1);

    if(frame1.cols >= 1920 && frame1.rows >= 1080){
            int x = frame1.cols/ 4; // Tọa độ x bắt đầu
            int y = frame1.rows/ 4;  // Tọa độ y bắt đầu

            // Cắt ảnh
            cv::Rect roi1(x, y, x* 2, y* 2);
            croppedFrame1 = frame1(roi1);
    }
    else croppedFrame1 = frame1.clone();

    // Keypoints and descriptors
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    while (true) {
        // Read the next frame
        auto start2 = std::chrono::high_resolution_clock::now();

        if (!capture.read(frame2)) {
            break; // End of video
        }
        if(frame2.cols >= 1920 && frame2.rows >= 1080){
            int x = frame2.cols/ 4; // Tọa độ x bắt đầu
            int y = frame2.rows/ 4;  // Tọa độ y bắt đầu

            // Cắt ảnh
            cv::Rect roi2(x, y, x* 2, y* 2);
            croppedFrame2 = frame2(roi2);
        }
        else croppedFrame2 = frame2.clone();
        

        // Check if frames are empty
        if (frame1.empty() || frame2.empty()) {
            cerr << "Empty frame encountered." << endl;
            break;
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end2 - start2;
        cout << "Thoi gian doc frame tu camera (CPU): " << elapsed.count() << " ms" << endl;

        vector<DMatch> matches;
        Mat stabilized_frame;
        auto start = getTickCount();
        ORB_GPU(croppedFrame1, croppedFrame2, keypoints1, keypoints2, descriptors1, descriptors2, matches);
        computeAndApplyAffineTransformation(frame1, frame2, keypoints1, keypoints2, matches, stabilized_frame);

        auto end = getTickCount();
        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;
        //cout << "FPS: " << fps << endl;
        putText(stabilized_frame, "FPS: " + to_string(int(fps)), Point(50, 50),
            FONT_HERSHEY_DUPLEX, 1, Scalar(255, 0, 0), 1);

        //cv::resize(stabilized_frame, stabilized_frame, stabilized_frame.size() / 2);
        Mat frame;
        //cv::resize(frame1, frame1, frame1.size() / 2);
        cv::resize(stabilized_frame, stabilized_frame, stabilized_frame.size() / 2);
        //hconcat(frame1, stabilized_frame, frame);
        //imshow("Combined Video", frame);
        imshow("Stabilized Video", stabilized_frame); // Display the stabilized frame

        if (waitKey(30) >= 0) {
            //imshow("Video", frame1);
            //imshow("Stabilized Video", stabilized_frame);
            //waitKey(0);
            break;
        }

        // Move to the next frame
        frame1 = frame2.clone();
        gray1 = gray2.clone();
        keypoints1.clear();
        keypoints1.shrink_to_fit();
        keypoints1 = keypoints2;
        descriptors2.copyTo(descriptors1);
        keypoints2.clear();
        keypoints2.shrink_to_fit();
    }
}


void plotGraph(std::vector<double>& original_angles, std::vector<double>& stabilized_angles, string &s);

void writeVectorToFile(const std::string& filename, const std::vector<double>& data) {
    // Mở tệp tin với chế độ ghi ('w') để xóa nội dung hiện tại
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Không thể mở tệp tin để ghi.");
    }

    // Ghi dữ liệu vào tệp tin
    for (const auto& value : data) {
        file << value << '\n';
    }

    file.close();
}

int main() {
    // Open video file
    //VideoCapture capture("Videos/Extremely_Shaky_Video.mp4");
    VideoCapture capture("Videos/Shaky_Video.mp4");
    //VideoCapture capture("Videos/ManhCuonglac.mp4");
    //VideoCapture capture("Videos/shaky video.mp4");
    //VideoCapture capture("Videos/animal_tracking_food_in_winter.mp4");
    //VideoCapture capture("Videos/outdoor_with_raining.mp4");
    //VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    
    //capture.set(CAP_PROP_FRAME_WIDTH, 960); // Chiều rộng khung hình
    //capture.set(CAP_PROP_FRAME_HEIGHT, 640); // Chiều cao khung hình
    Mat frame;
    
    processVideoFrames(capture);

    
    writeVectorToFile("Files/original_angles.txt", original_angles);
    writeVectorToFile("Files/original_x.txt", original_x);
    writeVectorToFile("Files/original_y.txt", original_y);
    writeVectorToFile("Files/stabilized_angles.txt", stabilized_angles);
    writeVectorToFile("Files/stabilized_x.txt", stabilized_x);
    writeVectorToFile("Files/stabilized_y.txt", stabilized_y);


    return 0;
}

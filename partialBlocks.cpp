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
#include <deque> 
#include <chrono>
const int window_size = 5;
const int blockNum = 4;
//std::vector<double> original_angles, original_x, original_y;
//std::vector<double> stabilized_angles, stabilized_x, stabilized_y;

// const float alpha = 0.5; // Hệ số downsampling
std::deque<cv::Vec2d> shift_history;

using namespace std;
using namespace cv;

double old_dx, old_dy, old_da;
double new_dx, new_dy, new_da;

double stab_old_dx, stab_old_dy, stab_old_da;
double stab_new_dx, stab_new_dy, stab_new_da;

std::vector<double> original_angles, original_x, original_y;
std::vector<double> stabilized_angles, stabilized_x, stabilized_y;

struct BlockInfo {
    double reliability;
    cv::Rect region;
};

// So sánh để sắp xếp các block theo độ tin cậy
bool compareReliability(const BlockInfo& a, const BlockInfo& b) {
    return a.reliability > b.reliability;
}
// Tính toán độ tin cậy của các khối dựa trên công thức peak analyses
double compute_reliability(const cv::Mat& acm) {
    // Lấy Rc max (giá trị tại trung tâm của ACM)
    double Rc_max = acm.at<float>(acm.rows / 2, acm.cols / 2);

    // Tìm Roc max (giá trị ngoài trung tâm)
    double Roc_max = 0.0;
    //
    for (int i = 0; i < acm.rows; i++) {
        for (int j = 0; j < acm.cols; j++) {
            if (!(i == acm.rows / 2 && j == acm.cols / 2)) {  // Bỏ qua đỉnh trung tâm
                Roc_max = max(Roc_max, (double)acm.at<float>(i, j));
            }
        }
    }

    // Tìm giá trị nhỏ nhất (Rmin) trong ACM
    double Rmin;
    minMaxLoc(acm, &Rmin, nullptr);
    //auto start = std::chrono::high_resolution_clock::now();
    // Tính toán độ tin cậy theo công thức
    double reliability = sqrt((Rc_max - Rmin) * (Rc_max - Roc_max) / Rc_max);
    /*auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    cout << "Thoi gian tinh toan do tin cay (CPU): " << elapsed.count() << " ms" << endl;*/
    return reliability;
}

void showImg(cv::Mat img, const std::string name) {
    cv::namedWindow(name.c_str());
    cv::imshow(name.c_str(), img);
}

void expand_img_to_optimal(cv::Mat& padded, cv::Mat& img) {
    int row = cv::getOptimalDFTSize(img.rows);
    int col = cv::getOptimalDFTSize(img.cols);
    cv::copyMakeBorder(img, padded, 0, row - img.rows, 0, col - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

void crop_and_rearrange(cv::Mat& magI) {
    magI = magI(cv::Rect(0, 0, magI.cols, magI.rows));
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

cv::Mat fourier_transform_subblock(cv::Mat& img, int block_index) {
    cv::Mat padded;

    expand_img_to_optimal(padded, img);

    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // Tính FFT
    cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);

    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    crop_and_rearrange(magI);

    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

    return magI;
}

std::vector<BlockInfo> block_info;

void compute_fft_for_subblocks(cv::Mat& img, int block_size, Mat& output_img) {
    int rows = img.rows;
    int cols = img.cols;
    int block_index = 0;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rows; i += block_size) {
        for (int j = 0; j < cols; j += block_size) {
            Mat img_cpy = img.clone();

            cv::Rect block_region(j, i, block_size, block_size);
            cv::Mat sub_block = img_cpy(block_region);

            cv::Mat fft_block = fourier_transform_subblock(sub_block, block_index);
            block_index++;
            auto start = std::chrono::high_resolution_clock::now();
            double reliability = compute_reliability(fft_block);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapse = end - start;

            if (!isnan(reliability) && reliability >= 0) {
                block_info.push_back({ reliability, block_region });
            }

            fft_block.copyTo(output_img(block_region));
        }

    }

    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;

    cout << "Thoi gian tinh toan FFT cho tung block (CPU): " << elapsed2.count() << " ms" << endl;

    std::sort(block_info.begin(), block_info.end(), compareReliability);

    cout << "Top " << blockNum << " reliable blocks:" << endl;
    for (int i = 0; i < blockNum; ++i) {
        cout << "Block " << (i + 1) << " - Reliability: " << block_info[i].reliability
            << ", Coordinates: (" << block_info[i].region.x << ", " << block_info[i].region.y << ")"
            << ", Size: (" << block_info[i].region.width << ", " << block_info[i].region.height << ")" << endl;
        rectangle(output_img, block_info[i].region, Scalar(0, 0, 255), 2); // Vẽ khung đỏ
    }
    block_info.resize(blockNum);
}

cv::Vec2d computeWeightedShifts(const std::vector<BlockInfo>& block_info, const cv::Mat& img1, const cv::Mat& img2) {
    double weighted_delta_x = 0.0;
    double weighted_delta_y = 0.0;
    double total_weight = 0.0;

    for (const auto& block : block_info) {
        cv::Rect region = block.region;
        // Chuyển đổi sub-block sang CV_32FC1
        cv::Mat sub_block1, sub_block2;
        img1(region).convertTo(sub_block1, CV_32FC1);
        img2(region).convertTo(sub_block2, CV_32FC1);

        cv::Point2d shift = phaseCorrelate(sub_block1, sub_block2);

        double response = block.reliability;

        weighted_delta_x += response * shift.x;
        weighted_delta_y += response * shift.y;
        total_weight += response;
    }

    double delta_x = total_weight > 0 ? weighted_delta_x / total_weight : 0;
    double delta_y = total_weight > 0 ? weighted_delta_y / total_weight : 0;

    return cv::Vec2d(delta_x, delta_y);

}

// Hàm để tính toán ma trận affine và biến đổi ảnh
cv::Mat applyAffineTransformation(const cv::Mat& img, double delta_x, double delta_y) {
    // Tạo ma trận affine 2x3
    cv::Mat affine_matrix = (cv::Mat_<double>(2, 3) << 1, 0, delta_x, 0, 1, delta_y);

    cv::Mat warped_img;
    auto start3 = std::chrono::high_resolution_clock::now();
    cv::warpAffine(img, warped_img, affine_matrix, img.size());
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
    cout << "Thoi gian ap dung bien doi affine (CPU): " << elapsed3.count() << " ms" << endl;
    return warped_img;
}

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

int main()
{
    // Mở video
    VideoCapture cap("Videos/Extremely_Shaky_Video.mp4");
    //VideoCapture cap("Videos/shaky video.mp4");
    //VideoCapture cap("Videos/Vietnam.mp4");
    //VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Could not open video" << endl;
        return -1;
    }

    Mat frame, prev_frame;
    bool first_frame = true;
    int samples = 0;

    while (true) {
        auto start6 = std::chrono::high_resolution_clock::now();
        cap >> frame; // Đọc khung hình tiếp theo
        if (frame.empty()) break; // Nếu không còn khung hình thì dừng


        //resize
        cv::resize(frame, frame, Size(540, 540));

        // Chuyển đổi khung hình sang grayscale
        cv::Mat gray_frame;
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        auto end6 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapse6 = end6 - start6;
        cout << "Thoi gian doc frame va grayscaling(CPU): " << elapse6.count() << " ms" << endl;
        cv::equalizeHist(gray_frame, gray_frame);
        auto start5 = std::chrono::high_resolution_clock::now();
        Canny(gray_frame, gray_frame, 100, 200);
        auto end5 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapse = end5 - start5;
        cout << "Thoi gian tinh toan va phat hien canh (CPU): " << elapse.count() << " ms" << endl;
        if (first_frame) {
            prev_frame = gray_frame.clone();
            first_frame = false;
            continue; // Bỏ qua khung đầu tiên
        }

        // Kích thước block 54
        int block_size = 108;
        auto start = getTickCount();
        // Tạo ảnh rỗng cho ảnh đã ổn định
        Mat output_img(gray_frame.rows, gray_frame.cols, CV_32F, Scalar(0));

        // Chia ảnh thành các khối và tính FFT cho từng khối
        compute_fft_for_subblocks(gray_frame, block_size, output_img);

        // Tính dịch chuyển giữa các khung hình
        cv::Vec2d total_shift = computeWeightedShifts(block_info, prev_frame, gray_frame);
        cout << "Dịch chuyển tổng (delta x, delta y): (" << total_shift[0] << ", " << total_shift[1] << ")" << endl;

        new_dx = total_shift[0];
        new_dy = total_shift[1];
        if(samples < 300){
            original_x.push_back(new_dx);
            original_y.push_back(new_dy);
        }

        // Lưu vào lịch sử dịch chuyển
        shift_history.push_back(total_shift);
        if (shift_history.size() > window_size) {
            shift_history.pop_front();  // Giữ kích thước lịch sử bằng kích thước cửa sổ
        }

        // Tính trung bình dịch chuyển
        // Khai báo và tính trung bình dịch chuyển
        double smoothed_shift_x = 0;
        double smoothed_shift_y = 0;

        if (!shift_history.empty()) {
            for (const auto& shift : shift_history) {
                smoothed_shift_x += shift[0];  // Cộng dịch chuyển x
                smoothed_shift_y += shift[1];  // Cộng dịch chuyển y
            }
            // Chia từng kênh cho kích thước của shift_history
            smoothed_shift_x /= shift_history.size();
            smoothed_shift_y /= shift_history.size();
        }
        else {
            // Nếu shift_history rỗng, giữ nguyên giá trị cũ hoặc gán giá trị mặc định
            smoothed_shift_x = total_shift[0];
            smoothed_shift_y = total_shift[1];
        }
        
        stab_new_dx = smoothed_shift_x;
        stab_new_dy = smoothed_shift_y;
        if(samples < 300){
            stabilized_x.push_back(stab_new_dx - stab_old_dx);
            stabilized_y.push_back(stab_new_dy - stab_old_dy);
            stab_old_dx = stab_new_dx;
            stab_old_dy = stab_new_dy;
        }

        // Áp dụng biến đổi affine
        Mat stabilized_frame = applyAffineTransformation(frame, -smoothed_shift_x, -smoothed_shift_y);
        auto end = getTickCount();
        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;
        putText(stabilized_frame, "fps: " + to_string(int(fps)), Point(50, 50),
            FONT_HERSHEY_DUPLEX, 1, Scalar(255, 0, 0), 1);
        // Hiển thị khung hình đã ổn định
        //imshow("Stabilized Frame", stabilized_frame);
        // Tạo một ảnh lớn để chứa cả frame gốc và frame đã ổn định
        Mat combined_frame(frame.rows, frame.cols * 2, frame.type());
        frame.copyTo(combined_frame(Rect(0, 0, frame.cols, frame.rows)));  // Copy frame gốc vào bên trái
        stabilized_frame.copyTo(combined_frame(Rect(frame.cols, 0, frame.cols, frame.rows)));  // Copy frame đã ổn định vào bên phải

        imshow("Original & Stabilized Frame", combined_frame);

        prev_frame = gray_frame.clone();
        block_info.clear();
        block_info.shrink_to_fit();
        frame = stabilized_frame.clone();
        //pre_fr = stabilized_frame.clone();

        if (waitKey(30) == 'q') break;

        // Ghi các chỉ số vào tệp tin
        //writeVectorToFile("Files/original_angles.txt", original_angles);
        writeVectorToFile("Files/original_x.txt", original_x);
        writeVectorToFile("Files/original_y.txt", original_y);
        //writeVectorToFile("Files/stabilized_angles.txt", stabilized_angles);
        writeVectorToFile("Files/stabilized_x.txt", stabilized_x);
        writeVectorToFile("Files/stabilized_y.txt", stabilized_y);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
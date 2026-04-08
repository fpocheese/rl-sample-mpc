#include <vector>
#include <chrono>
#include <cmath>
// -----------------gsy0422-------------------------------
#include <random>
// -----------------gsy0422-------------------------------

using high_prec_clock = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct DetectionResult
{
    high_prec_clock detection_time;
    double loc_x, loc_y , loc_A, loc_Vs;
    bool valid_result;
    DetectionResult()
    {
        valid_result = false;
        loc_x = loc_y = loc_A = loc_Vs = 0;
        detection_time = std::chrono::high_resolution_clock::now();
    }
    DetectionResult(high_prec_clock time, double loc_x_, double loc_y_ ,double loc_A_ ,double loc_Vs_) : detection_time(time), loc_x(loc_x_), loc_y(loc_y_), loc_A(loc_A_), loc_Vs(loc_Vs_) ,valid_result(true) {}
    bool is_timeout()
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = current_time - detection_time;
        auto delta_time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(delta_time).count();
        if (delta_time_sec > 2.0)
        {
            return true;
        }
        return false;
    }
    bool operator < (const DetectionResult& rhs)
    {
        if (valid_result != rhs.valid_result)
        {
            return valid_result;
        }
        return loc_x < rhs.loc_x;
    }

    double dist() {
        return std::sqrt(loc_x * loc_x + loc_y * loc_y);
    }
};



class TargetsDetectionStore
{
public:
    DetectionResult result;
    TargetsDetectionStore(double loc_x, double loc_y ,double loc_A, double loc_Vs, std::string mode) {
        if (mode == "simulator") {
            SetSimulatorNpcData(loc_x, loc_y ,loc_A, loc_Vs);
        }
        else if (mode == "real") {
            SetDetectionData(loc_x, loc_y ,loc_A, loc_Vs);
        }
        else if (mode == "lidar") {
            SetSimulatorNpcData(loc_x, loc_y ,loc_A, loc_Vs);
        }
        else {
            printf("Invalid mode: %s\n", mode.c_str());
        }

    }

    void output() {
        printf("Dist: %.2lf, x: %.2lf, y: %.2lf, A: %.2lf, Vs: %.2lf\n", result.dist(), result.loc_x, result.loc_y, result.loc_A, result.loc_Vs);
    }
    bool operator < (const TargetsDetectionStore& rhs)
    {
        return result < rhs.result;
    }

    bool SetSimulatorNpcData(double loc_x, double loc_y ,double loc_A, double loc_Vs)
    {
        // double mean = 0.0f;   // 均值
        // double stddev = 0.0f; // 标准差
        // // 生成高斯噪声
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::normal_distribution<double> noise(mean, stddev);
        // double noisy_loc_x = loc_x + noise(gen);
        // double noisy_loc_y = loc_y + noise(gen);
        // double noisy_loc_A = loc_A + noise(gen);
        // double noisy_loc_Vs = loc_Vs + noise(gen);

        double noisy_loc_x = loc_x ;
        double noisy_loc_y = loc_y ;
        double noisy_loc_A = loc_A ;
        double noisy_loc_Vs = loc_Vs ;

        if (loc_x <= -30 || loc_x >= 60 || fabs(loc_y) > 30)
        {
            // 如果超出范围，设置为无效结果
            return false;
        }

        if (result.dist() > 100 || loc_x > 100) {
            return false;
        }

        // // 随机决定是否检测失败
        // // 检测失败的概率
        // double detection_failure_prob = 0.2f; // 设定检测失败的概率为20%
        // std::uniform_real_distribution<double> dist(0.0f, 1.0f);
        // if (dist(gen) < detection_failure_prob)
        // {
        //     return false;
        // }

        // 更新结果
        result.loc_x = noisy_loc_x;
        result.loc_y = noisy_loc_y;
        result.loc_A = noisy_loc_A;
        result.loc_Vs = noisy_loc_Vs;
        result.detection_time = std::chrono::high_resolution_clock::now();
        result.valid_result = true;
        return true;
    }
    void SetDetectionData(double loc_x, double loc_y,double loc_A, double loc_Vs)
    {
        // if (loc_x <= 0)
        // {
        //     // if the car is back of us, ignore
        //     return;
        // }
        auto dis = std::sqrt(loc_x * loc_x + loc_y * loc_y);
        // if (dis > 500)
        // {
        //     // bad value detected
        //     return;
        // }
        result.loc_x = loc_x;
        result.loc_y = loc_y;
        result.loc_A = loc_A;
        result.loc_Vs = loc_Vs;
        result.detection_time = std::chrono::high_resolution_clock::now();
        result.valid_result = true;
    }
    bool GetDistance(double &out_distance)
    {
        if (!result.valid_result)
        {
            return false;
        }
        if (result.is_timeout())
        {
            return false;
        }
        out_distance = result.dist();
        return true;
    }
    bool GetCordinate(double &out_x, double &out_y ,double &out_A, double &out_Vs)
    {
        if (!result.valid_result)
        {
            return false;
        }
        if (result.is_timeout())
        {
            return false;
        }
        out_x = result.loc_x;
        out_y = result.loc_y;
        out_A = result.loc_A;
        out_Vs = result.loc_Vs;
        return true;
    }
};
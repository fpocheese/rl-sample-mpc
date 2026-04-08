#include <math.h>

#include <cstdarg>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>

struct CARdata_FOLLOW
{
    int N;                   /* Data Dimension */
    std::vector<double> V;   /* Velocity (m/s) */
    std::vector<double> An;  /* Normal Acc of Tyre (m/s2) */
    std::vector<double> Aw;  /* Tangential Acc of Tyre (m/s2) */
    std::vector<double> Ae0; /* Tangential Acc of Engine (PTP off) (m/s2) */
    std::vector<double> Ae1; /* Tangential Acc of Engine (PTP on) (m/s2) */
    double Mass;             /* Mass (kg) */
    double CD;               /* Drag Coefficient: Drag = CD*vel*vel */
};


class PIController
{
public:
    double kp;
    double ki;
    double kd;// 新增导数增益 xhy_create
    double cumulative_error;
    double max_cumulative_error;
    double last_error;
    
    // 添加时间戳成员变量
    std::chrono::steady_clock::time_point last_time_point;
    bool first_call;  // 标记是否是第一次调用

public:
    std::shared_ptr<CARdata_FOLLOW> Vehicle = std::make_shared<CARdata_FOLLOW>();
    bool PTP_on = false; // 新增PTP状态标识    

    /* =============== MyInterp.c =============== */
    double Interp1(std::vector<double> X, std::vector<double> Y, int N, double x)
    {
        double V1, V2, A1, A2;
        if (x <= X[0])
        {
            return Y[0];
        }
        else if (x >= X[N - 1])
        {
            return Y[N - 1];
        }
        else
        {
            for (int i = 1; i < N; ++i)
            {
                if (x < X[i])
                {
                    V1 = X[i - 1];
                    V2 = X[i];
                    A1 = Y[i - 1];
                    A2 = Y[i];
                    return A1 + (A2 - A1) * (x - V1) / (V2 - V1);
                }
            }
            return Y[N - 1];
        }
    }

    /* =============== COMMON Lib =============== */
    int ReadVehicle(int lenCarData, const std::string &filename)
    {
        /* ==================== Read CarData ==================== */
        Vehicle->N = lenCarData;
        Vehicle->V = std::vector<double>(Vehicle->N);
        Vehicle->An = std::vector<double>(Vehicle->N);
        Vehicle->Aw = std::vector<double>(Vehicle->N);
        Vehicle->Ae0 = std::vector<double>(Vehicle->N);
        Vehicle->Ae1 = std::vector<double>(Vehicle->N);
        Vehicle->Mass = 742;
        Vehicle->CD = 0.75;

        std::ifstream file(filename);

        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        std::string line;

        // 读取第一行，跳过标题
        std::getline(file, line);

        // 读取 1 - count < Vehicle->N 行数据
        int count = 0;
        while (std::getline(file, line) && count < Vehicle->N)
        {
            std::istringstream iss(line);
            std::string token;

            std::getline(iss, token, ',');
            Vehicle->V[count] = std::stod(token);

            std::getline(iss, token, ',');
            Vehicle->An[count] = std::stod(token);

            std::getline(iss, token, ',');
            Vehicle->Aw[count] = std::stod(token);

            std::getline(iss, token, ',');
            Vehicle->Ae0[count] = std::stod(token);

            std::getline(iss, token, ',');
            Vehicle->Ae1[count] = std::stod(token);

            count++;
        }

        file.close();

        return 0;
    }

    PIController()
        : cumulative_error(0.0), last_error(0.0), first_call(true) {}

    // double update(double v_actual, double error)
    // {
    //     // 获取当前时间
    //     auto current_time = std::chrono::steady_clock::now();
        
    //     // 计算真实dt
    //     double dt;
    //     if (first_call) {
    //         // 第一次调用，使用默认值
    //         dt = 0.01;
    //         first_call = false;
    //     } else {
    //         // 计算时间差并转换为秒
    //         std::chrono::duration<double> time_diff = current_time - last_time_point;
    //         dt = time_diff.count();
            
    //         // 安全检查：防止dt过小或过大
    //         if (dt < 0.001) dt = 0.001;  // 防止除以接近零的值
    //         if (dt > 0.1) dt = 0.01;     // 如果时间差异太大，可能是长时间未调用，使用默认值
    //     }
        
    //     // 更新时间戳
    //     last_time_point = current_time;
        
    //     // 非线性误差变换
    //     double tan_w = 10.0;
    //     if (error>0)
    //     {
    //         tan_w = 20.0;
    //     }
    //     double clamped_error = std::max(-M_PI/2 + 1e-3, 
    //                         std::min(M_PI/2 - 1e-3, 
    //                                 error * M_PI/2 * (1.0/tan_w)));
    //     double transformed_error = 10 * std::tan(clamped_error);
    //     // double transformed_error = error;
    //     double kp_dynamic = kp;
    //     double kd_dynamic = kd;
        
    //     // if (std::abs(error) > 10) {
    //     //     // 距离差距大，更积极响应
    //     //     kp_dynamic = kp * 1.2;
    //     //     kd_dynamic = kd * 0.8;
    //     // } else if (std::abs(error) <= 5) {
    //     //     // 接近目标距离，更保守响应
    //     //     kp_dynamic = kp * 0.8;
    //     //     kd_dynamic = kd * 1.2;
    //     // }
    //     // Calculate proportional term (使用变换后的error)
    //     double p_term = kp_dynamic * transformed_error;

    //     // Calculate integral term
    //     // double i_term = ki * cumulative_error;

    //     // // Calculate control input as the sum of P and I terms
    //     // double a_control_input = (p_term + i_term) / dt;

    //     // // 动态计算a_max和a_min
    //     // double Aw_current = Interp1(Vehicle->V, Vehicle->Aw, Vehicle->N, v_actual);
    //     // double Ae_current;
    //     // if (PTP_on)
    //     // {
    //     //     Ae_current = Interp1(Vehicle->V, Vehicle->Ae1, Vehicle->N, v_actual);
    //     // }
    //     // else
    //     // {
    //     //     Ae_current = Interp1(Vehicle->V, Vehicle->Ae0, Vehicle->N, v_actual);
    //     // }
    //     // double current_a_max = std::min(Aw_current, Ae_current);
    //     // double current_a_min = -Aw_current;

    //     // // 你可以看出来，这里我想计算一个control_input是速度的改变量，但是我还想确保达到这个速度的加速度不会超过a_max和a_min的范围，请帮我在下面写出这个代码。
    //     // a_control_input = std::max(current_a_min, std::min(a_control_input, current_a_max));
    //     // double vel_input = a_control_input * dt;

    //     //double output_vel = (p_term + i_term);
    //     // xhy create
    //     double derivative = (error - last_error) / dt;
    //     double d_term = kd_dynamic * derivative;
    //     last_error = error;
        
    //     // 计算输出
    //     double output_vel = p_term + d_term;
    //     // xhy create
    //     // 设置逻辑，如果error>50,输出2倍，如果30<error<50,输出1倍，如果error<30,输出0.5倍
    //     // if (error < -15 )
    //     // {
    //     //     output_vel = 6 * output_vel;
    //     // }
    //     // else if (error < -10)
    //     // {
    //     //     output_vel = 4 * output_vel;
    //     // }
    //     // else if (error > 15)
    //     // {
    //     //     output_vel = 2 * output_vel;
    //     // }
    //     //xhy create
    //     return output_vel;
    // }

    double update(double v_actual, double error)
    {
        // 获取当前时间
        auto current_time = std::chrono::steady_clock::now();

        // 计算真实dt
        double dt;
        if (first_call)
        {
            // 第一次调用，使用默认值
            dt = 0.01;
            first_call = false;
        }
        else
        {
            // 计算时间差并转换为秒
            std::chrono::duration<double> time_diff = current_time - last_time_point;
            dt = time_diff.count();

            // 安全检查：防止dt过小或过大
            if (dt < 0.001)
                dt = 0.001; // 防止除以接近零的值
            if (dt > 0.1)
                dt = 0.01; // 如果时间差异太大，可能是长时间未调用，使用默认值
        }

        // 更新时间戳
        last_time_point = current_time;

        // 非线性误差变换（修改：根据error符号调整tan_w，使负error更保守）
        double tan_w = 10.0;
        if (error > 0)
        {
            tan_w = 10.0; // 正error（太远）：较小tan_w，更激进响应以追赶
        }
        else
        {
            tan_w = 30.0; // 负error（太近）：更大tan_w，更温和响应避免猛刹
        }
        double clamped_error = std::max(-M_PI / 2 + 1e-3,
                                        std::min(M_PI / 2 - 1e-3,
                                                 error * M_PI / 2 * (1.0 / tan_w)));
        double transformed_error = 10 * std::tan(clamped_error);
        // double transformed_error = error;

        // 动态KP/KD调整（修改：解开并优化阈值，小error时减小KP增大KD，确保近距离稳定）
        double kp_dynamic = kp;
        double kd_dynamic = kd;

        if (std::abs(error) > 10)
        {
            // 距离差距大，更积极响应（激进P，稍弱D）
            kp_dynamic = kp * 1.5;
            kd_dynamic = kd * 0.5;
        }
        else if (std::abs(error) <= 5)
        {
            // 接近目标距离，更保守响应（弱P，强D以阻尼，避免振荡或猛刹）
            kp_dynamic = kp * 0.5;
            kd_dynamic = kd * 1.5;
        }
        else
        {
            // 中间范围，保持默认（平滑过渡）
            kp_dynamic = kp * (0.5 + (std::abs(error) - 5) / 5.0); // 线性插值，避免跳变
            kd_dynamic = kd * (1.5 - (std::abs(error) - 5) / 5.0 * 0.7);
        }

        // Calculate proportional term (使用变换后的error)
        double p_term = kp_dynamic * transformed_error;

        // xhy create
        double derivative = (error - last_error) / dt;
        double d_term = kd_dynamic * derivative;
        last_error = error;

        // 计算输出
        double output_vel = p_term + d_term;

        return output_vel;
    }
};
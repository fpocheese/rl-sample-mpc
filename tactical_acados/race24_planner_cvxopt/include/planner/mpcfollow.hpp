#ifndef MPC_FOLLOWING_HPP
#define MPC_FOLLOWING_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ecos/include/ecos.h"
#include <vector>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <algorithm>

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

class MPCFollowing
{
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

    // 构造函数：初始化MPC参数
    MPCFollowing(int horizon = 10, double dt = 0.025)
        : N(horizon), dt(dt), v_max(70.0), Q_d(100.0), Q_v(10.0), R_a(1.0) {}

   
    double compute(double v_actual, double d_actual, double d_ref)
    {
        // 动态计算a_max和a_min
        double Aw_current = Interp1(Vehicle->V, Vehicle->Aw, Vehicle->N, v_actual);
        double Ae_current;
        if (PTP_on)
        {
            Ae_current = Interp1(Vehicle->V, Vehicle->Ae1, Vehicle->N, v_actual);
        }
        else
        {
            Ae_current = Interp1(Vehicle->V, Vehicle->Ae0, Vehicle->N, v_actual);
        }
        double current_a_max = std::min(Aw_current, Ae_current);
        double current_a_min = -current_a_max;

        int nx = 2; // 状态变量: [d, v]
        int nu = 1; // 控制变量: [a]

        // 状态转移矩阵
        Eigen::MatrixXd A(nx, nx);
        A << 1, dt, 0, 1;

        Eigen::MatrixXd B(nx, nu);
        B << 0, dt;

        // 优化目标
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N * nu, N * nu);
        Eigen::VectorXd f = Eigen::VectorXd::Zero(N * nu);

        // Set up H (quadratic cost matrix)
        for (int i = 0; i < N; i++)
        {
            H(i, i) = R_a;
        }

        // 约束矩阵
        Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(N * nx, N * nu);
        Eigen::VectorXd beq = Eigen::VectorXd::Zero(N * nx);

        Eigen::VectorXd x0(nx);
        x0 << d_actual - d_ref, v_actual;

        for (int i = 0; i < N; i++)
        {
            Aeq.block(i * nx, i * nu, nx, nu) = -B;
            if (i == 0)
                beq.segment(i * nx, nx) = A * x0;
            else
                Aeq.block(i * nx, (i - 1) * nu, nx, nu) = A;
        }

        // ECOS 变量
        int num_ineq = 2 * N; // inequality constraints (upper and lower bounds for acceleration)

        std::vector<pfloat> G_data(num_ineq, 1.0); // 注意改为pfloat类型
        std::vector<pfloat> h_data(num_ineq);      // 注意改为pfloat类型
        std::vector<pfloat> C_data(N * nx * N);    // 等式约束矩阵
        std::vector<pfloat> d_data(N * nx);        // 等式约束向量

        // 填充不等式约束数据（改为pfloat类型）
        for (int i = 0; i < N; ++i)
        {
            G_data[i] = 1.0;           // 上界约束系数
            h_data[i] = current_a_max; // 上界值
        }
        for (int i = N; i < 2 * N; ++i)
        {
            G_data[i] = -1.0;           // 下界约束系数
            h_data[i] = -current_a_min; // 下界值
        }

        // 填充等式约束数据
        for (int i = 0; i < N * nx; ++i)
        {
            C_data[i] = static_cast<pfloat>(Aeq(i / nu, i % nu));
            d_data[i] = static_cast<pfloat>(beq(i));
        }

        // 创建稀疏矩阵参数（这里简化为密集矩阵转换）
        std::vector<pfloat> Gpr = G_data;        // 假设G是密集矩阵
        std::vector<idxint> Gir(num_ineq);       // 行索引
        std::vector<idxint> Gjc = {0, num_ineq}; // 列指针

        std::vector<pfloat> Apr = C_data;      // 假设A是密集矩阵
        std::vector<idxint> Air(N * nx);       // 行索引
        std::vector<idxint> Ajc = {0, N * nx}; // 列指针

        // 初始化索引（简化处理，实际应根据矩阵结构生成）
        std::iota(Gir.begin(), Gir.end(), 0);
        std::iota(Air.begin(), Air.end(), 0);

        // Declare c as an Eigen vector
        Eigen::VectorXd c(N * nu);
        c.setZero(); // Initialize it with zeros (or set the proper coefficients)
        for (int i = 0; i < N; i++)
        {
            c(i) = R_a; // Set cost related to acceleration for each time step
        }

        // ECOS 参数设置
        pwork *mywork = ECOS_setup(
            /* n = */ N,             // 优化变量数
            /* m = */ num_ineq,      // 不等式约束数
            /* p = */ N * nx,        // 等式约束数
            /* l = */ 0,             // 指数锥约束数
            /* ncones = */ 0,        // 二阶锥数量
            /* q = */ NULL,          // 锥尺寸数组
            /* nex = */ 0,           // 扩展锥数量
            /* Gpr = */ Gpr.data(),  // 不等式矩阵数据
            /* Gjc = */ Gjc.data(),  // 列指针
            /* Gir = */ Gir.data(),  // 行索引
            /* Apr = */ Apr.data(),  // 等式矩阵数据
            /* Ajc = */ Ajc.data(),  // 等式列指针
            /* Air = */ Air.data(),  // 等式行索引
            /* c = */ c.data(),      // 目标函数系数
            /* h = */ h_data.data(), // 不等式约束值
            /* b = */ d_data.data()  // 等式约束值
        );

        // Solve the problem
        ECOS_solve(mywork);

        // Extract solution for acceleration command
        double a_cmd = mywork->x[0];
        a_cmd = std::max(current_a_min, std::min(a_cmd, current_a_max)); // Apply constraints

        // Clean up ECOS solver
        ECOS_cleanup(mywork, 0);

        // Compute the desired velocity
        double v_cmd = v_actual + a_cmd * dt;
        return std::max(0.0, std::min(v_cmd, v_max));
    }

private:
    int N;        // 预测时域
    double dt;    // 时间步长
    double v_max; // 最大速度 (m/s)
    double Q_d;   // 距离误差权重
    double Q_v;   // 速度平滑权重
    double R_a;   // 控制输入权重
};

#endif // MPC_FOLLOWING_HPP
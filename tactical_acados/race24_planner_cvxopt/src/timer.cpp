#include <sys/timerfd.h>
#include <unistd.h>
#include <inttypes.h> // For fixed-width integers
#include <fcntl.h>    // For setting non-blocking mode
#include <iostream>
#include <cstring> // For strerror
#include <chrono>
#include <fcntl.h> // For F_SETFL, O_NONBLOCK
#include <sys/epoll.h>
#include <fstream>
#include <ctime>
#include <thread>
#include <pthread.h>

namespace base_planner
{
    std::ofstream logFile;
    std::ofstream logStepFile;
    void writeLog(std::string tag = "default");
    void *timerThreadFunction(void *param)
    {
        int fd = timerfd_create(CLOCK_MONOTONIC, 0);
        if (fd == -1)
        {
            std::cerr << "Failed to create timer: " << std::strerror(errno) << std::endl;
            return 0;
        }

        itimerspec newTime;
        newTime.it_value.tv_sec = 0;                    // First expiration after 20ms
        newTime.it_value.tv_nsec = 20 * 1000 * 1000;    // 20ms
        newTime.it_interval.tv_sec = 0;                 // Repeat every 20ms
        newTime.it_interval.tv_nsec = 20 * 1000 * 1000; // 20ms

        if (timerfd_settime(fd, 0, &newTime, NULL) == -1)
        {
            std::cerr << "Failed to start timer: " << std::strerror(errno) << std::endl;
            close(fd);
            return 0;
        }

        int efd = epoll_create1(0);
        if (efd == -1)
        {
            std::cerr << "Failed to create epoll: " << std::strerror(errno) << std::endl;
            close(fd);
            return 0;
        }

        epoll_event event;
        event.events = EPOLLIN;
        event.data.fd = fd;
        if (epoll_ctl(efd, EPOLL_CTL_ADD, fd, &event) == -1)
        {
            std::cerr << "Failed to add fd to epoll: " << std::strerror(errno) << std::endl;
            close(efd);
            close(fd);
            return 0;
        }

        while (true)
        {
            epoll_event events[10];
            int n = epoll_wait(efd, events, 10, -1);
            for (int i = 0; i < n; i++)
            {
                if (events[i].data.fd == fd)
                {
                    uint64_t expirations;
                    read(fd, &expirations, sizeof(expirations)); // Clear the event
                    writeLog();
                }
            }
        }

        close(efd);
        close(fd);
    }
    void writeStepLog(std::string tag)
    {
        if (!logStepFile.is_open())
        {
            char *homeDir = getenv("HOME");
            if (homeDir == nullptr)
            {
                std::cerr << "Failed to get HOME directory." << std::endl;
                return;
            }

            // Construct the full path to the log file
            std::string logFilePath = std::string(homeDir) + "/planner_step_log.txt";

            // Open the file in append mode
            logStepFile.open(logFilePath, std::ios_base::app);
            if (!logStepFile.is_open())
            {
                std::cerr << "Failed to open log file: " << logFilePath << std::endl;
                return;
            }
        }
        auto now = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        // Write the log entry
        logStepFile << "[" << tag << "] step log: " << nanoseconds << "\r\n";
        logStepFile.flush();
    }
    void writeLog(std::string tag)
    {
        if (!logFile.is_open())
        {
            char *homeDir = getenv("HOME");
            if (homeDir == nullptr)
            {
                std::cerr << "Failed to get HOME directory." << std::endl;
                return;
            }

            // Construct the full path to the log file
            std::string logFilePath = std::string(homeDir) + "/planner_timer_log.txt";

            // Open the file in append mode
            logFile.open(logFilePath, std::ios_base::app);
            if (!logFile.is_open())
            {
                std::cerr << "Failed to open log file: " << logFilePath << std::endl;
                return;
            }
        }
        auto now = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        // Write the log entry
        logFile << "Log entry at " << nanoseconds << "\r\n";
        logFile.flush();
    }
    void install_log_timer()
    {
        // std::thread timerThread(timerThreadFunction);
        pthread_t t;
        pthread_create(&t, NULL, &timerThreadFunction, 0);
    }
}
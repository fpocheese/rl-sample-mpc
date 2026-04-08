#pragma once
#include <atomic>
#include <boost/asio.hpp>
#include <iostream>
#include <msgpack.hpp>
#include <mutex>
#include <thread>

// msgpack publisher
class MessageSender
{
public:
    MessageSender()
        : socket_(io_context_,
                  boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0))
    {
    }

    void start(std::string_view ip, const unsigned short &port)
    {
        std::cout << "start() called with IP: " << ip << " and Port: " << port << std::endl;
    
        if (!running_.exchange(true))
        {
            thread_ = std::jthread(&MessageSender::run, this);
        }
        broadcast_ip = ip;
        broadcast_port = port;
    }

    void append(std::string const &field, double value) { data[field] = value; }

    void stop()
    {
        running_.exchange(false);
        if (thread_.joinable())
        {
            thread_.join();
        }
    }

    void sendMessage()
    {
        std::scoped_lock lock{mutex_};

        // Pack some data into a MessagePack object with field names
        msgpack::sbuffer buffer;
        msgpack::packer<msgpack::sbuffer> pk(&buffer);
        pk.pack(data);

        // Create an endpoint to send the data to
        boost::asio::ip::udp::endpoint remote_endpoint(
            boost::asio::ip::address::from_string(broadcast_ip), broadcast_port);

        // Send the data
        socket_.send_to(boost::asio::buffer(buffer.data(), buffer.size()),
                        remote_endpoint);
        // std::cout << "msg sent!" << std::endl;
    }

private:
    void run() { io_context_.run(); }

    boost::asio::io_context io_context_{};
    boost::asio::ip::udp::socket socket_;
    std::atomic<bool> running_{false};
    std::jthread thread_;
    std::mutex mutex_;
    std::map<std::string, double, std::less<>> data;
    std::string broadcast_ip;
    unsigned short broadcast_port;
};

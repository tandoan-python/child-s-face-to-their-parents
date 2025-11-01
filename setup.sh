#!/bin/bash

Dọn dẹp bộ nhớ đệm và cập nhật hệ thống

echo "Bắt đầu thiết lập hệ thống (OpenCV dependencies)..."
sudo apt-get clean
sudo apt-get update

Các thư viện cơ bản cho DeepFace/NumPy/SciPy

echo "Cài đặt các thư viện toán học và biên dịch..."
sudo apt-get install -y libgfortran5 build-essential

CÁC THƯ VIỆN CỐT LÕI CHO OPENCV (libsm6, libxrender1, libfontconfig1, libice6 đã được thêm ở lần trước)

Bổ sung các thư viện media/codec và giao diện (UI/X-server) để khắc phục lỗi ImportError.

echo "Cài đặt các thư viện media và UI cho OpenCV..."
sudo apt-get install -y libsm6 libxrender1 libfontconfig1 libice6 libgtk2.0-dev libcanberra-gtk-module

Tạo thư mục ẩn để lưu trữ các mô hình DeepFace

echo "Kiểm tra và tạo thư mục ~/.deepface..."
mkdir -p ~/.deepface

echo "Thiết lập hệ thống hoàn tất. Bắt đầu cài đặt Python packages..."
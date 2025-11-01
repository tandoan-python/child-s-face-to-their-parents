#!/bin/bash

echo "Bắt đầu thiết lập hệ thống (Streamlit Cloud)..."
sudo apt-get update

Cài đặt các thư viện hỗ trợ OpenCV và DeepFace:

libgfortran5 và build-essential: Cần thiết cho các phép toán số học cấp cao (DeepFace/NumPy).

libsm6, libxrender1, libfontconfig1, libice6: Các thư viện hiển thị và X-server cần thiết cho OpenCV hoạt động trong môi trường headless Linux.

echo "Cài đặt các thư viện hỗ trợ cốt lõi (OpenCV, DeepFace)..."
sudo apt-get install -y libgfortran5 build-essential libsm6 libxrender1 libfontconfig1 libice6

Tạo thư mục ẩn để lưu trữ các mô hình DeepFace (giảm thiểu lỗi quyền truy cập)

echo "Kiểm tra và tạo thư mục ~/.deepface..."
mkdir -p ~/.deepface

echo "Thiết lập hệ thống hoàn tất. Bắt đầu cài đặt Python packages..."
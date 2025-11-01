#!/bin/bash

echo "Bắt đầu thiết lập hệ thống..."
sudo apt-get update
sudo apt-get install -y libgfortran5 build-essential
mkdir -p ~/.deepface
echo "Thiết lập hệ thống hoàn tất."
## Giới thiệu
Đây là hệ thống nhận diện và phân loại biển báo giao thông được xây dựng bằng Python và thư viện OpenCV. Hệ thống sử dụng các kỹ thuật xử lý ảnh cổ điển để phát hiện vùng biển báo và so khớp mẫu (Template Matching) để định danh.

## Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.12.12
* **Thư viện:** OpenCV, NumPy
* **Kỹ thuật:**
    * Tiền xử lý ảnh (Gaussian Blur, Thresholding).
    * Phân đoạn màu sắc (Color Segmentation trong không gian HSV).
    * Phát hiện đường viền (Contour Detection).
    * Nhận diện biển báo (Template Matching).

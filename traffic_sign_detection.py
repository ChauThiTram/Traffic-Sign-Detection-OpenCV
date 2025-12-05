# import thư viện 
import cv2
import numpy as np
import os

# Hàm cắt khung ảnh
def get_frame_roi(frame):
    height, width, _ = frame.shape   # chiều cao và chiều rộng của frame
    end_y = int(height * 0.4)   # Chỉ lấy 40% phần trên của frame
    frame_roi = frame[:end_y, :]  # Cắt roi từ đầu đến end_y
    return frame_roi    # trả về roi  

# Hàm tạo mặt nạ màu đỏ 
def get_mask_red(hsv, lower_red1, upper_red1, lower_red2, upper_red2, kernel):
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), # Tạo mask cho dải đỏ thấp 
                              cv2.inRange(hsv, lower_red2, upper_red2)) # OR giữa hai dải đỏ (0-10, 160-179)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)# Loại nhiễu nhỏ bằng phép Morphology 
    return mask_red     # Trả về mask màu màu đỏ 
    
# Hàm tạo mặt nạ màu xanh dương
def get_mask_blue(hsv, lower_blue, upper_blue, kernel):
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)    # Tạo mask cho dải xanh 
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)    # Loại bỏ nhiễu nhỏ bằng phép Morphology 
    return mask_blue     # Trả về mask màu xanh 


def detect_shape(cnt):
    detected_shape = "unknown" # Cho mặc định unknown 
    peri = cv2.arcLength(cnt, True) # Tính chu vi 
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True) # làm trơn 
    v = len(approx) # số đỉnh polygon 
    
    # Nếu v = 3 thì gán "triangle" vào detected_shape
    if v == 3:
        detected_shape = "triangle"
    
    # Nếu v = 4 thì gán "rectangle" vào detected_shape
    elif v == 4:
        detected_shape = "rectangle"

    # Nếu v > 5 
    elif v > 5:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt) # khung tròn nhỏ nhất 

        # Nếu bán kính lớn hơn 0 
        if radius > 0:
            area = cv2.contourArea(cnt) # diện tích 
            circ_area = np.pi * radius * radius #diện tích vòng tròn 
            circularity = area / circ_area # tính độ tròn 

            # nếu circularity > 0.4 thì gán "circle" vào detected_shape
            if circularity > 0.4:
                detected_shape = "circle"
    
    return detected_shape   # Trả về loại hình dạng 


# Hàm template matching 
def template_matching(roi, color_name, detected_shape, w, h):
    best_match = None   # Biến lưu tên template có độ khớp cao 
    best_score = 0  # Biến lưu điểm khớp cao nhất 

    # Chạy qua tất cả các template đã tải 
    for tmpl in templates:
        # Nếu color hoặc shape không khớp với contour thì bỏ qua template 
        if tmpl["color"] != color_name or tmpl["shape"] != detected_shape:
            continue
        tmpl_img = cv2.resize(tmpl["image"], (w, h)) # resize về đúng kích thước bouding box của contour 

        # Thực hiện template matching với roi của contour 
        # TM_CCOEFF_NORMED: phương pháp chuẩn hóa để ra giá trị từ -1 đến 1, càng cao càng khớp  
        res = cv2.matchTemplate(roi, tmpl_img, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res) # score là giá trị khớp cao nhất 

        # Nếu điểm khớp này cao hơn best_score trước đó 
        if score > best_score:
            best_score = score # cập nhật điểm khớp cao nhất 
            best_match = tmpl["label"] # Cập nhật tên template khớp nhất 
    return best_match, best_score # Trả về template tốt nhất và điểm khớp với nó 

# Hàm xử lý frame 
def process_frame(frame):
    frame_roi = get_frame_roi(frame) # Cắt phần trên của frame 
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV) # Chuyển ROI sang HSV 
    mask_red = get_mask_red(hsv, lower_red1, upper_red1, lower_red2, upper_red2, kernel)    # Mask đỏ 
    mask_blue = get_mask_blue(hsv, lower_blue, upper_blue, kernel)  # Mask xanh 

    # Chạy vòng lặp qua từng màu
    for color_name, mask in [("red", mask_red), ("blue", mask_blue)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Tìm contour 
        # Lặp qua từng contour
        for cnt in contours:
            area = cv2.contourArea(cnt) # Diện tích contour 
            # Nếu diện tích nhỏ hơn 500 thì bỏ qua
            if area < 500:
                continue
            
            
            x, y, w, h = cv2.boundingRect(cnt) # Bounding rectangle 
            detected_shape = detect_shape(cnt) # Phát hiện hình dạng 
            roi = frame[y:y+h, x:x+w] # Cắt ROI theo bouding rectangle 

            # Nếu roi rỗng thì bỏ qua 
            if roi.size == 0:
                continue

            best_match, best_score = template_matching(roi, color_name, detected_shape, w, h) # So sánh tempalte 
            
            # Nếu best_score lớn hơn 0.2 thì vẽ rectangle và text 
            if best_score > 0.2:
                draw_detection(frame, x, y, w, h, color_name, best_match, best_score)
    cv2.putText(
        frame,
        "52200139_52300137",
        (10,30),# vị trí x,y trên frame
        cv2.FONT_HERSHEY_SIMPLEX,#front
        0.8,# size chữ
        (0,255,0),# màu (B,G,R) - ở đây là xanh lá
        2# độ dày nét
    )
    return frame # Trả về fram đã được xử lý 


# Hàm vẽ rectangle và text trên frame 
def draw_detection(frame, x, y, w, h, color_name, label, score):
    color_box = (0, 0, 255) if color_name == "red" else (255, 0, 0) # Chọn màu cho rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color_box, 2) # Vẽ rectangle 
    cv2.putText(frame, f"{label} ({score:.2f})",                         # Chèn text 
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)


# Hàm xử lý video 
def process_video(input_path, output_path):
    
    cap = cv2.VideoCapture(input_path) # Mở video 

    #Kiểm tra xem video có mở được không, nếu không mở được thì thông báo và return 
    if not cap.isOpened():
        print("Không mở được video", input_path)
        return
    
    frame_width = int(cap.get(3)) # Lấy chiều rộng 
    frame_height = int(cap.get(4)) # Lấy chiều cao 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec XVID
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))# Tạo VideoWriter
    print(f"Đang xử lý: {input_path}") # in ra thông báo đang xử lý video  
    # Chạy vòng lặp liên tục    
    while True:
        # Đọc frame, ret trả về True/False 
        ret, frame = cap.read()
        # Nếu ret = false thì thoát khỏi vòng lặp 
        if not ret:
            break
        
        processed = process_frame(frame) # Xử lý frame
        out.write(processed) # Ghi frame vào video output 
    
    cap.release() # Giải phóng video input
    out.release() # Giải phóng video output 
    print(f"Xuất video: {output_path}") # In ra thông báo xuất video
if __name__ == "__main__":
    
    template_folder = "templates" # Thư mục chứa template hình ảnh 
    # Các template hình ảnh 
    template_data = [
        {"label": "Bien bao cam queo trai", "file": "cam_queo_trai.jpg", "shape": "circle", "color": "red"},
        {"label": "Bien bao cam di nguoc chieu", "file": "cam_di_nguoc_chieu.jpg", "shape": "circle", "color": "red"},
        {"label": "Bien bao nguy hiem canh bao di cham", "file": "nguy_hiem.jpg", "shape": "triangle", "color": "red"},
        {"label": "Bien bao hieu lenh", "file": "hieu_lenh.jpg", "shape": "circle", "color": "blue"},
        {"label": "Bien bao cam dau xe", "file": "cam_dau_xe.jpg", "shape": "circle", "color": "red"},
        {"label": "Bien bao cam dau xe", "file": "cam_dau_xe_1.jpg", "shape": "circle", "color": "red"},
        {"label": "Bien bao nguy hiem co tre em", "file": "nguy_hiem_1.jpg", "shape": "triangle", "color": "red"},
        {"label": "Bien bao chi dan", "file": "chi_dan.jpg", "shape": "rectangle", "color": "blue"},
        {"label": "Bien bao chi dan", "file": "chi_dan_1.jpg", "shape": "rectangle", "color": "blue"}, 
        {"label": "Bien bao chi dan", "file": "chi_dan_2.jpg", "shape": "rectangle", "color": "blue"}
    ]

    # Tải các template vào bộ nhớ
    templates = [] # Tạo biến template với 1 mảng rỗng 

    # Chạy từng cái template hình ảnh 
    for t in template_data:
        img_path = os.path.join(template_folder, t["file"]) # Đường dẫn tới template 
        img = cv2.imread(img_path) # Đọc hình ảnh template 

        # Nếu img không None thì gán vào t["image"], và thêm t vào templates 
        if img is not None:
            t["image"] = img 
            templates.append(t)
        
        # Nếu img None thì print là thông báo 
        else:
            print(f"Loi: Khong tai duoc template {t['file']} tai {img_path}")

    # Thiết lập dải màu
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([179, 255, 255])
    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([130, 255, 255])
  
    

    kernel = np.ones((3, 3), np.uint8)  # Kernel 3x3 cho morphology 

    # Xử lý video1 và video 2
    process_video("video1.mp4", "52200139_52300137_video1.avi")
    process_video("video2.mp4", "52200139_52300137_video2.avi")

import streamlit as st
import pandas as pd
import numpy as np # <--- Đảm bảo numpy được import
import cv2
from PIL import Image
import io
from deepface import DeepFace

# --- (Các hàm tiện ích và thiết lập không đổi) ---

# Hàm tính khoảng cách Euclidean L2
def euclidean_l2(a, b):
    # Lỗi xảy ra nếu a và b không phải là numpy array.
    return np.linalg.norm(a - b)

# Hàm tính khoảng cách Cosine
def cosine_distance(a, b):
    # Lỗi xảy ra nếu a và b không phải là numpy array.
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Hàm xử lý và chuyển đổi ảnh
def load_image_from_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Chuyển đổi PIL Image sang numpy array (dạng OpenCV)
        image_np = np.array(image.convert('RGB'))
        return image_np
    return None

# --- (Giao diện Tải lên Ảnh không đổi) ---
st.set_page_config(
    page_title="Phân Tích Khuôn Mặt Bé Với Bố Mẹ (DeepFace)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👨‍👩‍👧 Ai là người bé giống hơn? - DeepFace Analyzer")
st.markdown("Sử dụng DeepFace để so sánh khoảng cách khuôn mặt giữa Bé với Bố và Bé với Mẹ.")

st.header("1. Tải lên Ảnh")

col1, col2, col3 = st.columns(3)

with col1:
    uploaded_father = st.file_uploader("Ảnh Bố", type=["jpg", "jpeg", "png"], key="father")
with col2:
    uploaded_child = st.file_uploader("Ảnh Bé", type=["jpg", "jpeg", "png"], key="child")
with col3:
    uploaded_mother = st.file_uploader("Ảnh Mẹ", type=["jpg", "jpeg", "png"], key="mother")


# --- 4. Logic Xử lý Ảnh và Phân tích (CẬP NHẬT Ở ĐÂY) --------------------

if uploaded_father and uploaded_child and uploaded_mother:
    
    # Hiển thị ảnh
    st.header("2. Ảnh đã Tải lên")
    
    colA, colB, colC = st.columns(3)
    
    img_father_np = load_image_from_uploaded_file(uploaded_father)
    img_child_np = load_image_from_uploaded_file(uploaded_child)
    img_mother_np = load_image_from_uploaded_file(uploaded_mother)
    
    with colA:
        st.subheader("Bố")
        st.image(img_father_np, use_column_width=True)
    with colB:
        st.subheader("Bé")
        st.image(img_child_np, use_column_width=True)
    with colC:
        st.subheader("Mẹ")
        st.image(img_mother_np, use_column_width=True)
        
    st.markdown("---")
    
    # Bắt đầu phân tích
    if st.button("Bắt Đầu Phân Tích Khuôn Mặt", type="primary"):
        with st.spinner('Đang trích xuất embeddings và tính toán khoảng cách...'):
            try:
                # 3. Trích xuất Embeddings (Đặc trưng Khuôn mặt)
                
                # CẬP NHẬT: Thêm np.asarray() để chuyển đổi list thành numpy array
                e_f_list = DeepFace.represent(img_father_np, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]
                e_c_list = DeepFace.represent(img_child_np, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]
                e_m_list = DeepFace.represent(img_mother_np, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]

                embedding_father = np.asarray(e_f_list)
                embedding_child = np.asarray(e_c_list)
                embedding_mother = np.asarray(e_m_list)
                
                st.success("Trích xuất Embeddings hoàn tất!")

                # 4. Tính toán Khoảng cách (Không cần thay đổi vì giờ đây chúng là numpy array)
                
                # Khoảng cách Bé - Bố (Child-Father)
                D_CF_L2 = euclidean_l2(embedding_child, embedding_father)
                D_CF_Cos = cosine_distance(embedding_child, embedding_father)
                
                # Khoảng cách Bé - Mẹ (Child-Mother)
                D_CM_L2 = euclidean_l2(embedding_child, embedding_mother)
                D_CM_Cos = cosine_distance(embedding_child, embedding_mother)

                # --- 5. Hiển thị Bảng Kết quả ---
                
                st.header("📊 Kết Quả So Sánh Định Lượng")
                
                results_data = {
                    "Cặp So sánh": ["Bé - Bố", "Bé - Mẹ"],
                    "Khoảng cách L2 (Euclidean L2)": [round(D_CF_L2, 4), round(D_CM_L2, 4)],
                    "Khoảng cách Cosine": [round(D_CF_Cos, 4), round(D_CM_Cos, 4)]
                }

                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results, hide_index=True, use_container_width=True)

                st.markdown("---")

                # --- 6. Đưa ra Kết luận Cuối cùng ---
                st.header("⭐ Kết Luận Cuối Cùng")
                
                # Xác định người giống hơn (khoảng cách nhỏ hơn -> giống hơn)
                if D_CF_L2 < D_CM_L2:
                    st.balloons()
                    st.markdown(f"""
                    #### Dựa trên Khoảng cách L2, Bé **giống Bố** hơn!
                    - **Bé - Bố (L2):** `{round(D_CF_L2, 4)}` (Nhỏ hơn)
                    - **Bé - Mẹ (L2):** `{round(D_CM_L2, 4)}`
                    """)
                elif D_CM_L2 < D_CF_L2:
                    st.balloons()
                    st.markdown(f"""
                    #### Dựa trên Khoảng cách L2, Bé **giống Mẹ** hơn!
                    - **Bé - Bố (L2):** `{round(D_CF_L2, 4)}`
                    - **Bé - Mẹ (L2):** `{round(D_CM_L2, 4)}` (Nhỏ hơn)
                    """)
                else:
                     st.markdown("#### Khoảng cách khuôn mặt Bé với Bố và Mẹ là gần như bằng nhau!")

                st.info("Lưu ý: Khoảng cách càng nhỏ, khuôn mặt càng giống nhau. Kết quả chỉ mang tính tham khảo và có thể thay đổi tùy thuộc vào chất lượng ảnh, góc chụp và mô hình DeepFace được sử dụng.")


            except ValueError as e:
                if "Face could not be detected" in str(e):
                     st.error("Lỗi: Không thể phát hiện khuôn mặt trong một hoặc nhiều ảnh. Vui lòng thử lại với ảnh rõ ràng hơn.")
                else:
                     st.error(f"Đã xảy ra lỗi trong quá trình xử lý DeepFace: {e}")
            except Exception as e:
                # In ra lỗi chi tiết hơn nếu cần debug
                st.error(f"Đã xảy ra lỗi không xác định: {e}")
                # st.exception(e) # Dùng lệnh này để in đầy đủ Traceback
                
else:
    st.info("Vui lòng tải lên đầy đủ 3 ảnh (Bố, Bé, Mẹ) để bắt đầu phân tích.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation



# استایل‌های سفارشی برای راست‌چین کردن
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        text-align: right;
        direction: rtl;
    }
    .stTextArea > div > div > textarea {
        text-align: right;
        direction: rtl;
    }
    .element-container:has(div.stMarkdown) {
        text-align: right;
        direction: rtl;
    }
    .stButton > button {
        float: right;
    }
    .stTabs [data-baseweb="tab-list"] {
        flex-direction: row-reverse;
    }
    .stTabs [data-baseweb="tab"] {
        direction: rtl;
    }
    .stTextInput > label {
        text-align: right;
        direction: rtl;
    }
    .stTextArea > label {
        text-align: right;
        direction: rtl;
    }
    h2, h3 {
        text-align: right !important;
        direction: rtl !important;
    }
    h1 {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# بارگذاری مدل‌ها
@st.cache_resource
def load_models():
    face_detector = FaceDetector()
    face_mesh = FaceMeshDetector(maxFaces=1)
    segmentor = SelfiSegmentation()
    return face_detector, face_mesh, segmentor

# تابع اعمال افکت‌های مختلف
def apply_effect(frame, face_detector, face_mesh, segmentor, effect_type):
    # تشخیص چهره
    frame, bboxs = face_detector.findFaces(frame)
    
    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            
            if effect_type == "کارتون":
                try:
                    # افکت کارتونی
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:  # بررسی خالی نبودن ROI
                        roi = cv2.bilateralFilter(roi, 9, 75, 75)
                        roi = cv2.edgePreservingFilter(roi, flags=1, sigma_s=60, sigma_r=0.4)
                        frame[y:y+h, x:x+w] = roi
                except:
                    pass
                
            elif effect_type == "پیکسلی":
                # افکت پیکسلی
                roi = frame[y:y+h, x:x+w]
                small = cv2.resize(roi, (10, 10))
                roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = roi
                
            elif effect_type == "رنگین‌کمان":
                # افکت رنگین‌کمان
                roi = frame[y:y+h, x:x+w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hsv[:,:,0] = (hsv[:,:,0] + 30) % 180
                roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame[y:y+h, x:x+w] = roi
                
            elif effect_type == "سایه":
                # افکت سایه
                roi = frame[y:y+h, x:x+w]
                roi = cv2.GaussianBlur(roi, (99, 99), 0)
                frame[y:y+h, x:x+w] = roi
                
            elif effect_type == "معکوس":
                # افکت معکوس
                roi = frame[y:y+h, x:x+w]
                roi = cv2.bitwise_not(roi)
                frame[y:y+h, x:x+w] = roi
                
            elif effect_type == "ماسک":
                # افکت ماسک
                frame, faces = face_mesh.findFaceMesh(frame)
                
            elif effect_type == "پس‌زمینه":
                # تغییر پس‌زمینه
                frame = segmentor.removeBG(frame, (255, 0, 0), threshold=0.8)
    
    return frame

# رابط کاربری اصلی
st.title("🎭 فیلترهای خلاقانه چهره")
st.markdown("""
این برنامه با استفاده از وب‌کم، چهره شما را تشخیص داده و افکت‌های خلاقانه مختلفی روی آن اعمال می‌کند.
""")

# انتخاب افکت
effect_type = st.selectbox(
    "انتخاب افکت:",
    ["کارتون", "پیکسلی", "رنگین‌کمان", "سایه", "معکوس", "ماسک", "پس‌زمینه"]
)

# محل نمایش تصویر
FRAME_WINDOW = st.empty()

# بارگذاری مدل‌ها
face_detector, face_mesh, segmentor = load_models()

# متغیر وضعیت اجرا و وب‌کم
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# دکمه‌های شروع و توقف
col1, col2 = st.columns(2)
with col1:
    start_button = st.button('شروع', key='start')
with col2:
    stop_button = st.button('توقف', key='stop')

# کنترل وضعیت اجرا
if start_button and not st.session_state.is_running:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.is_running = True

if stop_button and st.session_state.is_running:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.is_running = False
    FRAME_WINDOW.empty()

# حلقه اصلی برای نمایش ویدیو زنده
if st.session_state.is_running and st.session_state.cap is not None:
    while st.session_state.is_running:
        # خواندن فریم از وب‌کم
        ret, frame = st.session_state.cap.read()
        if ret:
            # اعمال افکت
            frame = apply_effect(frame, face_detector, face_mesh, segmentor, effect_type)
            
            # تبدیل BGR به RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # نمایش تصویر
            FRAME_WINDOW.image(frame)
            
            # تاخیر کوچک برای کاهش بار پردازشی
            time.sleep(0.01)
        else:
            st.error("خطا در دسترسی به وب‌کم!")
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.is_running = False
            break

# تب اطلاعات
with st.expander("درباره این پروژه"):
    st.markdown("""
    ### ویژگی‌های اصلی:
    - تشخیص چهره به صورت زنده
    - افکت‌های خلاقانه مختلف
    - رابط کاربری ساده و فارسی
    
    ### افکت‌های موجود:
    1. **کارتون**: تبدیل چهره به سبک کارتونی
    2. **پیکسلی**: تبدیل چهره به پیکسل‌های بزرگ
    3. **رنگین‌کمان**: تغییر رنگ چهره به صورت متناوب
    4. **سایه**: محو کردن چهره
    5. **معکوس**: معکوس کردن رنگ‌های چهره
    6. **ماسک**: نمایش نقاط شبکه چهره
    7. **پس‌زمینه**: تغییر پس‌زمینه تصویر
    
    ### نحوه استفاده:
    1. وب‌کم خود را روشن کنید
    2. افکت مورد نظر را انتخاب کنید
    3. دکمه شروع/توقف را بزنید
    """)

# پاورقی
st.markdown("---")
st.markdown("""
ساخته شده با ❤️ برای یادگیری بینایی ماشین
""")
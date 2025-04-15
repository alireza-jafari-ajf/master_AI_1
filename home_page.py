import streamlit as st

# استایل‌های سفارشی
st.markdown("""
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .project-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .project-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #2193b0, #6dd5ed);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .section-title {
        color: blue;
        margin-bottom: 20px;
        text-align: left;
    }
    .card-content {
        text-align: left;
        color: black;
    }
    .card-content p {
        margin: 5px 0;
        color: black;
    }
    .card-content h3 {
        margin: 5px 0;
        color: black;
    }
    .contact-info {
        text-align: left;
        color: black;
    }
    .contact-info h3 {
        color: white;
    }
    .contact-info p {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# هدر صفحه
st.markdown("""
    <div class="header">
        <h1>🎯 Alireza Jafari Projects</h1>
        <p>Welcome to My Projects Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# ایجاد سه ستون برای پروژه‌ها
col1, col2, col3 = st.columns(3)

# پروژه موتور جستجو
with col1:
    st.markdown("""
        <div class="project-card">
            <h3 class="section-title" style="color: black;">🔍 Search Engine</h3>
            <div class="card-content">
                <p>Knowledge Graphs RAG APP</p>
                <p>🔹 Intelligent Search</p>
                <p>🔹 Knowledge Graph Analysis</p>
                <p>🔹 Accurate Responses</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Run Search Engine Project"):
        st.switch_page("graph_app.py")
# پروژه یادگیری ماشین
with col2:
    st.markdown("""
        <div class="project-card">
            <h3 class="section-title" style="color: black;">🤖 Machine Learning</h3>
            <div class="card-content">
                <p>Torque Clustering</p>
                <p>🔹 Advanced Clustering</p>
                <p>🔹 Data Analysis</p>
                <p>🔹 Visual Representation</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Run Machine Learning Project"):
        st.switch_page("ml_app.py")
# پروژه یادگیری عمیق
with col3:
    st.markdown("""
        <div class="project-card">
            <h3 class="section-title" style="color: black;">🧠 Deep Learning</h3>
            <div class="card-content">
                <p>Face Detections</p>
                <p>🔹 Face Recognition</p>
                <p>🔹 Smart Filters</p>
                <p>🔹 Image Processing</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Run Deep Learning Project"):
        st.switch_page("dl_app.py")
# اطلاعات تماس در سایدبار
with st.sidebar:

    st.markdown("""
        <div class="contact-info">
            <h3>📞 Contact Us</h3>
            <p>📧 Email: mr.alireza.jafari08@gmail.com</p>
            <p>📱 Phone: +98-912-063-8732</p>
        </div>
    """, unsafe_allow_html=True)
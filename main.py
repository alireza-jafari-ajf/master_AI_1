import streamlit as st

st.set_page_config(
    page_title="پروژه‌های من",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto"
)

# استایل‌های سفارشی
st.markdown("""
    <style>
    
    .st-emotion-cache-1f2l9la.ejhh0er0{
            font-size:20px;
            }
    .st-emotion-cache-452g1r.e1c29vlm5{
            font-size:27px;
            }
    .stAppHeader.st-emotion-cache-h4xjwg.e4hpqof0{
            visibility:hidden;
            }
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
    }
    </style>
""", unsafe_allow_html=True)

pages = {
    "Home Page": [
        st.Page("home_page.py", title="Home" , default=True , icon=":material/home:"),
    ],
    "Search Engine": [
        st.Page("graph_app.py", title="Knowledge Graphs RAG APP" , icon=":material/monitoring:"),
    ],
    "Machine Learning": [
        st.Page("ml_app.py", title="Torque Clustering",icon=":material/network_intel_node:"),
    ],
    "Deep Learning": [
        st.Page("dl_app.py", title="Face Detections",icon=":material/tamper_detection_on:"),
    ],
}

pg = st.navigation(pages)
pg.run()


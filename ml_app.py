import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import pandas as pd


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

# پیاده‌سازی الگوریتم Torque Clustering
class TorqueClustering:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # مقداردهی اولیه مراکز خوشه‌ها
        centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # محاسبه فاصله‌ها
            distances = cdist(X, centers)
            
            # محاسبه گشتاور
            torque = np.zeros((n_samples, n_features))
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    torque[i] += (X[i] - centers[j]) * (1 / (distances[i, j] + 1e-10))
            
            # به‌روزرسانی مراکز
            new_centers = np.zeros_like(centers)
            for j in range(self.n_clusters):
                mask = distances.argmin(axis=1) == j
                if mask.any():
                    new_centers[j] = X[mask].mean(axis=0)
            
            # بررسی همگرایی
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
                
            centers = new_centers
            
        self.centers_ = centers
        self.labels_ = distances.argmin(axis=1)
        return self

# تابع تولید داده‌های مصنوعی
def generate_data(data_type, n_samples, noise):
    if data_type == "داده‌های کروی":
        return make_blobs(n_samples=n_samples, centers=3, cluster_std=noise, random_state=42)
    elif data_type == "داده‌های هلالی":
        return make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif data_type == "داده‌های حلقوی":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)

# تابع محاسبه معیارهای ارزیابی
def evaluate_clustering(X, labels):
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
    else:
        silhouette = calinski = davies = 0
    return silhouette, calinski, davies

# رابط کاربری اصلی
st.title("📊 مقایسه الگوریتم‌های خوشه‌بندی")
st.markdown("""
این برنامه الگوریتم‌های مختلف خوشه‌بندی را با هم مقایسه می‌کند.
""")

# تب‌های اصلی
tab1, tab2, tab3 = st.tabs(["مقایسه الگوریتم‌ها", "توضیحات الگوریتم‌ها", "درباره پروژه"])

with tab1:
    # تنظیمات داده‌ها
    col1, col2, col3 = st.columns(3)
    with col1:
        data_type = st.selectbox(
            "نوع داده:",
            ["داده‌های کروی", "داده‌های هلالی", "داده‌های حلقوی"]
        )
    with col2:
        n_samples = st.slider("تعداد نمونه‌ها:", 100, 1000, 500)
    with col3:
        noise = st.slider("نویز:", 0.01, 0.5, 0.1)
    
    # تولید داده‌ها
    X, y_true = generate_data(data_type, n_samples, noise)
    
    # تنظیمات الگوریتم‌ها
    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider("تعداد خوشه‌ها:", 2, 10, 3)
    with col2:
        eps = st.slider("فاصله اپسیلون (DBSCAN):", 0.1, 1.0, 0.3)
    with col3:
        min_samples = st.slider("حداقل نمونه‌ها (DBSCAN):", 2, 20, 5)
    
    # اجرای الگوریتم‌ها
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    torque = TorqueClustering(n_clusters=n_clusters).fit(X)
    
    # محاسبه معیارهای ارزیابی
    kmeans_metrics = evaluate_clustering(X, kmeans.labels_)
    dbscan_metrics = evaluate_clustering(X, dbscan.labels_)
    torque_metrics = evaluate_clustering(X, torque.labels_)
    
    # نمایش نتایج
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("نمودار خوشه‌بندی")
        fig = go.Figure()
        
        # اضافه کردن داده‌های اصلی
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color='gray', size=5),
            name='داده‌های اصلی'
        ))
        
        # اضافه کردن خوشه‌های K-Means
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=kmeans.labels_, size=5),
            name='K-Means'
        ))
        
        # اضافه کردن خوشه‌های DBSCAN
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=dbscan.labels_, size=5),
            name='DBSCAN'
        ))
        
        # اضافه کردن خوشه‌های Torque
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=torque.labels_, size=5),
            name='Torque'
        ))
        
        fig.update_layout(
            title='مقایسه نتایج خوشه‌بندی',
            xaxis_title='ویژگی 1',
            yaxis_title='ویژگی 2',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("معیارهای ارزیابی")
        
        # ایجاد جدول مقایسه
        metrics_data = {
            'الگوریتم': ['K-Means', 'DBSCAN', 'Torque'],
            'Silhouette Score': [kmeans_metrics[0], dbscan_metrics[0], torque_metrics[0]],
            'Calinski-Harabasz': [kmeans_metrics[1], dbscan_metrics[1], torque_metrics[1]],
            'Davies-Bouldin': [kmeans_metrics[2], dbscan_metrics[2], torque_metrics[2]]
        }
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
        
        # توضیحات معیارها
        st.markdown("""
        ### توضیح معیارهای ارزیابی:
        - **Silhouette Score**: هرچه بالاتر باشد، خوشه‌بندی بهتر است (محدوده: -1 تا 1)
        - **Calinski-Harabasz**: هرچه بالاتر باشد، خوشه‌بندی بهتر است
        - **Davies-Bouldin**: هرچه پایین‌تر باشد، خوشه‌بندی بهتر است
        """)

with tab2:
    st.markdown("""
    ### توضیحات الگوریتم‌های خوشه‌بندی
    
    #### 1. K-Means
    - یک الگوریتم خوشه‌بندی ساده و پرکاربرد
    - نیاز به تعیین تعداد خوشه‌ها از قبل دارد
    - مناسب برای داده‌های کروی و با اندازه یکسان
    - حساس به مقداردهی اولیه مراکز خوشه‌ها
    
    #### 2. DBSCAN
    - الگوریتم خوشه‌بندی مبتنی بر چگالی
    - نیازی به تعیین تعداد خوشه‌ها ندارد
    - قادر به تشخیص خوشه‌های با شکل‌های مختلف
    - می‌تواند نویز را تشخیص دهد
    - حساس به پارامترهای اپسیلون و حداقل نمونه‌ها
    
    #### 3. Torque Clustering
    - الگوریتم خوشه‌بندی مبتنی بر گشتاور
    - از مفهوم گشتاور برای بهبود خوشه‌بندی استفاده می‌کند
    - مناسب برای داده‌های با توزیع پیچیده
    - نیاز به تعیین تعداد خوشه‌ها دارد
    - مقاوم‌تر در برابر نویز نسبت به K-Means
    """)

with tab3:
    st.markdown("""
    ### درباره پروژه
    
    این پروژه به عنوان پروژه نهایی درس یادگیری ماشین طراحی شده است و هدف آن مقایسه سه الگوریتم خوشه‌بندی مختلف است:
    
    - K-Means (الگوریتم کلاسیک)
    - DBSCAN (الگوریتم مبتنی بر چگالی)
    - Torque Clustering (الگوریتم مبتنی بر گشتاور)
    
    #### ویژگی‌های پروژه:
    - رابط کاربری ساده و فارسی
    - امکان تولید داده‌های مصنوعی با شکل‌های مختلف
    - مقایسه بصری نتایج خوشه‌بندی
    - ارزیابی کمی با معیارهای مختلف
    - توضیحات کامل درباره هر الگوریتم
    
    #### نحوه استفاده:
    1. نوع داده مورد نظر را انتخاب کنید
    2. تعداد نمونه‌ها و میزان نویز را تنظیم کنید
    3. پارامترهای الگوریتم‌ها را تنظیم کنید
    4. نتایج را در نمودار و جدول مقایسه مشاهده کنید
    
    #### کتابخانه‌های استفاده شده:
    - Streamlit برای رابط کاربری
    - Scikit-learn برای الگوریتم‌های خوشه‌بندی
    - Plotly برای رسم نمودارها
    - NumPy و Pandas برای پردازش داده‌ها
    """)

# پاورقی
st.markdown("---")
st.markdown("""
ساخته شده با ❤️ برای پروژه یادگیری ماشین
""")
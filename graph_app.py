import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from knowledge_graph import KnowledgeGraph
import os


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
    /* راست‌چین کردن متن‌های بالای تکست‌اینپوت‌ها */
    .stTextInput > label {
        text-align: right;
        direction: rtl;
    }
    .stTextArea > label {
        text-align: right;
        direction: rtl;
    }
    /* راست‌چین کردن هدرها */
    h2, h3 {
        text-align: right !important;
        direction: rtl !important;
    }
    h1 {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# ایجاد یک نمونه از KnowledgeGraph
@st.cache_resource
def get_knowledge_graph():
    return KnowledgeGraph()

kg = get_knowledge_graph()

# رابط کاربری اصلی
st.title("🧠 Knowledge Graph RAG Application")
st.markdown("""
این برنامه یک نمونه عملی از استفاده از گراف‌های دانش در سیستم‌های بازیابی اطلاعات است.
""")

# تب‌های اصلی
tab1, tab2, tab3, tab4 = st.tabs(["ورود داده", "جستجو و پرس و جو", "نمایش گراف", "تحلیل آماری"])

with tab1:
    st.header("ورود داده‌های جدید")
    input_text = st.text_area("متن خود را وارد کنید:", height=200)
    if st.button("پردازش متن"):
        if input_text:
            with st.spinner("در حال پردازش متن..."):
                kg.process_text(input_text)
                st.success("متن با موفقیت پردازش شد!")
        else:
            st.warning("لطفاً متنی را وارد کنید.")

with tab2:
    st.header("جستجو و پرس و جو")
    query = st.text_input("سوال خود را مطرح کنید:")
    if st.button("جستجو"):
        if query:
            with st.spinner("در حال جستجو..."):
                results = kg.query_graph(query)
                if results:
                    st.subheader("نتایج جستجو:")
                    for result in results:
                        st.write(f"موجودیت: {result['text']}")
                        st.write(f"نوع: {result['type']}")
                        st.write(f"میزان شباهت: {result['similarity']:.2f}")
                        st.markdown("---")
                else:
                    st.info("هیچ نتیجه‌ای یافت نشد.")
        else:
            st.warning("لطفاً سوالی را وارد کنید.")

with tab3:
    st.header("نمایش گراف دانش")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("نمایش گراف"):
            pos, graph = kg.visualize()
            if pos is not None and graph is not None:
                st.pyplot(plt.gcf())
            else:
                st.info("هیچ داده‌ای برای نمایش وجود ندارد. لطفاً ابتدا متنی را وارد کنید.")
    with col2:
        st.markdown("""
        ### راهنمای رنگ‌ها:
        - 🔴 افراد (PERSON)
        - 🟢 سازمان‌ها (ORG)
        - 🔵 مکان‌ها (GPE)
        - 🟡 محصولات (PRODUCT)
        - 🟣 تاریخ‌ها (DATE)
        - ⚪ سایر موارد
        """)

with tab4:
    st.header("تحلیل آماری")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("توزیع انواع موجودیت‌ها")
        entity_stats = kg.get_entity_statistics()
        if entity_stats is not None:
            st.pyplot(entity_stats)
        else:
            st.info("هیچ داده‌ای برای تحلیل وجود ندارد. لطفاً ابتدا متنی را وارد کنید.")
    with col2:
        st.subheader("توزیع انواع روابط")
        relation_stats = kg.get_relation_statistics()
        if relation_stats is not None:
            st.pyplot(relation_stats)
        else:
            st.info("هیچ داده‌ای برای تحلیل وجود ندارد. لطفاً ابتدا متنی را وارد کنید.")

# پاورقی
st.markdown("---")
st.markdown("""
### درباره این پروژه
این برنامه یک نمونه عملی از استفاده از گراف‌های دانش در سیستم‌های بازیابی اطلاعات است که شامل:
- استخراج موجودیت‌ها و روابط از متن
- ذخیره‌سازی اطلاعات در یک گراف دانش
- جستجو و بازیابی اطلاعات
- نمایش بصری گراف دانش
- تحلیل آماری موجودیت‌ها و روابط
            
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
Steve Jobs was one of the co-founders of Apple. 
The company is known for its iPhone, iPad, and Mac computers.
""")
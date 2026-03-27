import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


st.set_page_config(
    page_title="AEGIS Intelligence Dashboard", 
    page_icon="🤖", 
    layout="wide"
)


def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("⚠️ ไม่พบไฟล์ style.css กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกัน")

local_css("style.css")


@st.cache_resource
def load_all_assets():
    # Model 1: Ensemble (Gen Z Shopping)
    m1 = joblib.load('shopping_ensemble_v2.pkl')
    s1 = joblib.load('shopping_scaler_v2.pkl')
    enc1 = joblib.load('shopping_encoders_v2.pkl')
    # Model 2: Neural Network (Football Injury)
    m2 = tf.keras.models.load_model('injury_nn_model.h5')
    s2 = joblib.load('injury_scaler_nn.pkl')
    enc2 = joblib.load('injury_le_pos.pkl')
    return m1, s1, enc1, m2, s2, enc2

m1, s1, enc1, m2, s2, enc2 = load_all_assets()


with st.sidebar:
    # โลโก้โครงการ
    st.image("https://media.discordapp.net/attachments/1238726601001144322/1486825269263339530/Gemini_Generated_Image_b3dvwob3dvwob3dv-removebg-preview.png?ex=69c6e970&is=69c597f0&hm=498a381f553f1eb516cd7e5034dd106f3e917e5ca74bde5ab1b3827f734a86a6&=&format=webp&quality=lossless", width=300)
    st.title("AEGIS Intelligence")
    st.markdown("---")
    page = st.selectbox("🌐 PAGE:", [
        "📊 ข้อมูลโมเดล (ML Ensemble)",
        "🛍️ ทดสอบโมเดล (Shopping)",
        "🧠 ข้อมูลโมเดล (Neural Network)",
        "⚽ ทดสอบโมเดล (Football)"
    ])
    st.markdown("---")
    st.caption("🚀 Version 1.0")
    st.info("Sirasit 6704062617067")


# หน้าที่ 1: ข้อมูลโมเดล 1 (Theory - Ensemble)

if page == "📊 ข้อมูลโมเดล (ML Ensemble)":
    st.title("📊 Machine Learning Architecture")
    st.markdown("#### หัวข้อ: การพยากรณ์พฤติกรรมการซื้อของ Gen Z ด้วยระบบ Ensemble Learning")
    
    

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.header("🛠️ Data Preparation & Strategy")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ที่มาและการจัดการข้อมูล")
        st.write("""
        * **Gemini Generated Data:** ชุดข้อมูลทั้งหมดถูกสร้างขึ้น (Synthesized) โดย **Gemini AI** เพื่อจำลองพฤติกรรมการซื้อจริง 5,000 ตัวอย่าง
        * **Cleaning:** จัดการข้อมูลซ้ำซ้อนและเติมค่าว่าง (Null) ด้วยค่ามัธยฐานเพื่อรักษาความถูกต้องของข้อมูล
        """)
    with col_b:
        st.subheader("ยุทธศาสตร์การปรับสมดุล")
        st.write("""
        * **StandardScaler:** ปรับสเกลข้อมูลให้สมดุลเพื่อป้องกัน Bias จากตัวแปรที่มีค่าตัวเลขสูง
        * **Encoding:** แปลงข้อมูลเชิงคุณภาพเป็นตัวเลขผ่าน Label Encoding เพื่อให้โมเดลประมวลผลได้
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.header("🧠 Algorithm Theory: Ensemble Learning")
    st.write("ระบบใช้เทคนิค **Voting Classifier (Soft Voting)** ซึ่งผสานพลังของ 3 โมเดล:")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("**1. Random Forest**")
        st.caption("ลดความแปรปรวนของข้อมูลผ่านต้นไม้ตัดสินใจหลายชุด")
    with t_col2:
        st.markdown("**2. Gradient Boosting**")
        st.caption("เพิ่มประสิทธิภาพโดยการเรียนรู้จากจุดที่โมเดลก่อนหน้าทำผิดพลาด")
    with t_col3:
        st.markdown("**3. Logistic Regression**")
        st.caption("เป็นรากฐานเชิงเส้นที่ช่วยให้การรวมน้ำหนักมีความเสถียร")
    st.markdown('</div>', unsafe_allow_html=True)


# หน้าที่ 2: ทดสอบโมเดล 1 (Test - Shopping)

elif page == "🛍️ ทดสอบโมเดล (Shopping)":
    st.title("🛍️ Shopping Intent Analysis")
    st.write("พยากรณ์พฤติกรรมการตัดสินใจซื้อของกลุ่ม Gen Z ในโลกดิจิทัล(คลิปออนไลน์)")
    
    with st.container():
        with st.form("shop_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("อายุเป้าหมาย", 15, 27, 20)
                gen = st.selectbox("เพศ", enc1['Gender'].classes_)
                inf = st.selectbox("ประเภท Influencer", enc1['Influencer_Type'].classes_)
                cat = st.selectbox("หมวดหมู่สินค้า", enc1['Product_Category'].classes_)
            with c2:
                price = st.number_input("ราคาสินค้า (บาท)", 99, 5000, 990)
                aes = st.slider("Aesthetic Score (ความสวยงามคอนเทนต์)", 1, 10, 8)
                social = st.number_input("ยอดคอมเมนต์รีวิว", 0, 5000, 500)
                time = st.slider("Watch Time (วินาที)", 1, 180, 45)
                disc = st.radio("มีส่วนลด/คูปองหรือไม่?", ["Yes", "No"], horizontal=True)
            
            sub1 = st.form_submit_button("🔥 วิเคราะห์โอกาสการซื้อ")
            
        if sub1:
            in1 = pd.DataFrame({
                'Age': [age],
                'Gender': [enc1['Gender'].transform([gen])[0]],
                'Influencer_Type': [enc1['Influencer_Type'].transform([inf])[0]],
                'Product_Category': [enc1['Product_Category'].transform([cat])[0]],
                'Product_Price_THB': [price],
                'Aesthetic_Score': [aes],
                'Social_Proof_Comments': [social],
                'Discount_Code_Available': [enc1['Discount_Code_Available'].transform([disc])[0]],
                'Time_Spent_Watching_Seconds': [time]
            })
            
            prob1 = m1.predict_proba(s1.transform(in1))[0][1]
            st.markdown("---")
            st.metric("โอกาสที่ลูกค้าจะซื้อ", f"{prob1*100:.2f}%")
            st.progress(float(prob1))


# หน้าที่ 3: ข้อมูลโมเดล 2 (Theory - NN)

elif page == "🧠 ข้อมูลโมเดล (Neural Network)":
    st.title("🧠 Deep Learning Architecture")
    st.markdown("#### หัวข้อ: ระบบประเมินความเสี่ยงบาดเจ็บนักกีฬาด้วยโครงข่ายประสาทเทียม")

    

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.header("🧪 Data Strategy & Provenance")
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("ที่มาของข้อมูลไบโอเมตริกซ์")
        st.write("""
        * **Gemini Generated Bio-Data:** ชุดข้อมูลสุขภาพและประสิทธิภาพนักกีฬาทั้งหมดสร้างโดย **Gemini AI** โดยอ้างอิงจากตัวเลขสากลของ Sport Science
        * **Multimodal Integration:** ผสานข้อมูลความล้าสะสม (Load) และการฟื้นตัว (Recovery)
        """)
    with col_d:
        st.subheader("มาตรฐานสากล")
        st.write("""
        * **Global Scaling:** ใช้ StandardScaler ปรับค่าหน่วยที่ต่างกันให้เป็นมาตรฐาน (Mean=0, Std=1)
        * **Pattern Detection:** เน้นการตรวจจับรูปแบบ Non-linear ของความเสี่ยงที่มองไม่เห็นด้วยตาเปล่า
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.header("🏗️ Deep Learning Architecture: MLP")
    n_col1, n_col2 = st.columns(2)
    with n_col1:
        st.markdown("**1. โครงสร้างโหนด (Structure)**")
        st.write("""
        - **Input Layer:** 9 Neurons (ตัวแปรไบโอเมตริกซ์)
        - **Hidden Layers:** 3 ชั้น (128, 64, 32 Nodes) สกัดความซับซ้อนของข้อมูล
        - **Output Layer:** Sigmoid Activation สำหรับพยากรณ์ความเสี่ยง 0-1
        """)
    with n_col2:
        st.markdown("**2. การปรับปรุงประสิทธิภาพ (Optimization)**")
        st.write("""
        - **Batch Normalization:** ช่วยให้การเรียนรู้เสถียรและแม่นยำขึ้น
        - **Dropout (30%):** ป้องกันโมเดล Overfitting หรือจำข้อมูลเก่าเกินไป
        - **Adam Optimizer:** อัลกอริทึมการจูนน้ำหนักที่ล้ำสมัยที่สุด
        """)
    st.markdown('</div>', unsafe_allow_html=True)


# หน้าที่ 4: ทดสอบโมเดล 2 (Test - Football)

else:
    st.title("⚽ Football Injury Predictor")
    st.write("วิเคราะห์ความพร้อมของร่างกายนักเตะด้วยระบบ Neural Network AI")
    
    with st.container():
        with st.form("ball_form"):
            c3, c4 = st.columns(2)
            with c3:
                p_age = st.slider("อายุ", 17, 40, 25)
                p_pos = st.selectbox("ตำแหน่งในสนาม", enc2.classes_)
                p_dist = st.number_input("ระยะวิ่งสะสม 3 นัดหลัง (KM)", 10.0, 60.0, 30.0)
                p_match = st.number_input("จำนวนนัดที่ลงเล่นติดต่อกัน", 1, 15, 3)
            with c4:
                p_sleep = st.slider("ชั่วโมงการนอนเฉลี่ย", 4.0, 10.0, 8.0)
                p_hrv = st.number_input("ค่า HRV (20-120)", 20, 120, 70)
                p_speed = st.number_input("ความเร็วสูงสุด (KM/H)", 20.0, 38.0, 32.0)
                p_prev = st.number_input("ประวัติการบาดเจ็บในอดีต (ครั้ง)", 0, 10, 1)
                p_rpe = st.slider("ระดับความเหนื่อยซ้อม (RPE)", 1, 10, 6)
            
            sub2 = st.form_submit_button("⚡ วิเคราะห์ความเสี่ยงบาดเจ็บ")
            
        if sub2:
            in2 = pd.DataFrame({
                'Age': [p_age],
                'Position': [enc2.transform([p_pos])[0]],
                'Total_Distance_3_Matches_KM': [p_dist],
                'Top_Speed_KMH': [p_speed],
                'Average_Sleep_Hours': [p_sleep],
                'HRV_Score': [p_hrv],
                'Matches_Played_Consecutive': [p_match],
                'Training_Intensity_RPE': [p_rpe],
                'Previous_Injuries_Count': [p_prev]
            })
            
            prob2 = m2.predict(s2.transform(in2))[0][0]
            st.markdown("---")
            st.metric("Injury Risk Score", f"{prob2*100:.2f}%")
            st.progress(float(prob2))
            
            if prob2 > 0.7:
                st.error("🚨 **ความเสี่ยงสูง:** ควรพักนักเตะด่วนเพื่อป้องกันการเจ็บหนัก")
            elif prob2 > 0.4:
                st.warning("⚠️ **เฝ้าระวัง:** ควรลดความเข้มข้นในการซ้อมลง")
            else:
                st.success("✅ **สภาพร่างกายพร้อม:** นักเตะสมบูรณ์พร้อมลงแข่งขัน")
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 📌 إعداد واجهة المستخدم
st.set_page_config(layout="wide")
st.title("🔍 كشف الحسابات الشاذة باستخدام التحليل الهيكلي و Autoencoder")

# 📌 تحميل بيانات الشبكة
st.sidebar.header("📂 تحميل بيانات الشبكة الاجتماعية")
uploaded_file = st.sidebar.file_uploader("اختر ملف CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("✅ **تم تحميل البيانات بنجاح!**")
    st.dataframe(data.head())

    # 🔍 تحليل الشبكة باستخدام Graph
    st.subheader("🔗 تحليل بنية الشبكة")
    G = nx.from_pandas_edgelist(data, source="source", target="target")
    
    # 📌 رسم الشبكة
    plt.figure(figsize=(10, 6))
    nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")
    st.pyplot(plt)

    # 📌 استخراج ميزات الشبكة لاستخدامها في Autoencoder
    degrees = dict(G.degree())
    features = np.array(list(degrees.values())).reshape(-1, 1)

    # 📌 نموذج Autoencoder لاكتشاف الشذوذ
    model = Sequential([
        Dense(8, activation='relu', input_shape=(1,)),
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(features, features, epochs=50, verbose=0)
    
    predictions = model.predict(features)
    reconstruction_errors = np.abs(predictions - features)

    # 📌 تحديد الشذوذ بناءً على أعلى 5 قيم للخطأ
    anomalies = np.argsort(reconstruction_errors.flatten())[-5:]

    st.subheader("🚨 العقد الشاذة المكتشفة")
    st.write("🔴 هذه هي الحسابات الأكثر شذوذًا بناءً على تحليل الشبكة:")
    st.write(anomalies)

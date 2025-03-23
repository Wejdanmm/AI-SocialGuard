import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 🎨 تحسين الصفحة العامة
st.set_page_config(
    page_title="كشف الحسابات المشبوهة", 
    page_icon="🔍",
    layout="wide"
)

# 💡 تحسين التصميم باستخدام CSS
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
    }
    .subtitle {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #117A65;
    }
    .metric-box {
        background-color: #2E86C1;
        color: white;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown('<p class="title">🔍 كشف الحسابات الشاذة باستخدام التحليل الهيكلي و Autoencoder</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📊 تحليل بيانات الشبكات الاجتماعية وتحديد الحسابات المشبوهة</p>', unsafe_allow_html=True)

# 📂 تحميل البيانات
st.sidebar.header("📂 تحميل بيانات الشبكة الاجتماعية")
uploaded_file = st.sidebar.file_uploader("📥 **اختر ملف CSV**", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ **تم تحميل البيانات بنجاح!**")
    st.dataframe(data.head())

    # 📌 التعرف التلقائي على أسماء الأعمدة
    actual_columns = data.columns.tolist()
    rename_dict = {}

    for col in actual_columns:
        if col.lower() in ["source", "node1", "from", "user1"]:
            rename_dict[col] = "source"
        if col.lower() in ["target", "node2", "to", "user2"]:
            rename_dict[col] = "target"

    data = data.rename(columns=rename_dict)

    # ✅ التحقق من توفر الأعمدة المطلوبة
    if "source" not in data.columns or "target" not in data.columns:
        st.error("🚨 **خطأ:** لم يتم العثور على أعمدة `source` و `target`. يرجى التأكد من صحة ملف CSV.")
    else:
        # 🔗 تحليل الشبكة الاجتماعية
        st.subheader("📡 **تحليل بنية الشبكة**")
        G = nx.from_pandas_edgelist(data, source="source", target="target")

        # 📊 استخراج ميزات الشبكة
        degrees = dict(G.degree())
        features = np.array(list(degrees.values())).reshape(-1, 1)

        # 🤖 نموذج Autoencoder لاكتشاف الشذوذ
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

        # 🚨 تحديد الحسابات المشبوهة
        anomalies = np.argsort(reconstruction_errors.flatten())[-5:]

        st.subheader("🚨 **العقد الشاذة المكتشفة**")
        st.warning("🔴 **هذه هي الحسابات الأكثر شذوذًا بناءً على تحليل الشبكة:**")
        st.write(anomalies)

        # 🎨 تحسين رسم الشبكة
        plt.figure(figsize=(10, 6))
        nx.draw(G, with_labels=True, node_size=500, node_color="#FF5733", edge_color="gray", font_size=10)
        st.pyplot(plt)

        # 📊 **إضافة لوحة تحكم (Dashboard)**
        st.subheader("📈 **إحصائيات الشبكة**")

        # 🔢 عدد العقد (الحسابات) وعدد الروابط
        num_nodes = len(G.nodes)
        num_edges = len(G.edges)

        # 🔍 عدد الحسابات المشبوهة
        num_anomalies = len(anomalies)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box"><h3>🔗 عدد الحسابات</h3><h2>{}</h2></div>'.format(num_nodes), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><h3>🔄 عدد الروابط</h3><h2>{}</h2></div>'.format(num_edges), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box"><h3>🚨 الحسابات المشبوهة</h3><h2>{}</h2></div>'.format(num_anomalies), unsafe_allow_html=True)

        # 📊 **رسم بياني يوضح توزيع الحسابات المشبوهة**
        anomaly_data = pd.DataFrame({"حسابات": list(anomalies), "خطأ الإعادة": reconstruction_errors.flatten()[anomalies]})
        fig = px.bar(anomaly_data, x="حسابات", y="خطأ الإعادة", title="🔍 توزيع الحسابات المشبوهة")
        st.plotly_chart(fig)

        # ✅ **إضافة زر لتحميل الشبكة بعد معالجتها**
        st.subheader("📤 **تحميل بيانات الشبكة بعد المعالجة**")
        processed_data = pd.DataFrame({"Source": data["source"], "Target": data["target"]})
        st.download_button("📥 تحميل الشبكة", processed_data.to_csv(index=False), "processed_network.csv", "text/csv")

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# 🎨 إعداد الصفحة
st.set_page_config(page_title="كشف الحسابات المشبوهة", layout="wide")
st.title("🔍 كشف الحسابات الشاذة في الشبكات الاجتماعية")

# 📂 تحميل ملف CSV
st.sidebar.header("📂 تحميل بيانات الشبكة")
uploaded_file = st.sidebar.file_uploader("اختر ملف الشبكة (CSV)", type=["csv"])

if uploaded_file:
    # 📊 قراءة البيانات
    data = pd.read_csv(uploaded_file)
    st.write("✅ **تم تحميل البيانات بنجاح!**")
    st.dataframe(data.head())

    # 🕸️ إنشاء الشبكة
    try:
        source_col, target_col = data.columns[:2]
        G = nx.from_pandas_edgelist(data, source=source_col, target=target_col)

        # 📌 تصفية العقد (عرض جزء من الشبكة)
        nodes_list = list(G.nodes)
        selected_nodes = st.sidebar.multiselect("📌 اختر العقد لعرضها", nodes_list, default=nodes_list[:10])

        # ✅ إنشاء شبكة مصغرة تحتوي على العقد المحددة
        subgraph = G.subgraph(selected_nodes)

        # 🔴 كشف الشذوذ باستخدام Isolation Forest
        node_features = pd.DataFrame(subgraph.degree(), columns=['node', 'degree'])
        model = IsolationForest(contamination=0.1)
        node_features["anomaly"] = model.fit_predict(node_features[['degree']])

        # 🎨 تحديد الألوان (الأحمر للشذوذ، الأزرق للعقد العادية)
        color_map = ['red' if anomaly == -1 else 'blue' for anomaly in node_features["anomaly"]]

        # 📉 رسم الشبكة باستخدام NetworkX
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color=color_map, edge_color='gray', font_size=10)
        st.pyplot(fig)

        st.success(f"🔴 تم اكتشاف {sum(node_features['anomaly'] == -1)} حسابات شاذة")

        # 📊 **إضافة رسم بياني لعدد الحسابات الشاذة والعادية**
        anomaly_count = node_features["anomaly"].value_counts()
        fig_bar = go.Figure(data=[go.Bar(
            x=["عقد عادية", "عقد شاذة"],
            y=[anomaly_count.get(1, 0), anomaly_count.get(-1, 0)],
            marker_color=['blue', 'red']
        )])
        fig_bar.update_layout(title="📊 توزيع الحسابات العادية والشاذة", xaxis_title="النوع", yaxis_title="العدد")
        st.plotly_chart(fig_bar)

        # 📝 تحميل تقرير الحسابات المشبوهة
        if st.button("📥 تحميل قائمة الحسابات الشاذة"):
            anomaly_df = node_features[node_features["anomaly"] == -1]
            anomaly_df.to_csv("anomalies.csv", index=False)
            st.download_button(label="تحميل CSV", data=anomaly_df.to_csv().encode(), file_name="anomalies.csv")

    except Exception as e:
        st.error(f"❌ حدث خطأ: {e}")

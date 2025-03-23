import random

def generate_fake_csv():
    users = [f"user_{i}" for i in range(1, 51)]  # 50 مستخدم
    connections = []

    for _ in range(100):  # 100 علاقة بين المستخدمين
        u1, u2 = random.sample(users, 2)
        connections.append([u1, u2])

    df_fake = pd.DataFrame(connections, columns=["source", "target"])
    df_fake.to_csv("fake_network.csv", index=False)
    return df_fake

if st.sidebar.button("📥 تحميل ملف تجريبي"):
    df = generate_fake_csv()
    st.write("✅ **تم إنشاء ملف بيانات تجريبي بنجاح!**")
    st.dataframe(df)
    st.download_button("📥 تحميل ملف CSV", df.to_csv(index=False), "fake_network.csv", "text/csv")

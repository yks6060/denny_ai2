import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 1. CSV 파일 업로드 ====================
st.sidebar.header("📁 CSV 파일 업로드")
uploaded_file = st.sidebar.file_uploader("새로운 데이터를 업로드하세요 (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ 업로드된 파일로 모델 재학습 완료")
else:
    data = pd.read_csv("ai_po.csv")
    st.sidebar.info("기본 파일(ai_po.csv)을 사용합니다")

# ==================== 2. 전처리 ====================
required_cols = {'ITEM', 'BRAND', 'PURCHASEPRICE', 'MARKUP'}
if not required_cols.issubset(data.columns):
    st.error("❌ CSV에 ITEM, BRAND, PURCHASEPRICE, MARKUP 컬럼이 필요합니다")
    st.stop()

data = data.dropna(subset=['ITEM', 'BRAND', 'PURCHASEPRICE', 'MARKUP'])
data['MARKUP'] = data['MARKUP'].astype(str).str.replace(',', '', regex=False)
data['MARKUP'] = pd.to_numeric(data['MARKUP'], errors='coerce')
data['PURCHASEPRICE'] = pd.to_numeric(data['PURCHASEPRICE'], errors='coerce')
data = data[(data['MARKUP'] >= 1) & (data['MARKUP'] < 20)]

# ==================== 3. 모델 학습 ====================
X = data[['ITEM']]
y = data['MARKUP']
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ==================== 4. UI 필터 ====================
st.title("📈 브랜드 / ITEM / 구매가격 기반 마진율 예측")

brand_list = sorted(data['BRAND'].unique())
brand_list_with_all = ['전체'] + brand_list
selected_brands = st.multiselect("브랜드 선택", brand_list_with_all, default='전체')

if '전체' in selected_brands:
    filtered_data = data.copy()
else:
    filtered_data = data[data['BRAND'].isin(selected_brands)]

filtered_items = sorted(filtered_data['ITEM'].unique())
item_list_with_all = ['전체'] + filtered_items
selected_items = st.multiselect("ITEM 선택", item_list_with_all, default='전체')

min_price = int(filtered_data['PURCHASEPRICE'].min())
max_price = int(filtered_data['PURCHASEPRICE'].max())
price_range = st.slider("PURCHASEPRICE 범위 (원)", min_price, max_price, (min_price, max_price), step=1000)

price_filtered_data = filtered_data[
    (filtered_data['PURCHASEPRICE'] >= price_range[0]) & 
    (filtered_data['PURCHASEPRICE'] <= price_range[1])
]

if '전체' in selected_items:
    items_to_predict = sorted(price_filtered_data['ITEM'].unique())
else:
    items_to_predict = [item for item in selected_items if item in price_filtered_data['ITEM'].values]

# ==================== 5. 예측 및 결과 ====================
if items_to_predict:
    results = []
    for item in items_to_predict:
        item_rows = price_filtered_data[price_filtered_data['ITEM'] == item]
        avg_price = item_rows['PURCHASEPRICE'].mean()
        input_df = pd.DataFrame([[item]], columns=['ITEM'])
        input_encoded = encoder.transform(input_df).toarray()
        predicted = model.predict(input_encoded)[0]
        results.append((item, avg_price, predicted))

    results_df = pd.DataFrame(results, columns=['ITEM', 'PURCHASEPRICE', 'PREDICTED_MARKUP'])

    st.subheader("📄 예측 결과")
    st.dataframe(results_df)

    # ==================== 6. 그래프 (트렌드선 포함) ====================
    st.subheader("📊 구매가격별 예측 마진율 + 트렌드선")
    x = results_df['PURCHASEPRICE'].values.reshape(-1, 1)
    y = results_df['PREDICTED_MARKUP'].values
    reg = LinearRegression().fit(x, y)
    y_trend = reg.predict(x)

    fig, ax = plt.subplots()
    ax.scatter(x, y, label='예측값', color='blue')
    ax.plot(x, y_trend, color='red', linestyle='--', label='트렌드선')
    for i in range(len(results_df)):
        ax.text(x[i][0], y[i] + 0.1, results_df['ITEM'][i], ha='center', fontsize=8)
    ax.set_xlabel("구매가격 (PURCHASEPRICE)")
    ax.set_ylabel("예상 마진율 (MARKUP)")
    ax.set_title("구매가격별 예측 마진율")
    ax.legend()
    st.pyplot(fig)

    # ==================== 7. CSV 다운로드 ====================
    st.markdown("### ⬇️ 예측 결과 저장")
    csv_data = results_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 예측 결과 CSV 다운로드",
        data=csv_data,
        file_name="predicted_markup.csv",
        mime="text/csv"
    )

# ==================== 8. 모델 성능 표시 ====================
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
r2 = r2_score(y_test, model.predict(X_test))
st.caption(f"📊 모델 성능 → RMSE: {rmse:.2f}, R²: {r2:.2f}")

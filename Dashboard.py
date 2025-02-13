import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import folium
from streamlit_folium import folium_static
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sidebar menu
st.sidebar.title("Pilih Program")
menu = st.sidebar.selectbox("Pilih fitur:", ["Sentimen", "Peta Interaktif", "Prediksi Penjualan", "Pola Pembelian", "Korelasi", "Clustering"])

# Tampilan berdasarkan menu yang dipilih
if menu == "Sentimen":
    st.title("10123115 - Muhammad Favian Jiwani")
    st.title("Analisis Sentimen Review Produk")
    # Function analisis sentimen
    # Load dataset
    order_reviews = pd.read_csv("order_reviews_dataset.csv")
    
    # Remove missing values and duplicates
    order_reviews.dropna(subset=['review_comment_message'], inplace=True)
    order_reviews.drop_duplicates(inplace=True)
    
    # Function analisis sentimen
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity
    
    order_reviews['sentiment_score'] = order_reviews['review_comment_message'].apply(get_sentiment)
    order_reviews['sentiment_label'] = order_reviews['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    
    # Visualisasi Distribusi Sentimen
    st.subheader("Distribusi Sentimen")
    fig, ax = plt.subplots(figsize=(6,4))
    ax = order_reviews['sentiment_label'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    st.pyplot(fig)
    
    # Analisis Tren Sentimen dari Waktu ke Waktu
    if 'review_creation_date' in order_reviews.columns:
        order_reviews['review_creation_date'] = pd.to_datetime(order_reviews['review_creation_date'])
        sentiment_trend = order_reviews.groupby(order_reviews['review_creation_date'].dt.to_period("M")).sentiment_score.mean()
        
        st.subheader("Tren Sentimen Pelanggan")
        fig, ax = plt.subplots(figsize=(10,5))
        sentiment_trend.plot(kind='line', marker='o', linestyle='-', ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Average Sentiment Score")
        ax.set_title("Trend of Customer Sentiment Over Time")
        st.pyplot(fig)

elif menu == "Peta Interaktif":
    st.title("10123090 - Raihan Fathir Muhammad")
    # Fungsi untuk Loading Data
    def load_data():
        # Load dataset
        customers = pd.read_csv('customers_dataset.csv')
        geolocations = pd.read_csv('geolocation_dataset.csv')
        order_items = pd.read_csv('order_items_dataset.csv')
        order_dataset = pd.read_csv('orders_dataset.csv')
        seller_dataset = pd.read_csv('sellers_dataset.csv')

        # Hapus duplikat pada dataset geolocation
        geolocations = geolocations.drop_duplicates(subset=['geolocation_zip_code_prefix'])

        # Merge customer dengan geolocation berdasarkan zip_code_prefix
        customers_geo = pd.merge(customers, geolocations, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left').drop(columns=['geolocation_zip_code_prefix'])

        # Merge seller dengan geolocation untuk mendapatkan latitude & longitude
        sellers_geo = pd.merge(seller_dataset, geolocations, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left').drop(columns=['geolocation_zip_code_prefix'])

        # Pastikan seller_geo memiliki lat dan lng
        sellers_geo.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)

        # Merge customer dengan order
        CustomersOrders = pd.merge(customers_geo, order_dataset, on='customer_id')

        # Merge dengan order_items
        CustomerOrdersItems = pd.merge(order_items, CustomersOrders, on='order_id')

        # Merge dengan seller (yang sudah memiliki lat-lng)
        CustomerSeller = pd.merge(sellers_geo, CustomerOrdersItems, on='seller_id')

        # Tambahkan kolom 'WilayahSama' untuk cek apakah pelanggan dan penjual dari wilayah yang sama
        CustomerSeller['WilayahSama'] = CustomerSeller['customer_state'] == CustomerSeller['seller_state']

        return CustomerSeller

    # Load data
    data = load_data()

    # Streamlit Title
    st.title("Visualisasi Perbandingan Wilayah Pembeli dan Penjual")
    st.write("Menampilkan analisis apakah pelanggan cenderung membeli dari penjual di wilayah yang sama.")

    # Hitung persentase
    JumlahWilayahSama = data['WilayahSama'].sum()
    JumlahPembelian = len(data)
    PersentaseWilayahSama = (JumlahWilayahSama / JumlahPembelian) * 100

    # Tampilkan statistik
    st.write(' ')
    st.subheader("Statistik Wilayah Sama vs Wilayah Berbeda")
    st.write(f"**Total Pembelian:** {JumlahPembelian}")
    st.write(f"**Wilayah Sama:** {JumlahWilayahSama} ({PersentaseWilayahSama:.2f}%)")
    st.write(f"**Wilayah Berbeda:** {JumlahPembelian - JumlahWilayahSama} ({100 - PersentaseWilayahSama:.2f}%)")
    st.write(' ')

    # Visualisasi Perbandingan Wilayah Sama vs Wilayah Berbeda
    st.write('Pie Chart')
    fig, ax = plt.subplots()
    ax.pie(
        [JumlahWilayahSama, JumlahPembelian - JumlahWilayahSama],
        labels=['Wilayah Sama', 'Wilayah Berbeda'],
        autopct='%1.1f%%',
        colors=['skyblue', 'orange']
    )
    ax.set_title('Perbandingan Wilayah Sama vs Wilayah Berbeda')
    st.pyplot(fig)

    # Tampilkan peta distribusi pelanggan & penjual
    st.subheader("Peta Distribusi Pelanggan & Penjual")
    st.write('Merah -> Penjual')
    st.write('Biru -> Pelanggan')

    # Hapus data Null/Nan
    data_clean = data.dropna(subset=['geolocation_lat', 'geolocation_lng', 'seller_lat', 'seller_lng'])

    # Membatasi jumlah data yang akan ditampilkan di peta
    sample_data = data_clean.sample(n=min(2000, len(data_clean)), random_state=42)

    # Buat peta
    map = folium.Map(location=[sample_data.iloc[0]['geolocation_lat'], sample_data.iloc[0]['geolocation_lng']], zoom_start=5)

    # Tambahkan titik pelanggan dan penjual
    for idx, row in sample_data.iterrows():
        # Marker pelanggan
        folium.Circle(
            location=[row['geolocation_lat'], row['geolocation_lng']],
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            popup=f"Pelanggan: {row['customer_city']}, {row['customer_state']}"
        ).add_to(map)

        # Marker penjual
        folium.Circle(
            location=[row['seller_lat'], row['seller_lng']],
            radius=3,
            color="red",
            fill=True,
            fill_color="red",
            popup=f"Penjual: {row['seller_city']}, {row['seller_state']}"
        ).add_to(map)

    # Tampilkan peta di Streamlit
    folium_static(map)

elif menu == "Prediksi Penjualan":
    st.title("10123104 - Muhammad Ilyasa Ar'rahman")
    st.title("Prediksi Penjualan 12 Bulan ke Depan dengan LSTM")

    # Load dataset langsung dari file CSV
    orders = pd.read_csv("orders_dataset.csv", parse_dates=['order_purchase_timestamp'])
    order_items = pd.read_csv("order_items_dataset.csv")

    # Merge dataset
    sales_data = pd.merge(order_items, orders, on="order_id")
    sales_data = sales_data[sales_data["order_status"] == "delivered"]
    sales_data["total_price"] = sales_data["price"]
    sales_data['year'] = sales_data['order_purchase_timestamp'].dt.year
    sales_data['month'] = sales_data['order_purchase_timestamp'].dt.month

    # Agregasi penjualan per bulan
    monthly_sales = sales_data.groupby(['year', 'month'])[['total_price']].sum().reset_index()
    monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
    monthly_sales.set_index('date', inplace=True)
    monthly_sales = monthly_sales[['total_price']]

    # Normalisasi data
    scaler = MinMaxScaler()
    monthly_sales_scaled = scaler.fit_transform(monthly_sales)

    # Data untuk LSTM
    window_size = 12
    X, y = [], []
    for i in range(len(monthly_sales_scaled) - window_size):
        X.append(monthly_sales_scaled[i:i+window_size])
        y.append(monthly_sales_scaled[i+window_size])
    X, y = np.array(X), np.array(y)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model LSTM
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    # Prediksi 12 bulan ke depan
    future_predictions = []
    last_window = monthly_sales_scaled[-window_size:]

    for _ in range(12):
        next_pred = model.predict(last_window.reshape(1, window_size, 1))
        future_predictions.append(next_pred[0, 0])
        last_window = np.append(last_window[1:], next_pred).reshape(window_size, 1)

    # Transform kembali ke skala asli
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Tanggal prediksi
    future_dates = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

    # Visualisasi
    st.subheader("Visualisasi Prediksi Penjualan")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_sales.index, monthly_sales['total_price'], marker='o', linestyle='-', label="Data Historis", color="blue")
    ax.plot(future_dates, future_predictions, marker='o', linestyle='--', label="Prediksi Masa Depan", color="red")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Total Penjualan")
    ax.set_title("Prediksi Penjualan 12 Bulan ke Depan")
    ax.legend()
    st.pyplot(fig)

elif menu == "Pola Pembelian":
    st.title("10123091 - Alghifari Raspati")
    st.title("Analisis Pola Pembelian")
    # === 1. Memuat Dataset ===
    product_df = pd.read_csv('products_dataset.csv')
    order_df = pd.read_csv('order_items_dataset.csv')
    payment_df = pd.read_csv('order_payments_dataset.csv')

    # === 2. Preprocessing Data ===
    # Konversi tipe data untuk konsistensi
    for df in [product_df, order_df, payment_df]:
        for col in ['product_id', 'order_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # Hapus duplikasi produk
    product_df.drop_duplicates(subset='product_id', inplace=True)

    # === 3. Gabungkan Dataset ===
    # Gabungkan order_df dengan product_df berdasarkan product_id
    combined_df = order_df.merge(product_df, on='product_id', how='inner')

    # Gabungkan dengan payment_df berdasarkan order_id
    combined_df = combined_df.merge(payment_df, on='order_id', how='inner')

    # === 4. Mengelompokkan Data ===
    # Periksa apakah kolom yang dibutuhkan ada
    required_cols = {'product_category_name', 'payment_type'}
    if not required_cols.issubset(combined_df.columns):
        raise ValueError(f"Kolom yang dibutuhkan tidak ditemukan: {required_cols - set(combined_df.columns)}")

    # Group by order_id dan gabungkan kategori produk serta metode pembayaran
    grouped_data = combined_df.groupby('order_id')[['product_category_name', 'payment_type']].agg(list).reset_index()

    # Gabungkan kategori produk dan metode pembayaran dalam satu transaksi
    grouped_data['items'] = grouped_data.apply(lambda row: row['product_category_name'] + row['payment_type'], axis=1)

    # === 5. Encoding Transaksi ===
    # Pastikan semua elemen dalam 'items' adalah string
    grouped_data['items'] = grouped_data['items'].apply(lambda x: [str(item) for item in x if pd.notna(item)])

    te = TransactionEncoder()
    te_ary = te.fit(grouped_data['items']).transform(grouped_data['items'])
    df_te = pd.DataFrame(te_ary, columns=te.columns_)

    # === 6. Menjalankan Apriori untuk Frequent Itemsets ===
    # Jalankan apriori untuk menemukan itemset yang sering muncul
    frequent_itemsets = apriori(df_te, min_support=0.02, use_colnames=True)

    # Buat aturan asosiasi berdasarkan confidence
    association_results = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Ambil 10 aturan asosiasi dengan confidence tertinggi
    top_10_rules = association_results.sort_values(by='confidence', ascending=False).head(10)

    # Ubah antecedents dan consequents menjadi string untuk visualisasi
    top_10_rules['antecedents'] = top_10_rules['antecedents'].apply(lambda x: ', '.join(x))
    top_10_rules['consequents'] = top_10_rules['consequents'].apply(lambda x: ', '.join(x))

    # === 7. Visualisasi Hasil dengan Matplotlib ===
    # Buat figure dan axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Warna berdasarkan support menggunakan colormap
    colors = plt.cm.plasma(top_10_rules['support'] / max(top_10_rules['support']))

    # Buat bar chart
    bars = ax.bar(top_10_rules['antecedents'], top_10_rules['confidence'], color=colors)

    # Tambahkan nilai confidence di atas batang
    for bar, confidence in zip(bars, top_10_rules['confidence']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{confidence:.2%}", ha='center', va='bottom', fontsize=10)

    # Konfigurasi sumbu dan judul
    ax.set_xlabel("Kategori Produk", fontsize=12)
    ax.set_ylabel("Confidence Level", fontsize=12)
    ax.set_title("Aturan Asosiasi: Produk vs Pembayaran", fontsize=14)

    # Atur posisi label sumbu x
    ax.set_xticks(range(len(top_10_rules['antecedents'])))
    ax.set_xticklabels(top_10_rules['antecedents'], rotation=45, ha='right')

    # Tambahkan colorbar untuk support
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(top_10_rules['support']), vmax=max(top_10_rules['support'])))
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=ax)  
    cbar.set_label("Support")

    # Tampilkan plot
    st.pyplot(fig)

elif menu == "Korelasi":
    st.title("10123113 - Muhammad Agung Hidayah")
    st.title('Korelasi Antara lama pengiriman dengan ulasan pembeli')

    # Load Data
    def load_data():
        orders = pd.read_csv("orders_dataset.csv", parse_dates=['order_purchase_timestamp', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
        order_reviews = pd.read_csv("order_reviews_dataset.csv")
        return orders, order_reviews

    orders, order_reviews = load_data()

    # Menggabungkan orders dengan order_reviews
    merged_data = pd.merge(orders, order_reviews, on='order_id', how='inner')
    selected_columns = [
        'order_id', 
        'order_purchase_timestamp',
        'order_delivered_carrier_date',
        'order_delivered_customer_date', 
        'order_estimated_delivery_date', 
        'order_status', 
        'review_score' 
    ]
    merged_data = merged_data[selected_columns]

    # Menghitung lama pengiriman dan keterlambatan
    merged_data['delivery_time'] = (merged_data['order_delivered_customer_date'] - merged_data['order_purchase_timestamp']).dt.days
    merged_data['delay'] = (merged_data['order_delivered_customer_date'] - merged_data['order_estimated_delivery_date']).dt.days
    merged_data['delay'] = merged_data['delay'].apply(lambda x: x if x > 0 else 0)

    # Kategorisasi waktu pengiriman
    bins = [0, 3, 7, 14, 30, 60] 
    labels = ['0-3 hari', '4-7 hari', '8-14 hari', '15-30 hari', '>30 hari']
    merged_data['delivery_time_category'] = pd.cut(merged_data['delivery_time'], bins=bins, labels=labels, right=False)

    # Hitung rata-rata skor ulasan berdasarkan kategori lama pengiriman
    review_by_delivery_time = merged_data.groupby('delivery_time_category', observed=True)['review_score'].mean().reset_index()

    # Visualisasi data
    st.subheader("Rata-Rata Skor Ulasan Berdasarkan Lama Pengiriman")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(review_by_delivery_time['delivery_time_category'], review_by_delivery_time['review_score'], color='steelblue')
    ax.set_title('Rata-Rata Skor Ulasan Berdasarkan Lama Pengiriman', fontsize=14)
    ax.set_xlabel('Kategori Lama Pengiriman', fontsize=12)
    ax.set_ylabel('Rata-Rata Skor Ulasan', fontsize=12)
    ax.set_xticklabels(review_by_delivery_time['delivery_time_category'], rotation=45)
    st.pyplot(fig)

elif menu == "Clustering":
    st.title("10123122 - Samuel Tigor H.S.")
    st.title('Clustering Pelanggan berdasarkan Pola Pembelian')
    
    # Load dataset
    file_path = "order_items_dataset.csv"
    df = pd.read_csv(file_path)

    # Agregasi data berdasarkan customer_id (jumlah transaksi dan total belanja)
    df_grouped = df.groupby("order_id").agg({"price": "sum", "order_item_id": "count"}).reset_index()
    df_grouped.columns = ["customer_id", "total_spent", "total_orders"]

    # Normalisasi data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_grouped[["total_spent", "total_orders"]])

    # Menentukan jumlah klaster menggunakan metode Elbow
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Visualisasi Elbow Method
    st.subheader("Elbow Method for Optimal K")
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal K')
    st.pyplot(fig)

    # Menjalankan K-Means dengan jumlah klaster optimal (misal k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_grouped["cluster"] = kmeans.fit_predict(df_scaled)

    # Visualisasi hasil clustering
    st.subheader("Customer Segmentation")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_grouped["total_spent"], y=df_grouped["total_orders"], hue=df_grouped["cluster"], palette="viridis", ax=ax)
    ax.set_xlabel("Total Spent")
    ax.set_ylabel("Total Orders")
    ax.set_title("Customer Segmentation based on Purchase Patterns")
    st.pyplot(fig)
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """
    Memuat dataset dari path file yang ditentukan.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {file_path}")
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)


def _is_string_col(series):
    """Cek apakah kolom bertipe string/object (kompatibel Pandas 2.x dan 3.x)."""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def preprocess_data(df):
    """
    Melakukan data preprocessing otomatis meliputi:
    - Konversi tipe data TotalCharges dan imputasi median
    - Konversi target Churn ke binary (0 dan 1)
    - Penghapusan fitur ID yang tidak relevan
    - Encoding variabel kategorikal binary (Label Encoding)
    - Encoding variabel kategorikal multi-class (One-Hot Encoding)
    - Standardisasi fitur numerik (StandardScaler)
    """
    df_clean = df.copy()

    # 1. Mengubah TotalCharges ke numerik dan menangani NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    median_val = df_clean['TotalCharges'].median()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val)
    print("-> Selesai memproses tipe data TotalCharges dan menangani missing values.")

    # 2. Konversi target Churn ke numerik jika bertipe string
    if 'Churn' in df_clean.columns and _is_string_col(df_clean['Churn']):
        df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})
        print("-> Selesai konversi target 'Churn' ke binary (0/1).")

    # 3. Menghapus ID kolom jika ada
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        print("-> Selesai menghapus kolom 'customerID'.")

    # 4. Label Encoding untuk kolom kategorikal dengan 2 nilai unik
    le = LabelEncoder()
    binary_cols = [col for col in df_clean.columns
                   if _is_string_col(df_clean[col]) and df_clean[col].nunique() == 2]
    for col in binary_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    print(f"-> Selesai Label Encoding untuk kolom binary: {binary_cols}")

    # 5. One-Hot Encoding untuk kolom kategorikal multi-class
    multiclass_cols = [col for col in df_clean.columns if _is_string_col(df_clean[col])]
    if multiclass_cols:
        df_clean = pd.get_dummies(df_clean, columns=multiclass_cols, drop_first=True)
        # Pastikan kolom-kolom boolean hasil OHE diconvert ke 0/1 agar rapi
        bool_cols = df_clean.select_dtypes(include='bool').columns
        if len(bool_cols) > 0:
            df_clean[bool_cols] = df_clean[bool_cols].astype(int)
        print(f"-> Selesai One-Hot Encoding untuk kolom multi-class: {multiclass_cols}")

    # 6. Standardisasi fitur numerik
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    existing_num_cols = [col for col in num_cols if col in df_clean.columns]
    if existing_num_cols:
        df_clean[existing_num_cols] = scaler.fit_transform(df_clean[existing_num_cols])
        print(f"-> Selesai scaling data numerik untuk kolom: {existing_num_cols}")

    return df_clean


def save_data(df, output_path):
    """
    Menyimpan dataframe hasil preprocessing ke path tujuan.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset hasil preprocessing berhasil disimpan ke: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Otomatisasi Preprocessing Telco Customer Churn")
    parser.add_argument(
        "--input",
        type=str,
        default="../telco_customer_churn_dataset_raw.csv",
        help="Path ke file dataset mentah (raw)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../telco_customer_churn_dataset_preprocessing/preprocessed_churn.csv",
        help="Path tujuan file dataset hasil preprocessing"
    )
    args = parser.parse_args()

    # Jalankan alur preprocessing otomatis
    try:
        raw_df = load_data(args.input)
        processed_df = preprocess_data(raw_df)
        save_data(processed_df, args.output)
        print(f"\n=== PROSES PREPROCESSING SELESAI DENGAN SUKSES ===")
        print(f"    Shape akhir: {processed_df.shape}")
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses data: {e}")
        exit(1)


if __name__ == "__main__":
    main()

import pandas as pd

# Veriyi yükleme
file_path = "avocado.csv" 
df = pd.read_csv(file_path)

# Tarih formatını datetime'a çevirme ve ay bilgisi ekleme
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month

# Veri küpü oluşturma
data_cube = df.pivot_table(
    values=['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags'],  # Ölçütler
    index=['region', 'type'],  # Satır boyutları (bölge ve tür)
    columns=['year', 'month'],  # Sütun boyutları (yıl ve ay)
    aggfunc='mean',  # Özetleme işlevi: Ortalama
    fill_value=0  # Eksik değerleri doldurma
)

# Veri küpünü CSV formatında kaydetme
data_cube.to_csv("data_cube.csv")
print("Veri küpü 'data_cube.csv' olarak kaydedildi.")




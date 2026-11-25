# ==========================================================
# PRÁCTICA: IDENTIFICACIÓN DE PROBLEMAS EN LOS DATOS (TITANIC)
# Alumno: Hugo Fabián
# Número de Control: 0040
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Evita errores de ventana gráfica en Windows
import matplotlib.pyplot as plt
import seaborn as sns

# Detectar automáticamente la carpeta donde está este archivo .py
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_datos = os.path.join(carpeta_actual, "datos")
carpeta_resultados = os.path.join(carpeta_actual, "resultados")

os.makedirs(carpeta_datos, exist_ok=True)
os.makedirs(carpeta_resultados, exist_ok=True)

ruta_csv = os.path.join(carpeta_datos, "titanic.csv")

# Si no tienes titanic.csv descargado, lo descarga automático
if not os.path.exists(ruta_csv):
    print("📥 Descargando titanic.csv automáticamente...")
    df = sns.load_dataset("titanic")
    df.to_csv(ruta_csv, index=False)
else:
    df = pd.read_csv(ruta_csv)

print("====================================================")
print("Alumno: Hugo Fabián | No. Control: 0040")
print("🚀 Iniciando análisis del dataset Titanic...")
print("====================================================\n")

print("===== 1. VISTA PREVIA DE DATOS =====")
print(df.head(3), "\n")

print("===== 2. DIMENSIONES =====")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")

print("===== 3. TIPOS DE DATOS =====")
print(df.dtypes, "\n")

print("===== 4. DATOS NULOS (Cantidad) =====")
nulos = df.isnull().sum()
print(nulos[nulos > 0], "\n")

print("===== 5. PORCENTAJE DE NULOS =====")
porcentaje = (df.isnull().mean() * 100).round(2)
print(porcentaje[porcentaje > 0], "%\n")

duplicados = df.duplicated().sum()
print(f"===== 6. REGISTROS DUPLICADOS: {duplicados} =====\n")

print("===== 7. VALORES ATÍPICOS EN 'AGE' =====")
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
atipicos = df[(df['age'] < Q1 - 1.5 * IQR) | (df['age'] > Q3 + 1.5 * IQR)]
print(f"Se encontraron {len(atipicos)} valores atípicos en la edad.\n")

# Gráfica
ruta_grafica = os.path.join(carpeta_resultados, "nulos_titanic_0040.png")
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Mapa de calor de datos faltantes - Titanic (Hugo Fabián - 0040)")
plt.tight_layout()
plt.savefig(ruta_grafica)
plt.close()
print(f"✅ Gráfica guardada en: {ruta_grafica}")

# Limpieza
df_limpio = df.copy().drop_duplicates()
df_limpio['age'] = df_limpio['age'].fillna(df_limpio['age'].median())
df_limpio['embark_town'] = df_limpio['embark_town'].fillna(df_limpio['embark_town'].mode()[0])

ruta_limpio = os.path.join(carpeta_resultados, "titanic_limpio_0040.csv")
df_limpio.to_csv(ruta_limpio, index=False)
print(f"✅ Dataset limpio guardado en: {ruta_limpio}")
print("\n🎉 ¡Análisis completado con éxito!")

print("Hugo Fabian Antonio")
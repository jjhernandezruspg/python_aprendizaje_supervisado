import pandas as pd
from sklearn.linear_model import LinearRegression

datos = {
    'metros_cuadrados': [100, 150, 200, 120, 180, 250],
    'numero_habitaciones': [2, 3, 4, 2, 3, 4],
    'precio_miles': [250, 350, 450, 290, 400, 520]
}
df = pd.DataFrame(datos)

X = df[['metros_cuadrados', 'numero_habitaciones']]
y = df['precio_miles']

modelo_regresion = LinearRegression()
modelo_regresion.fit(X, y)

try:
    metros_cuadrados_nuevo = float(input("Ingresa los metros cuadrados de la casa: "))
    habitaciones_nuevo = int(input("Ingresa el número de habitaciones: "))
    
    nueva_casa = pd.DataFrame([[metros_cuadrados_nuevo, habitaciones_nuevo]], columns=['metros_cuadrados', 'numero_habitaciones'])
    
    precio_predicho = modelo_regresion.predict(nueva_casa)

    print(f"\nEl precio predicho para una casa de {metros_cuadrados_nuevo:.0f} m² y "
          f"{habitaciones_nuevo} habitaciones es: ${precio_predicho[0]:.2f} mil.")
    
    print("\n--- Explicación del resultado ---")
    print("Este resultado se basa en un modelo de **regresión lineal** de los datos históricos.")
    print("Se encontró la siguiente relación:")
    print(f"- Por cada metro cuadrado adicional, el precio aumenta en ${modelo_regresion.coef_[0]:.2f} mil.")
    print(f"- Por cada habitación adicional, el precio aumenta en ${modelo_regresion.coef_[1]:.2f} mil.")
    print(f"La predicción se calcula aplicando la relación en la base de conocimientos.")
except ValueError:
    print("Entrada inválida. Por favor, asegúrate de ingresar números.")
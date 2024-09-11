import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import streamlit as st

# Configuración de la página
st.set_page_config(layout="wide")

# Cargar datos
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

players = load_data("https://raw.githubusercontent.com/lbgabriel/TFM/main/TFM2.csv")

# Eliminar espacios en nombres de columnas
players.columns = players.columns.str.strip()

# Convertir la columna 'Date' a datetime
players['Date'] = pd.to_datetime(players['Date'], errors='coerce')

# Extraer solo la fecha (día, mes, año) sin hora y minutos
players['Date'] = players['Date'].dt.date

# Lista de columnas a procesar
columnas_procesar = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
columnas_prediccion = ['BLK', 'PTS', 'AST', 'DRB', 'TOV']

# Convertir columnas a numérico
for col in columnas_procesar:
    players[col] = pd.to_numeric(players[col], errors='coerce')

# Calcular la media por jugador y llenar valores NaN
promedio_por_jugador = players.groupby('Player Name')[columnas_procesar].transform(lambda x: x.fillna(x.mean()))
df_asis = players.copy()
df_asis[columnas_procesar] = df_asis[columnas_procesar].fillna(promedio_por_jugador)

# Función para calcular media acumulativa
def calcular_media_acumulativa(df):
    new_columns = []
    for col in columnas_procesar:
        new_col = col + '_mean'
        new_columns.append(df[col].expanding().mean().rename(new_col))
    new_df = pd.concat(new_columns, axis=1)
    return new_df

# Aplicación Streamlit
st.sidebar.title("Menú Principal")
menu_options = ["Inicio", "Historial", "Resultados", "Acerca de"]
choice = st.sidebar.selectbox("Selecciona una opción", menu_options)

# Equipo del Proyecto en el Sidebar
st.sidebar.title("Equipo del Proyecto")
st.sidebar.write("Luis Gabriel Leiva Baltodano [LinkedIn](https://www.linkedin.com)")
st.sidebar.write("Gian Franco Ramos  [LinkedIn](https://www.linkedin.com)")
st.sidebar.write("William Alexander Valero Alfonso [LinkedIn](https://www.linkedin.com)")

if choice == "Inicio":
    st.title("Predicciones de las estadísticas de los jugadores de la NBA")
    st.video("https://www.youtube.com/watch?v=4Y0Nwddjz0A")
    st.write("""
    ### ¿Qué hace la app?
    Esta aplicación toma los datos de los jugadores de la NBA, que se registran en cada partido jugado...
    """)

elif choice == "Historial":
    st.header("Historial de Jugadores")
    player_selected = st.sidebar.selectbox("Jugador", options=players['Player Name'].unique())
    filtered_data = players[players['Player Name'] == player_selected]
    
    if filtered_data.empty:
        st.write(f"No hay datos disponibles para {player_selected}.")    
    else:
        st.dataframe(filtered_data)

    st.write("### Glosario")
    glosario = """
    Rk -- Rank  
    G -- Games  
    MP -- Minutes Played  
    ...
    """
    st.markdown(glosario)

elif choice == "Resultados":
    st.header("Resultados del Modelo")

    predicciones = {}
    errores = {}

    player_selected = st.sidebar.selectbox("Jugador", options=players['Player Name'].unique())
    filtered_data = players[players['Player Name'] == player_selected]
    new_df = calcular_media_acumulativa(filtered_data)

    for column_selected in columnas_prediccion:
        columnas_disponibles = [col for col in columnas_procesar if col + '_mean' in new_df.columns]
        luka_work = new_df[[col + '_mean' for col in columnas_disponibles]]

        X, y = luka_work.drop(columns=[column_selected + '_mean']).values, luka_work[column_selected + '_mean'].values
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=42)

        imputer = SimpleImputer(strategy='mean')
        X_train2 = imputer.fit_transform(X_train2)
        X_test2 = imputer.transform(X_test2)
        y_train2 = imputer.fit_transform(y_train2.reshape(-1, 1)).ravel()
        y_test2 = imputer.transform(y_test2.reshape(-1, 1)).ravel()

        # Gradient Boosting
        modelo = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        modelo.fit(X_train2, y_train2)
        gb_predictions = modelo.predict(X_test2)

        # Reemplazar MAE por MSE en la evaluación
        gb_mse = mean_squared_error(y_test2, gb_predictions)
        errores[column_selected] = gb_mse

        # Redondear las predicciones a un decimal
        predicciones[column_selected] = np.round(gb_predictions, 1)

    st.header("Predicciones del Modelo")
    
    # Colocar las métricas en filas horizontales
    columns = st.columns(len(columnas_prediccion))
    for idx, column_selected in enumerate(columnas_prediccion):
        with columns[idx]:
            st.subheader(column_selected)
            real_vals = filtered_data[column_selected].dropna().values
            pred_vals = predicciones[column_selected]

            # Alinear las longitudes para la comparación
            if len(pred_vals) > len(real_vals):
                pred_vals = pred_vals[:len(real_vals)]
            elif len(real_vals) > len(pred_vals):
                real_vals = real_vals[-len(pred_vals):]

            comparison_df = pd.DataFrame({
                'Fecha': filtered_data['Date'].iloc[-len(pred_vals):],
                'Real': real_vals,
                'Predicción': pred_vals
            })

            st.dataframe(comparison_df)
            st.metric(label=f"Última Predicción ({column_selected})", value=f"{pred_vals[-1]:.1f}")
            st.metric(label=f"MSE ({column_selected})", value=f"{errores[column_selected]:.1f}")

elif choice == "Acerca de":
    st.header("Acerca de")
    st.write("""
    La base de datos usada tiene información de las temporadas 2020-2021 hasta el inicio de los Playoff 2024...
    """)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
#import streamlit as st


# Configuración de la página
st.set_page_config(layout="wide")

# Cargar datos
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

players = load_data("C:\\Users\\gabol\\Desktop\\TFM_NBA\\DF-PLAYERS.csv")

# Eliminar espacios en nombres de columnas
players.columns = players.columns.str.strip()

# Convertir la columna 'Date' a datetime
players['Date'] = pd.to_datetime(players['Date'], errors='coerce')

# Extraer solo la fecha (día, mes, año) sin hora y minutos
players['Date'] = players['Date'].dt.date

# Lista de columnas a procesar
columnas_procesar = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-']
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
menu_options = ["Inicio", "Historial", "Resultados", "Gráficos", "Acerca de"]
choice = st.sidebar.selectbox("Selecciona una opción", menu_options)

if choice == "Inicio":
    st.title("Predicciones de las estdisticas de los jugadores de la NBA")
    st.video("https://www.youtube.com/watch?v=4Y0Nwddjz0A")
    st.write("""
    ### ¿Qué hace la app?
    Esta aplicación toma los datos de los jugadores de la NBA, que se registran en cada partido jugado, como los minutos por partido, rebotes, tiros encestados, tiros fallados, rebotes ofensivos y defensivos, tiros libres, tiros de dos y tres puntos, asistencias, robos, entre otros, para poder determinar las predicciones de su desempeño en el siguiente partido.
    
    ### ¿Cómo se hace esto?
    Primero, buscamos que los datos tengan relación entre sí, como sucede con los datos de cada partido por jugador, que sean actualizados y además de acceso público, como lo son los datos de la NBA, los cuales se actualizan diariamente y en tiempo real durante los partidos.
    
    Ahora, sabiendo de dónde viene la información y conociendo que es una fuente primaria, tenemos la viabilidad de los datos, lo que aporta un valor real a la aplicación del modelo.
    
    Pasamos a los siguientes pasos que son el análisis de los datos y las iteraciones de estos mismos. La necesidad de iterar en la predicción de series temporales surge de la naturaleza secuencial y dependiente de los datos temporales. Cada paso predictivo se basa en el paso anterior, permitiendo extender las predicciones hacia el futuro de manera coherente y efectiva, a pesar de no tener las futuras X disponibles inicialmente.
    """)

elif choice == "Historial":
    st.header("Historial de Jugadores")
    player_selected = st.sidebar.selectbox("Jugador", options=players['Player Name'].unique())
    filtered_data = players[players['Player Name'] == player_selected]
    
    if filtered_data.empty:
        st.write(f"No hay datos disponibles para {player_selected}.")    
    else:
        st.dataframe(filtered_data)

 # Glosario
    st.write("### Glosario")
    glosario = """
    **Rk** -- Rank  
    **G** -- Games  
    **MP** -- Minutes Played  
    **FG** -- Field Goals  
    **FGA** -- Field Goal Attempts  
    **FG%** -- Field Goal Percentage  
    **3P** -- 3-Point Field Goals  
    **3PA** -- 3-Point Field Goal Attempts  
    **3P%** -- 3-Point Field Goal Percentage  
    **2P** -- 2-Point Field Goals  
    **2PA** -- 2-point Field Goal Attempts  
    **2P%** -- 2-Point Field Goal Percentage  
    **FT** -- Free Throws  
    **FTA** -- Free Throw Attempts  
    **FT%** -- Free Throw Percentage  
    **ORB** -- Offensive Rebounds  
    **DRB** -- Defensive Rebounds  
    **TRB** -- Total Rebounds  
    **AST** -- Assists  
    **STL** -- Steals  
    **BLK** -- Blocks  
    **TOV** -- Turnovers  
    **PF** -- Personal Fouls  
    **PTS** -- Points  
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

        reg = LinearRegression()
        reg.fit(X_train2, y_train2)
        lr_predictions = reg.predict(X_test2)
        lr_mse = mean_squared_error(y_test2, lr_predictions)
        lr_r2 = r2_score(y_test2, lr_predictions)

        # Redondear las predicciones a un decimal
        predicciones[column_selected] = np.round(lr_predictions, 1)
        errores[column_selected] = (lr_mse, lr_r2)

    st.header("Predicciones del Modelo ")
    columns = st.columns(len(columnas_prediccion))
    for idx, column_selected in enumerate(columnas_prediccion):
        with columns[idx]:
            st.subheader(column_selected)
            pred_df = pd.DataFrame({
                'Fecha': filtered_data['Date'].iloc[-len(predicciones[column_selected]):],
                'Predicción': predicciones[column_selected],
                'Real': y_test2[:len(predicciones[column_selected])]
            })
            st.dataframe(pred_df)
            st.metric(label=f"Última Predicción ({column_selected})", value=f"{predicciones[column_selected][-1]:.1f}")
            st.metric(label=f"MSE ({column_selected})", value=f"{errores[column_selected][0]:.1f}")
            st.metric(label=f"R² ({column_selected})", value=f"{errores[column_selected][1]:.1f}")

elif choice == "Gráficos":
    st.header("Comparación de Predicciones")
    player_selected = st.sidebar.selectbox("Jugador", options=players['Player Name'].unique())
    filtered_data = players[players['Player Name'] == player_selected]
    new_df = calcular_media_acumulativa(filtered_data)

    predicciones = {}

    for column_selected in columnas_prediccion:
        columnas_disponibles = [col for col in columnas_procesar if col + '_mean' in new_df.columns]
        luka_work = new_df[[col + '_mean' for col in columnas_disponibles]]

        X, y = luka_work.drop(columns=[column_selected + '_mean']).values, luka_work[column_selected + '_mean'].values

        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        y = imputer.transform(y.reshape(-1, 1)).ravel()

        reg = LinearRegression()
        reg.fit(X, y)
        lr_predictions = reg.predict(X)

        predicciones[column_selected] = np.round(lr_predictions, 1)

    for column_selected in columnas_prediccion:
        real_vals = filtered_data[[column_selected, 'Date']].dropna()
        pred_vals = predicciones[column_selected]

        if len(pred_vals) == len(real_vals):
            fig = px.line(
                real_vals,
                x='Date',
                y=column_selected,
                title=f"{column_selected} - Real vs Predicción",
                labels={'value': column_selected, 'variable': 'Series'}
            )
            fig.add_scatter(
                x=real_vals['Date'],
                y=pred_vals,
                mode='lines',
                name=f'Predicción {column_selected}',
                line=dict(dash='dash')
            )
            st.plotly_chart(fig)
        else:
            st.write(f"Datos no coinciden en longitud para {column_selected}")

elif choice == "Acerca de":
    st.header("Acerca de")
    st.write("""
    La base de datos usada tiene información de las temporadas 2020-2021 hasta el inicio de los Playoff 2024.
             
    Detalle del Código:
    1. Carga y Preparación de Datos:
        - Carga de Datos: Se lee un archivo CSV que contiene datos de los jugadores de la NBA.
        - Limpieza de Datos: Se eliminan los espacios en blanco en los nombres de las columnas y se convierte la columna Date a formato datetime.
        - Procesamiento de Columnas: Se seleccionan columnas relevantes para el análisis, convirtiéndolas a valores numéricos y manejando valores faltantes al reemplazarlos por la media de cada jugador.
    2. Cálculo de Medias Acumulativas:
        - Función calcular_media_acumulativa: Esta función calcula la media acumulativa de las estadísticas de rendimiento de cada jugador para cada columna seleccionada. La media acumulativa ayuda a capturar la tendencia general del rendimiento del jugador a lo largo del tiempo.
    3. Configuración de la Interfaz de Usuario (UI):
        - Inicio: Muestra un video introductorio y proporciona información general sobre la aplicación.
        - Historial: Permite seleccionar un jugador y ver sus datos históricos.
        - Resultados: Para el jugador seleccionado, la aplicación entrena un modelo de regresión lineal para predecir varias estadísticas (BLK, PTS, AST, DRB, TOV). Los resultados de las predicciones, así como las métricas de error del modelo (MSE y R²), se muestran en la UI.
        - Gráficos: Compara las predicciones del modelo con los valores reales para las estadísticas seleccionadas, mostrando gráficos de líneas que visualizan las predicciones versus los valores reales.
        - Acerca de: Proporciona una breve descripción sobre el propósito y desarrollo de la aplicación.
    4. Modelo Predictivo:
        - Preparación de Datos para el Modelo: Los datos se dividen en conjuntos de entrenamiento y prueba. Los valores faltantes se manejan con la estrategia de imputación de la media.
        - Entrenamiento del Modelo: Se usa LinearRegression de scikit-learn para entrenar el modelo con los datos disponibles. Se evalúa el modelo con métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R²).
    """)

st.sidebar.title("Desarrolladores")
st.sidebar.write("[Geanfranco Ramos ](gframos138@gmail.com)")
st.sidebar.write("[Gabriel Leiva Baltodano](galeiva14@gmail.com)")
st.sidebar.write("[Alex Valero](alexvalero0488@gmail.com)")


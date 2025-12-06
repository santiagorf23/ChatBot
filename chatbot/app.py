import re
import unicodedata
import os
import pandas as pd
import json
import numpy as np
import requests
import psutil
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
CORS(app) 

# Configuración para archivos grandes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB límite
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- CREDENCIALES DE USUARIOS (hardcoded para simplicidad) ---
USUARIOS = {
    'usuario': {
        'user1': 'pass123',
        'user2': 'pass456',
        'usuario': 'usuario123'
    },
    'administrador': {
        'admin': 'admin123',
        'admin2': 'admin456',
        'root': 'root123'
    }
}

# --- DECORADOR PARA PROTEGER RUTAS ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


# Configuración de Groq API usando requests
API_KEY_GROQ = os.getenv("API_KEY_GROQ")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" 

class SimpleGroqClient:
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completions_create(self, model, messages, temperature=0.7, max_tokens=1024, response_format=None):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error en API Groq: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Respuesta error: {e.response.text}")
            raise

# Inicializar cliente
client = SimpleGroqClient(api_key=API_KEY_GROQ, api_url=GROQ_API_URL)

# Variables globales para DOCUMENTO DE SOPORTE (usuarios)
vectorizer_soporte = None
X_soporte = None
chunks_originales_soporte = []
chunks_normalizados_soporte = []

# Variables globales para DOCUMENTO DE ANÁLISIS (admins)
vectorizer_analisis = None
X_analisis = None
chunks_originales_analisis = []
chunks_normalizados_analisis = []
df_data = None 

# Variables para información del archivo de soporte (usuarios)
archivo_soporte = {
    'nombre': None,
    'tipo': None,
    'descripcion': None,
    'fecha_carga': None,
    'cargado_por': None
}

# Variables para información del archivo de análisis (admins)
archivo_analisis = {
    'nombre': None,
    'tipo': None,
    'descripcion': None,
    'fecha_carga': None,
    'cargado_por': None
}

# Ruta del documento por defecto para usuarios
DOCUMENTO_DEFAULT = os.path.join('uploads', 'documento_soporte_techcorp.txt')
 

# --- FUNCIONES AUXILIARES ---

def cargar_documento_default():
    """Carga el documento de soporte por defecto al iniciar el servidor (NO se reemplaza)."""
    global vectorizer_soporte, X_soporte, chunks_originales_soporte, chunks_normalizados_soporte, archivo_soporte
    
    # Verificar si existe el archivo por defecto
    if not os.path.exists(DOCUMENTO_DEFAULT):
        print("⚠️  No se encontró documento por defecto. Creando uno de ejemplo...")
        # Crear documento de ejemplo
        os.makedirs('uploads', exist_ok=True)
        with open(DOCUMENTO_DEFAULT, 'w', encoding='utf-8') as f:
            f.write("""DOCUMENTO DE SOPORTE - LOS ARTIFICIALES

Bienvenido al sistema de soporte de Los Artificiales.

PREGUNTAS FRECUENTES:

1. ¿Cómo resetear mi contraseña?
Para resetear tu contraseña, ve a la página de login y haz clic en "Olvidé mi contraseña". 
Recibirás un correo con instrucciones para crear una nueva contraseña.

2. ¿Cómo contactar con soporte técnico?
Puedes contactar con soporte técnico a través de:
- Email: soporte@losartificiales.com
- Teléfono: +1 (555) 123-4567
- Chat en vivo: Disponible de Lunes a Viernes, 9AM - 6PM

3. ¿Cómo actualizar mi perfil?
Para actualizar tu perfil:
1. Inicia sesión en tu cuenta
2. Ve a "Mi Perfil" en el menú superior
3. Haz clic en "Editar Perfil"
4. Actualiza la información necesaria
5. Guarda los cambios

4. ¿Qué navegadores son compatibles?
Nuestro sistema es compatible con:
- Google Chrome (versión 90+)
- Mozilla Firefox (versión 88+)
- Microsoft Edge (versión 90+)
- Safari (versión 14+)

5. ¿Cómo reportar un problema?
Para reportar un problema:
1. Ve a la sección "Ayuda"
2. Selecciona "Reportar Problema"
3. Describe el problema en detalle
4. Adjunta capturas de pantalla si es posible
5. Envía el reporte

POLÍTICAS:

- Todos los usuarios deben mantener sus credenciales seguras
- No compartir contraseñas con terceros
- Reportar cualquier actividad sospechosa inmediatamente
- Respetar las políticas de uso aceptable

CONTACTO:
Para más información, visita nuestra página web o contacta con nuestro equipo de soporte.
""")
        print("✅ Documento de ejemplo creado")
    
    try:
        print(f"📄 Cargando documento por defecto: {DOCUMENTO_DEFAULT}")
        
        # Leer el archivo
        with open(DOCUMENTO_DEFAULT, 'r', encoding='utf-8') as f:
            texto_empresa = f.read()
        
        if not texto_empresa:
            print("❌ El documento está vacío")
            return False
        
        # Procesar el documento de soporte
        cortar_en_trozos(texto_empresa, chunks_originales_soporte, chunks_normalizados_soporte)
        
        if not chunks_normalizados_soporte:
            print("❌ No se pudieron crear chunks")
            return False
        
        # Crear vectorizador para soporte
        vectorizer_soporte = TfidfVectorizer(max_features=1000)
        X_soporte = vectorizer_soporte.fit_transform(chunks_normalizados_soporte)
        
        # Generar descripción con IA
        try:
            preview = texto_empresa[:1500] if len(texto_empresa) > 1500 else texto_empresa
            prompt = f"Resume en una línea de máximo 100 caracteres de qué trata este documento:\n\n{preview}"
            descripcion_response = client.chat_completions_create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            descripcion = descripcion_response["choices"][0]["message"]["content"].strip()
        except:
            descripcion = "Documento de soporte y preguntas frecuentes"
        
        # Actualizar información del archivo de soporte
        from datetime import datetime
        archivo_soporte = {
            'nombre': 'documento_soporte_techcorp.txt',
            'tipo': 'TXT',
            'descripcion': descripcion,
            'fecha_carga': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'cargado_por': 'Sistema'
        }
        
        print(f"✅ Documento de soporte cargado exitosamente")
        print(f"   Chunks: {len(chunks_originales_soporte)}")
        print(f"   Descripción: {descripcion}")
        return True
        
    except Exception as e:
        print(f"❌ Error al cargar documento por defecto: {e}")
        return False

def get_memory_usage():
    """Obtiene el uso de memoria actual en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def normalize_text(text):
    """Convierte texto a minúsculas y elimina acentos y caracteres no alfanuméricos."""
    text = text.lower()
    # Eliminar acentos
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Mantener solo letras y espacios
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extraer_palabras_clave(texto):
    """Extrae palabras clave importantes del texto."""
    stop_words = {'y', 'o', 'de', 'la', 'el', 'en', 'que', 'los', 'las', 'un', 'una', 'con', 
                  'del', 'se', 'por', 'para', 'es', 'al', 'lo', 'como', 'mas', 'pero', 'sus'}
    palabras = texto.split()
    return [p for p in palabras if p not in stop_words and len(p) > 3]

# --- FUNCIONES DE PROCESAMIENTO DE CSV CON PUNTO Y COMA ---

def convertir_csv_a_texto(filepath):
    """Lee un archivo CSV con separador ; y lo convierte en texto."""
    global df_data
    try:
        print(f"Leyendo archivo CSV: {filepath}")
        
        # Detectar el delimitador
        with open(filepath, 'r', encoding='utf-8') as f:
            primera_linea = f.readline()
        
        # Verificar si usa punto y coma
        if ';' in primera_linea:
            print("Detectado separador: punto y coma (;)")
            df_data = pd.read_csv(filepath, encoding='utf-8', sep=';', low_memory=False)
        else:
            print("Usando separador por defecto: coma")
            df_data = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        
        print(f"CSV cargado: {len(df_data):,} filas, {len(df_data.columns)} columnas")
        print(f"Columnas: {list(df_data.columns)}")
        
        # Limpiar nombres de columnas (eliminar espacios)
        df_data.columns = df_data.columns.str.strip()
        
        # Crear resumen
        resumen = []
        resumen.append("RESUMEN DEL DATASET:")
        resumen.append(f"Total de registros: {len(df_data):,}")
        resumen.append(f"Total de columnas: {len(df_data.columns)}")
        
        resumen.append("\nCOLUMNAS DISPONIBLES:")
        for i, col in enumerate(df_data.columns):
            dtype = str(df_data[col].dtype)
            resumen.append(f"{i+1}. {col} (tipo: {dtype})")
        
        # Muestra de datos
        resumen.append("\nMUESTRA DE DATOS (primeras 3 filas):")
        resumen.append(df_data.head(3).to_string(index=False))
        
        # Estadísticas básicas
        resumen.append("\nESTADÍSTICAS BÁSICAS:")
        numeric_cols = df_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            try:
                resumen.append(f"{col}: min={df_data[col].min():.0f}, max={df_data[col].max():.0f}, media={df_data[col].mean():.2f}")
            except:
                pass
        
        return "\n".join(resumen)
        
    except Exception as e:
        print(f"Error al procesar CSV: {e}")
        return None

def cortar_en_trozos(texto, chunks_orig_list, chunks_norm_list, tamaño_max=1500):
    """Divide el texto en trozos y los almacena en las listas proporcionadas."""
    chunks_orig_list.clear()
    chunks_norm_list.clear()
    
    # Dividir por líneas
    lines = texto.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line)
        if current_length + line_length <= tamaño_max:
            current_chunk.append(line)
            current_length += line_length
        else:
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks_orig_list.append(chunk_text)
                chunks_norm_list.append(normalize_text(chunk_text))
            current_chunk = [line]
            current_length = line_length
    
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks_orig_list.append(chunk_text)
        chunks_norm_list.append(normalize_text(chunk_text))
    
    print(f"Total chunks creados: {len(chunks_orig_list)}")

def buscar_contexto(pregunta, chunks_orig, chunks_norm, top_k=3):
    """Busca los fragmentos de texto más relevantes en las listas proporcionadas."""
    if not chunks_norm:
        return ""
    
    pregunta_normalizada = normalize_text(pregunta)
    
    # Usar búsqueda simple por palabras clave para CSV
    palabras_clave = extraer_palabras_clave(pregunta_normalizada)
    if not palabras_clave:
        palabras_clave = pregunta_normalizada.split()[:3]
    
    chunks_relevantes = []
    for i, chunk in enumerate(chunks_norm):
        coincidencias = 0
        for palabra in palabras_clave:
            if palabra in chunk:
                coincidencias += 1
        if coincidencias > 0:
            chunks_relevantes.append((i, coincidencias))
    
    if chunks_relevantes:
        chunks_relevantes.sort(key=lambda x: x[1], reverse=True)
        mejores_indices = [idx for idx, _ in chunks_relevantes[:top_k]]
        mejores_chunks = [chunks_orig[i] for i in mejores_indices]
        contexto = "\n\n".join(mejores_chunks)
    else:
        # Si no encuentra, devolver información general
        contexto = chunks_orig[0] if chunks_orig else ""
    
    return contexto

# --- FUNCIONES DE RESPUESTA ---

def responder_general(pregunta):
    """Responde preguntas sin contexto."""
    try:
        response = client.chat_completions_create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": pregunta}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error en responder_general: {e}")
        return f"Error al obtener respuesta"

def responder_documento(pregunta, chunks_orig, chunks_norm):
    """Responde preguntas usando el contexto de las listas proporcionadas."""
    contexto = buscar_contexto(pregunta, chunks_orig, chunks_norm)
    
    if not contexto:
        return "No encuentro información relacionada."
    
    # Limitar contexto
    if len(contexto) > 2000:
        contexto = contexto[:2000] + "..."
    
    mensaje_sistema = f"""Eres un asistente de análisis de datos. Responde usando esta información:

{contexto}

Si no sabes la respuesta, di que no tienes esa información."""
    
    try:
        response = client.chat_completions_create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": mensaje_sistema},
                {"role": "user", "content": pregunta}
            ],
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error en IA: {e}")
        return f"Error al procesar la respuesta"

# --- FUNCIONES PARA PROCESAR PREGUNTAS DE DATOS ---

def procesar_pregunta_simple(pregunta):
    """Procesa preguntas comunes sobre el CSV y devuelve texto y datos para gráficos."""
    if df_data is None:
        return None
    
    pregunta_lower = pregunta.lower()
    columnas = [col.strip() for col in df_data.columns]
    
    # Verificar si CLASE existe
    columna_clase = None
    posibles_nombres = ['CLASE', 'CLASE ', 'CLASE;', ' CLASE', ';CLASE']
    for col in columnas:
        col_limpia = col.strip()
        if col_limpia == 'CLASE' or any(p in col for p in posibles_nombres):
            columna_clase = col
            break
    
    # "cantidad por vehículo" o "cantidad por clase"
    if any(p in pregunta_lower for p in ['cantidad por vehículo', 'cantidad por clase', 'por vehículo', 'por clase']):
        if columna_clase and 'CANTIDAD' in ''.join(columnas):
            try:
                columna_cantidad = next((c for c in columnas if 'CANTIDAD' in c), None)
                
                if columna_cantidad:
                    resultado = df_data.groupby(columna_clase)[columna_cantidad].sum().reset_index()
                    resultado = resultado.sort_values(columna_cantidad, ascending=False)
                    
                    # Crear tabla Markdown
                    respuesta = f"### 📊 Cantidad por Clase de Vehículo\n\n"
                    respuesta += "| Clase | Total |\n|---|---|\n"
                    for _, row in resultado.iterrows():
                        respuesta += f"| {row[columna_clase]} | {row[columna_cantidad]:,.0f} |\n"
                    
                    total = resultado[columna_cantidad].sum()
                    respuesta += f"\n**📈 Total general:** {total:,.0f}"
                    
                    # Datos para el gráfico
                    chart_data = {
                        'type': 'bar',
                        'labels': resultado[columna_clase].head(10).tolist(), # Top 10 para no saturar
                        'data': resultado[columna_cantidad].head(10).tolist(),
                        'label': 'Cantidad por Vehículo'
                    }
                    
                    return {'text': respuesta, 'chart_data': chart_data}
            except Exception as e:
                print(f"Error en cálculo por clase: {e}")
    
    # "cantidad por año"
    elif any(p in pregunta_lower for p in ['cantidad por año', 'por año', 'anual']):
        columna_anio = next((c for c in columnas if 'ANIO' in c or 'AÑO' in c.upper()), None)
        columna_cantidad = next((c for c in columnas if 'CANTIDAD' in c), None)
        
        if columna_anio and columna_cantidad:
            try:
                resultado = df_data.groupby(columna_anio)[columna_cantidad].sum().reset_index()
                resultado = resultado.sort_values(columna_anio)
                
                respuesta = f"### 📊 Cantidad por Año\n\n"
                respuesta += "| Año | Total |\n|---|---|\n"
                for _, row in resultado.iterrows():
                    respuesta += f"| {row[columna_anio]} | {row[columna_cantidad]:,.0f} |\n"
                
                chart_data = {
                    'type': 'line',
                    'labels': resultado[columna_anio].tolist(),
                    'data': resultado[columna_cantidad].tolist(),
                    'label': 'Tendencia Anual'
                }
                
                return {'text': respuesta, 'chart_data': chart_data}
            except Exception as e:
                print(f"Error en cálculo por año: {e}")
    
    # "por departamento"
    elif any(p in pregunta_lower for p in ['por departamento', 'departamento']):
        columna_depto = next((c for c in columnas if 'DEPARTAMENTO' in c), None)
        columna_cantidad = next((c for c in columnas if 'CANTIDAD' in c), None)
        
        if columna_depto and columna_cantidad:
            try:
                resultado = df_data.groupby(columna_depto)[columna_cantidad].sum().reset_index()
                resultado = resultado.sort_values(columna_cantidad, ascending=False)
                
                respuesta = f"### 🏢 Cantidad por Departamento\n\n"
                respuesta += "| Departamento | Total |\n|---|---|\n"
                for _, row in resultado.head(10).iterrows():
                    respuesta += f"| {row[columna_depto]} | {row[columna_cantidad]:,.0f} |\n"
                
                if len(resultado) > 10:
                    respuesta += f"\n... y {len(resultado) - 10} departamentos más"
                
                chart_data = {
                    'type': 'pie',
                    'labels': resultado[columna_depto].head(8).tolist(),
                    'data': resultado[columna_cantidad].head(8).tolist(),
                    'label': 'Distribución por Departamento'
                }
                
                return {'text': respuesta, 'chart_data': chart_data}
            except Exception as e:
                print(f"Error en cálculo por departamento: {e}")
    
    # "total" o "suma"
    elif any(p in pregunta_lower for p in ['total', 'suma', 'sumar', 'cuanto es']):
        columna_cantidad = next((c for c in columnas if 'CANTIDAD' in c), None)
        
        if columna_cantidad:
            try:
                total = df_data[columna_cantidad].sum()
                promedio = df_data[columna_cantidad].mean()
                maximo = df_data[columna_cantidad].max()
                minimo = df_data[columna_cantidad].min()
                
                respuesta = "### 📈 Estadísticas de Cantidad\n\n"
                respuesta += f"- **Total:** {total:,.0f}\n"
                respuesta += f"- **Promedio:** {promedio:,.2f}\n"
                respuesta += f"- **Máximo:** {maximo:,.0f}\n"
                respuesta += f"- **Mínimo:** {minimo:,.0f}\n"
                
                return {'text': respuesta, 'chart_data': None}
            except Exception as e:
                print(f"Error en cálculo de total: {e}")
    
    # "columnas"
    elif any(p in pregunta_lower for p in ['columnas', 'qué columnas', 'campos', 'datos']):
        respuesta = "### 📋 Columnas disponibles\n\n"
        for i, col in enumerate(columnas, 1):
            respuesta += f"{i}. {col}\n"
        return {'text': respuesta, 'chart_data': None}
    
    # "predicción" o "tendencia futura"
    elif any(p in pregunta_lower for p in ['predicción', 'prediccion', 'predice', 'futuro', 'tendencia', 'proyección', 'proyeccion', '2023', '2024', '2025']):
        columna_anio = next((c for c in columnas if 'ANIO' in c or 'AÑO' in c.upper()), None)
        columna_cantidad = next((c for c in columnas if 'CANTIDAD' in c), None)
        
        if columna_anio and columna_cantidad:
            try:
                # Agrupar por año
                datos_anio = df_data.groupby(columna_anio)[columna_cantidad].sum().reset_index()
                datos_anio = datos_anio.sort_values(columna_anio)
                
                # Preparar datos para regresión
                X = datos_anio[columna_anio].values.reshape(-1, 1)
                y = datos_anio[columna_cantidad].values
                
                # Entrenar modelo polinomial (grado 3)
                poly_features = PolynomialFeatures(degree=3)
                X_poly = poly_features.fit_transform(X)
                
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Calcular R²
                y_pred = model.predict(X_poly)
                r2 = r2_score(y, y_pred)
                
                # Predecir próximos 3 años
                ultimo_anio = int(X.max())
                future_years = np.array([[ultimo_anio + 1], [ultimo_anio + 2], [ultimo_anio + 3]])
                future_years_poly = poly_features.transform(future_years)
                predictions = model.predict(future_years_poly)
                
                # Crear respuesta
                respuesta = f"### 🔮 Predicción de Tendencia Futura\n\n"
                respuesta += f"**Modelo de Regresión Polinomial (Grado 3)**\n\n"
                respuesta += f"- **Precisión del modelo (R²):** {r2:.3f} ({r2*100:.1f}% de explicación)\n\n"
                respuesta += f"**Predicciones para los próximos años:**\n\n"
                respuesta += "| Año | Predicción |\n|---|---|\n"
                
                for year, pred in zip(future_years.flatten(), predictions):
                    respuesta += f"| {year} | {int(pred):,} vehículos |\n"
                
                # Calcular crecimiento
                crecimiento = ((predictions[-1] / y[-1]) - 1) * 100
                respuesta += f"\n**📈 Crecimiento proyectado ({ultimo_anio}-{ultimo_anio+3}):** {crecimiento:+.1f}%\n\n"
                respuesta += f"*Nota: Esta predicción se basa en la tendencia histórica de {int(X.min())}-{ultimo_anio}*"
                
                # Datos para gráfico (histórico + predicción)
                all_years = np.concatenate([X.flatten(), future_years.flatten()])
                all_values = np.concatenate([y, predictions])
                
                chart_data = {
                    'type': 'line',
                    'labels': all_years.tolist(),
                    'data': all_values.tolist(),
                    'label': 'Tendencia y Predicción'
                }
                
                return {'text': respuesta, 'chart_data': chart_data}
                
            except Exception as e:
                print(f"Error en predicción: {e}")
                return {'text': f"⚠️ No pude generar la predicción. Error: {str(e)}", 'chart_data': None}
    
    return None

# --- RUTAS DE FLASK ---

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'GET':
        # Si ya está logueado, redirigir al chatbot
        if 'logged_in' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
    
    # POST - Procesar login
    data = request.json
    user_type = data.get('userType')
    username = data.get('username')
    password = data.get('password')
    
    # Validar credenciales
    if user_type in USUARIOS and username in USUARIOS[user_type]:
        if USUARIOS[user_type][username] == password:
            # Login exitoso
            session['logged_in'] = True
            session['username'] = username
            session['user_type'] = user_type
            return jsonify({'success': True, 'message': 'Inicio de sesión exitoso'})
    
    # Credenciales incorrectas
    return jsonify({'success': False, 'error': 'Usuario o contraseña incorrectos'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/')
@login_required
def index():
    # Obtener información del usuario para mostrar en la interfaz
    username = session.get('username', 'Usuario')
    user_type = session.get('user_type', 'usuario')
    return render_template('index.html', 
                         username=username, 
                         user_type=user_type,
                         archivo_info=archivo_soporte)  # Siempre mostrar info del documento de soporte


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Carga archivos para ANÁLISIS (admins) - NO reemplaza el documento de soporte."""
    global vectorizer_analisis, X_analisis, chunks_originales_analisis, chunks_normalizados_analisis, df_data, archivo_analisis
    
    # Verificar que solo administradores puedan cargar archivos
    if session.get('user_type') != 'administrador':
        return jsonify({'error': '🔒 Acceso denegado. Solo administradores pueden cargar archivos.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Verificar tamaño
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > 100 * 1024 * 1024:
        return jsonify({'error': 'Archivo demasiado grande. Límite: 100MB'})
    
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    texto_empresa = None
    tipo_archivo = None
    
    try:
        if file.filename.endswith('.csv'):
            print(f"Procesando archivo CSV...")
            texto_empresa = convertir_csv_a_texto(filepath)
            tipo_archivo = 'CSV'
        elif file.filename.endswith('.txt'):
            with open(filepath, "r", encoding="utf-8") as f:
                texto_empresa = f.read()
            df_data = None
            tipo_archivo = 'TXT'
        elif file.filename.endswith('.pdf'):
            print(f"Procesando archivo PDF...")
            try:
                import PyPDF2
                with open(filepath, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    texto_empresa = ""
                    for page in pdf_reader.pages:
                        texto_empresa += page.extract_text() + "\n"
                df_data = None
                tipo_archivo = 'PDF'
            except ImportError:
                os.remove(filepath)
                return jsonify({'error': 'PyPDF2 no está instalado. Instala con: pip install PyPDF2'})
        else:
            os.remove(filepath)
            return jsonify({'error': 'Formato no soportado. Use .txt, .csv o .pdf'})

        if not texto_empresa:
            os.remove(filepath)
            return jsonify({'error': 'Error al leer el archivo'})
            
        print(f"Texto procesado: {len(texto_empresa):,} caracteres")
        
        # Cortar en trozos (para ANÁLISIS)
        cortar_en_trozos(texto_empresa, chunks_originales_analisis, chunks_normalizados_analisis)

        if not chunks_normalizados_analisis:
            os.remove(filepath)
            return jsonify({'error': 'Archivo vacío'})
            
        # Crear vectorizador para análisis
        vectorizer_analisis = TfidfVectorizer(max_features=1000)
        X_analisis = vectorizer_analisis.fit_transform(chunks_normalizados_analisis)
        
        # Generar descripción automática usando IA
        try:
            preview = texto_empresa[:1500] if len(texto_empresa) > 1500 else texto_empresa
            prompt = f"Resume en una línea de máximo 100 caracteres de qué trata este documento:\n\n{preview}"
            descripcion_response = client.chat_completions_create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            descripcion = descripcion_response["choices"][0]["message"]["content"].strip()
        except:
            descripcion = f"Documento {tipo_archivo} cargado"
        
        # Actualizar información del archivo de ANÁLISIS
        from datetime import datetime
        archivo_analisis = {
            'nombre': file.filename,
            'tipo': tipo_archivo,
            'descripcion': descripcion,
            'fecha_carga': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'cargado_por': session.get('username', 'Admin')
        }
        
        # Estadísticas
        stats = {
            'chunks': len(chunks_originales_analisis),
            'data_rows': len(df_data) if df_data is not None else 0,
            'data_columns': len(df_data.columns) if df_data is not None else 0,
            'archivo_info': archivo_analisis
        }
        
        if df_data is not None:
            stats['column_names'] = list(df_data.columns)
        
        print(f"Procesamiento completado (ANÁLISIS)")
        print(f"Archivo: {archivo_analisis['nombre']}")
        print(f"Descripción: {archivo_analisis['descripcion']}")
        
        return jsonify({
            'success': 'Archivo cargado correctamente',
            'stats': stats
        })
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    pregunta = data.get('pregunta')
    modo = data.get('modo')  
    
    print(f"\nPregunta: '{pregunta}'")
    print(f"Modo: {modo}")
    print(f"Usuario: {session.get('username')} ({session.get('user_type')})")
    
    # USUARIOS REGULARES: Pueden usar chat general Y análisis de documento de SOPORTE
    if session.get('user_type') == 'usuario':
        if modo == 'general':
            # Chat general
            respuesta = responder_general(pregunta)
            return jsonify({'respuesta': respuesta})
        
        elif modo == 'documento':
            # Análisis del documento de SOPORTE (precargado)
            if not chunks_originales_soporte:
                return jsonify({
                    'respuesta': '⚠️ **Sin Documento**\n\nNo hay ningún documento de soporte cargado.'
                })
            
            # Usar RAG para responder basado en el documento de SOPORTE
            respuesta = responder_documento(pregunta, chunks_originales_soporte, chunks_normalizados_soporte)
            return jsonify({'respuesta': respuesta})
    
    # ADMINISTRADORES: Chat general + análisis de su documento cargado
    if session.get('user_type') == 'administrador':
        if modo == 'general':
            respuesta = responder_general(pregunta)
            return jsonify({'respuesta': respuesta})
        
        elif modo == 'documento':
            # Admins pueden analizar su documento cargado (ANÁLISIS)
            if not chunks_originales_analisis:
                return jsonify({
                    'respuesta': '⚠️ **Sin Documento de Análisis**\n\nNo has cargado ningún archivo para análisis. Carga un archivo primero.'
                })
            
            # Procesar pregunta con el documento de ANÁLISIS
            if df_data is not None:
                resultado_simple = procesar_pregunta_simple(pregunta)
                if resultado_simple:
                    if isinstance(resultado_simple, dict):
                        return jsonify({
                            'respuesta': resultado_simple['text'],
                            'chart_data': resultado_simple['chart_data']
                        })
                    return jsonify({'respuesta': resultado_simple})
            
            # Usar RAG para responder basado en el documento de ANÁLISIS
            respuesta = responder_documento(pregunta, chunks_originales_analisis, chunks_normalizados_analisis)
            return jsonify({'respuesta': respuesta})
    
    # Fallback
    if modo == 'general':
        respuesta = responder_general(pregunta)
        return jsonify({'respuesta': respuesta})
    
    elif modo == 'documento':
        if not chunks_originales: 
            respuesta = "Primero sube un archivo."
            return jsonify({'respuesta': respuesta})
        
        # Primero intentar procesamiento simple
        if df_data is not None:
            resultado_simple = procesar_pregunta_simple(pregunta)
            if resultado_simple:
                print("✓ Respuesta generada por procesamiento simple")
                # Si es un diccionario (texto + chart), lo devolvemos tal cual
                if isinstance(resultado_simple, dict):
                    return jsonify({
                        'respuesta': resultado_simple['text'],
                        'chart_data': resultado_simple['chart_data']
                    })
                # Si es solo texto (compatibilidad antigua)
                return jsonify({'respuesta': resultado_simple})
        
        # Si no hay respuesta simple, usar RAG
        print("⏳ Usando RAG...")
        respuesta = responder_documento(pregunta)
        return jsonify({'respuesta': respuesta})

@app.route('/archivo-info', methods=['GET'])
@login_required
def get_archivo_info():
    """Obtener información del documento de soporte (disponible para todos los usuarios)."""
    return jsonify(archivo_soporte)

@app.route('/stats', methods=['GET'])
@login_required
def get_stats():
    """Obtener estadísticas."""
    # Solo administradores pueden ver estadísticas
    if session.get('user_type') != 'administrador':
        return jsonify({'error': 'Acceso denegado'})
    
    stats = {
        'chunks_loaded': len(chunks_originales),
        'data_loaded': df_data is not None
    }
    
    if df_data is not None:
        stats['data_rows'] = len(df_data)
        stats['data_columns'] = len(df_data.columns)
        stats['column_names'] = list(df_data.columns)
    
    return jsonify(stats)

@app.route('/clear', methods=['POST'])
@login_required
def clear_data():
    """Limpiar datos."""
    # Solo administradores pueden limpiar datos
    if session.get('user_type') != 'administrador':
        return jsonify({'error': 'Acceso denegado'})
    
    global vectorizer, X, chunks_originales, chunks_normalizados, df_data
    
    vectorizer = None
    X = None
    chunks_originales = []
    chunks_normalizados = []
    df_data = None
    
    import gc
    gc.collect()
    
    return jsonify({'success': 'Datos limpiados'})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    
    print("=" * 60)
    print("SISTEMA DE SOPORTE - LOS ARTIFICIALES")
    print("=" * 60)
    
    # Cargar documento por defecto para usuarios
    print("\n🔄 Cargando documento por defecto...")
    if cargar_documento_default():
        print("✅ Sistema listo con documento precargado")
    else:
        print("⚠️  Sistema iniciado sin documento por defecto")
    
    print("\n" + "=" * 60)
    print("Servidor iniciado en http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
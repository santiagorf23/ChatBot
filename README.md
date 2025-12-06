# 🤖 Proyecto Los Artificiales - Chatbot Empresarial con IA

Sistema inteligente de chatbot empresarial con análisis de datos, utilizando IA (LLaMA 3.1), RAG (Retrieval-Augmented Generation) y Machine Learning para predicciones.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Ejecución del Chatbot](#-ejecución-del-chatbot)
- [Ejecución del Notebook de Análisis](#-ejecución-del-notebook-de-análisis)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Credenciales de Acceso](#-credenciales-de-acceso)
- [Uso del Sistema](#-uso-del-sistema)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 🎯 Descripción del Proyecto

Este proyecto combina dos componentes principales:

1. **Chatbot Empresarial** (`chatbot/app.py`): Sistema web de chatbot con IA que responde preguntas basándose en documentos de soporte empresarial usando RAG.

2. **Análisis de Datos** (`data_graficos.ipynb`): Notebook de Jupyter con análisis estadístico y visualización de datos de vehículos eléctricos en Colombia.

---

## ✨ Características Principales

### Chatbot Empresarial

- ✅ **Autenticación con roles** (Usuario y Administrador)
- ✅ **RAG (Retrieval-Augmented Generation)** para respuestas precisas
- ✅ **Procesamiento de documentos** (TXT, CSV, PDF)
- ✅ **Análisis de datos con IA** y generación de gráficos
- ✅ **Predicciones con Machine Learning** (regresión polinomial)
- ✅ **Interfaz web moderna** con Flask

### Análisis de Datos

- ✅ **Análisis estadístico completo** (media, mediana, varianza, desviación estándar)
- ✅ **Visualizaciones con Matplotlib y Seaborn**
- ✅ **Clustering con K-Means**
- ✅ **Regresión lineal** para predicciones
- ✅ **PCA** para reducción de dimensionalidad

---

## 🔧 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.8 o superior** ([Descargar aquí](https://www.python.org/downloads/))
- **pip** (gestor de paquetes de Python, viene con Python)
- **Jupyter Notebook** o **JupyterLab** (para el análisis de datos)

### Verificar instalación de Python

```bash
python --version
```

Debería mostrar algo como: `Python 3.10.x` o superior.

---

## 📦 Instalación

### 1. Clonar o descargar el proyecto

Si tienes el proyecto en tu escritorio, navega a la carpeta:

```bash
cd "C:\Users\elian\OneDrive\Escritorio\Los Artificiales"
```

### 2. Crear un entorno virtual (Recomendado)

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate
```

**Windows (CMD):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar dependencias adicionales para el notebook

```bash
pip install jupyter matplotlib seaborn numpy pandas scikit-learn
```

---

## 🚀 Ejecución del Chatbot

### Opción 1: Ejecución Rápida

1. **Navegar a la carpeta del chatbot:**

```bash
cd chatbot
```

2. **Ejecutar el servidor:**

```bash
python app.py
```

3. **Abrir el navegador:**

Abre tu navegador y ve a: **http://localhost:5000**

4. **Iniciar sesión:**

Usa las credenciales de la sección [Credenciales de Acceso](#-credenciales-de-acceso)

### Opción 2: Ejecución con Variables de Entorno

Si necesitas configurar la API key de Groq:

1. Crea un archivo `.env` en la carpeta `chatbot/`:

```env
GROQ_API_KEY=tu_api_key_aqui
```

2. Ejecuta el servidor:

```bash
python app.py
```

### Detener el Servidor

Presiona `Ctrl + C` en la terminal donde está corriendo el servidor.

---

## 📊 Ejecución del Notebook de Análisis

### Opción 1: Usando Jupyter Notebook

1. **Asegúrate de estar en la carpeta raíz del proyecto:**

```bash
cd "C:\Users\elian\OneDrive\Escritorio\Los Artificiales"
```

2. **Iniciar Jupyter Notebook:**

```bash
jupyter notebook
```

3. **Se abrirá tu navegador automáticamente.** Si no, copia la URL que aparece en la terminal (algo como `http://localhost:8888/...`)

4. **En el navegador, haz clic en:**

   - `data_graficos.ipynb`

5. **Ejecutar el notebook:**
   - **Ejecutar todas las celdas:** Menú → `Cell` → `Run All`
   - **Ejecutar celda por celda:** Presiona `Shift + Enter` en cada celda

### Opción 2: Usando VS Code

1. **Abrir el archivo** `data_graficos.ipynb` en VS Code

2. **Instalar la extensión de Jupyter** (si no la tienes):

   - Busca "Jupyter" en el marketplace de VS Code
   - Instala la extensión oficial de Microsoft

3. **Seleccionar el kernel de Python:**

   - Haz clic en "Select Kernel" en la parte superior derecha
   - Selecciona el entorno virtual `.venv` que creaste

4. **Ejecutar celdas:**
   - Haz clic en el botón ▶️ junto a cada celda
   - O usa `Shift + Enter`

### Opción 3: Usando JupyterLab (Interfaz Moderna)

1. **Instalar JupyterLab:**

```bash
pip install jupyterlab
```

2. **Iniciar JupyterLab:**

```bash
jupyter lab
```

3. **Navegar al archivo** `data_graficos.ipynb` y ejecutar las celdas.

---

## 📁 Estructura del Proyecto

```
Los Artificiales/
│
├── chatbot/                          # Aplicación del chatbot
│   ├── app.py                        # Servidor Flask principal
│   ├── templates/                    # Plantillas HTML
│   │   ├── index.html               # Interfaz del chatbot
│   │   └── login.html               # Página de login
│   ├── uploads/                      # Documentos cargados
│   │   └── documento_soporte_techcorp.txt  # Documento de ejemplo
│   └── .gitignore
│
├── docs/                             # Documentación y datos
│   ├── data.csv                      # Dataset de vehículos eléctricos
│   └── presentacion/                 # Presentaciones del proyecto
│
├── data_graficos.ipynb               # Notebook de análisis de datos
├── requirements.txt                  # Dependencias del proyecto
├── .env                              # Variables de entorno (API keys)
└── README.md                         # Este archivo
```

---

## 🔑 Credenciales de Acceso

### Usuarios Regulares

Pueden usar el chat general y consultar el documento de soporte.

| Usuario   | Contraseña   |
| --------- | ------------ |
| `usuario` | `usuario123` |
| `user1`   | `pass123`    |
| `user2`   | `pass456`    |

### Administradores

Tienen acceso completo: chat, carga de archivos, análisis de datos y predicciones.

| Usuario  | Contraseña |
| -------- | ---------- |
| `admin`  | `admin123` |
| `admin2` | `admin456` |
| `root`   | `root123`  |

---

## 💡 Uso del Sistema

### Chatbot - Modo Usuario

1. **Inicia sesión** con credenciales de usuario
2. **Selecciona el modo:**

   - **Chat General:** Preguntas generales a la IA
   - **Documento:** Preguntas sobre el documento de soporte

3. **Ejemplos de preguntas sobre el documento:**
   - "¿Cómo contactar con soporte técnico?"
   - "¿Qué productos ofrece TechCorp?"
   - "¿Cómo recuperar mi contraseña?"
   - "¿Cuáles son los métodos de pago?"

### Chatbot - Modo Administrador

1. **Inicia sesión** con credenciales de administrador
2. **Cargar archivo para análisis:**

   - Haz clic en "Cargar Archivo"
   - Selecciona un archivo CSV, TXT o PDF
   - Espera a que se procese

3. **Hacer preguntas sobre los datos:**

   - "¿Cantidad por vehículo?"
   - "¿Cantidad por año?"
   - "¿Cantidad por departamento?"
   - "Predice la tendencia para 2025"

4. **Ver gráficos generados automáticamente**

### Notebook de Análisis

El notebook `data_graficos.ipynb` contiene:

1. **Carga de datos** del archivo `docs/data.csv`
2. **Limpieza de datos** (eliminación de columnas innecesarias)
3. **Análisis estadístico:**

   - Media, mediana, moda
   - Varianza y desviación estándar
   - Valores mínimos y máximos

4. **Visualizaciones:**

   - Gráficos de barras
   - Gráficos de líneas
   - Distribuciones

5. **Machine Learning:**
   - Clustering con K-Means
   - Regresión lineal
   - PCA (Análisis de Componentes Principales)

---

## 🛠️ Tecnologías Utilizadas

### Backend

- **Python 3.10+**
- **Flask** - Framework web
- **Groq API** - Servicio de IA (LLaMA 3.1)
- **Scikit-learn** - Machine Learning
- **Pandas** - Procesamiento de datos
- **NumPy** - Cálculos numéricos

### Frontend

- **HTML5 / CSS3 / JavaScript**
- **Chart.js** - Gráficos interactivos
- **Marked.js** - Renderizado de Markdown

### Análisis de Datos

- **Jupyter Notebook**
- **Matplotlib** - Visualizaciones
- **Seaborn** - Gráficos estadísticos
- **Scikit-learn** - Clustering y regresión

### Otras Herramientas

- **PyPDF2** - Lectura de PDFs
- **Flask-CORS** - Manejo de CORS
- **psutil** - Monitoreo de recursos

---

## 🐛 Solución de Problemas

### El servidor no inicia

**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solución:**

```bash
pip install -r requirements.txt
```

### El puerto 5000 está en uso

**Error:** `Address already in use`

**Solución:**

**Windows:**

```bash
# Encontrar el proceso
netstat -ano | findstr :5000

# Matar el proceso (reemplaza PID con el número que aparece)
taskkill /PID <PID> /F
```

**Linux/Mac:**

```bash
# Encontrar y matar el proceso
lsof -ti:5000 | xargs kill -9
```

### Jupyter Notebook no se abre

**Solución:**

```bash
# Reinstalar Jupyter
pip uninstall jupyter
pip install jupyter

# Iniciar con navegador específico
jupyter notebook --browser=chrome
```

### Error de API Key de Groq

**Error:** `Invalid API Key`

**Solución:**

1. Verifica que la API key en `app.py` (línea 52) sea válida
2. O crea un archivo `.env` con tu propia API key:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```
3. Obtén una API key gratuita en: https://console.groq.com

### El chatbot no encuentra el documento

**Error:** `No se encontró documento por defecto`

**Solución:**

1. Verifica que exista el archivo: `chatbot/uploads/documento_soporte_techcorp.txt`
2. Si no existe, el sistema creará uno automáticamente al iniciar

---

## 📝 Notas Adicionales

### Documento de Soporte Predeterminado

El chatbot carga automáticamente un documento de soporte de ejemplo (`documento_soporte_techcorp.txt`) al iniciar. Este documento contiene:

- Información de la empresa TechCorp Solutions
- Preguntas frecuentes (FAQs)
- Políticas y procedimientos
- Información de contacto

### Carga de Archivos CSV

Para analizar tus propios datos:

1. Inicia sesión como **administrador**
2. Haz clic en **"Cargar Archivo"**
3. Selecciona un archivo CSV con las siguientes columnas (ejemplo):
   - `ANIO_REGISTRO`
   - `CLASE`
   - `DEPARTAMENTO`
   - `CANTIDAD`

### Límites del Sistema

- **Tamaño máximo de archivo:** 100MB
- **Formatos soportados:** TXT, CSV, PDF
- **Tiempo de sesión:** 2 horas de inactividad

---

## 👥 Autores

**Proyecto Los Artificiales**

- Análisis de Vehículos Eléctricos en Colombia
- Chatbot Empresarial con IA

---

## 📄 Licencia

Este proyecto es de uso educativo y demostrativo.

---

## 🚀 Inicio Rápido (Resumen)

### Para ejecutar el Chatbot:

```bash
# 1. Navegar a la carpeta
cd "C:\Users\elian\OneDrive\Escritorio\Los Artificiales\chatbot"

# 2. Ejecutar el servidor
python app.py

# 3. Abrir navegador en: http://localhost:5000
# 4. Login: usuario / usuario123
```

### Para ejecutar el Notebook:

```bash
# 1. Navegar a la carpeta raíz
cd "C:\Users\elian\OneDrive\Escritorio\Los Artificiales"

# 2. Iniciar Jupyter
jupyter notebook

# 3. Abrir data_graficos.ipynb
# 4. Ejecutar: Cell → Run All
```

---

**¡Listo para usar! 🎉**

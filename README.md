# 🤖 Proyecto Los Artificiales - Chatbot Empresarial con IA

Sistema inteligente de chatbot empresarial con análisis de datos, utilizando IA (LLaMA 3.1), RAG (Retrieval-Augmented Generation) y Machine Learning para predicciones.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Ejecución del Chatbot](#-ejecución-del-chatbot)
- [Ejecución del Notebook de Análisis](#-ejecución-del-notebook-de-análisis)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Credenciales de Acceso](#-credenciales-de-acceso)
- [Uso del Sistema](#-uso-del-sistema)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 🎯 Descripción del Proyecto

Este proyecto combina dos componentes principales:

1. **Chatbot Empresarial** (`chatbot/app.py`): Sistema web de chatbot con IA que responde preguntas basándose en documentos de soporte empresarial usando RAG. Permite cargas de archivos CSV/PDF/TXT para análisis en tiempo real.

2. **Análisis de Datos** (`data_graficos.ipynb`): Notebook de Jupyter con análisis estadístico detallado y visualización de datos de vehículos eléctricos en Colombia.

---

## ✨ Características Principales

### Chatbot Empresarial

- ✅ **Autenticación con roles** (Usuario y Administrador)
- ✅ **RAG (Retrieval-Augmented Generation)** para respuestas precisas basadas en contexto
- ✅ **Procesamiento de documentos** (TXT, CSV, PDF)
- ✅ **Análisis de datos con IA** y generación de gráficos dinámicos
- ✅ **Predicciones con Machine Learning** (regresión polinomial)
- ✅ **Interfaz web moderna** construida con Flask
- ✅ **Manejo seguro de credenciales** mediante variables de entorno

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
- **Git**
- **Clave de API de Groq** (Obtenla en [Groq Console](https://console.groq.com/))

### Verificar instalación de Python

```bash
python --version
# O en Linux/Mac
python3 --version
```

---

## 📦 Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/santiagorf23/ChatBot.git
cd ChatBot
```

### 2. Crear y activar entorno virtual

**Windows:**

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

### 4. Configurar Variables de Entorno

**Importante:** Nunca subas tus claves API al repositorio.

1.  Crea un archivo llamado `.env` en la raíz del proyecto (al nivel de `setup.py` o donde prefieras, pero el código actual busca variables de entorno del sistema o puedes cargarlas con `python-dotenv`).
    - _Nota: La aplicación actual lee `os.getenv("API_KEY_GROQ")`. Asegúrate de exportarla o configurar tu entorno._

**En Linux/Mac:**

```bash
export API_KEY_GROQ="tu_api_key_aqui"
```

**En Windows (PowerShell):**

```powershell
$env:API_KEY_GROQ="tu_api_key_aqui"
```

**(Opcional pero recomendado)**: Puedes crear un archivo `.env` en la raíz y usar el paquete `python-dotenv` para cargarlo automáticamente (si modificas el código para usar `load_dotenv`).

---

## 🚀 Ejecución del Chatbot

1. **Navegar a la carpeta del chatbot:**

```bash
cd chatbot
```

2. **Ejecutar el servidor:**

```bash
python app.py
# O en Linux
python3 app.py
```

3. **Abrir el navegador:**
   Ve a: **http://localhost:5000**

---

## 📊 Ejecución del Notebook de Análisis

El análisis de datos se encuentra en `data_graficos.ipynb`.

1. **Asegúrate de tener las dependencias de Jupyter instaladas:**

   ```bash
   pip install jupyter matplotlib seaborn numpy pandas scikit-learn
   ```

2. **Iniciar Jupyter Notebook:**

   ```bash
   # Desde la raíz del proyecto
   jupyter notebook
   ```

3. **Ejecutar:** Abre `data_graficos.ipynb` en la interfaz que se abre en tu navegador y selecciona "Run All".

---

## 📁 Estructura del Proyecto

```
Los Artificiales/
│
├── chatbot/                          # Aplicación Flask
│   ├── app.py                        # Servidor principal
│   ├── templates/                    # HTML Templates
│   ├── uploads/                      # Archivos temporales (no rastreados por git)
│   └── ...
│
├── docs/                             # Documentación y Datos
│   ├── data_limpia/                  # Datos procesados (no rastreados)
│   └── presentacion/                 # (no rastreados)
│
├── .gitignore                        # Archivos ignorados por Git
├── data_graficos.ipynb               # Análisis de datos (Jupyter)
├── requirements.txt                  # Dependencias
└── README.md                         # Documentación
```

---

## 🔑 Credenciales de Acceso (Demo)

El sistema viene pre-configurado con usuarios para demostración:

### Usuarios Regulares (Solo consultas)

| Usuario   | Contraseña   |
| --------- | ------------ |
| `usuario` | `usuario123` |
| `user1`   | `pass123`    |

### Administradores (Carga de archivos y Análisis)

| Usuario | Contraseña |
| ------- | ---------- |
| `admin` | `admin123` |
| `root`  | `root123`  |

---

## 🐛 Solución de Problemas Comunes

### Error: `ModuleNotFoundError: No module named 'pandas'` o similar

Asegúrate de haber activado el entorno virtual (`source .venv/bin/activate` o `.venv\Scripts\activate`) antes de ejecutar el código.

### Error de API Key

Si ves errores relacionados con la API de Groq, verifica que la variable de entorno `API_KEY_GROQ` esté configurada correctamente en tu terminal antes de lanzar la app.

### El puerto 5000 está ocupado

Mata el proceso que lo usa o cambia el puerto en `app.py`.

- **Linux:** `lsof -ti:5000 | xargs kill -9`
- **Windows:** `taskkill /PID <PID> /F` (encuentra el PID con `netstat -ano | findstr :5000`)

---

## 👥 Autores

**Proyecto Los Artificiales**

- Análisis de Vehículos Eléctricos en Colombia
- Chatbot Empresarial con IA

---

## 📄 Licencia

Este proyecto es de fines educativos y demostrativos.

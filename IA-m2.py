################################################################################################################################ Importación de paquetes
import unicodedata
import string
import heapq
import openai
import networkx as nx
import json
import re
import sys
import wikipediaapi
import numpy as np
import os
import nltk
import time
import matplotlib.pyplot as plt
import signal
import torch
import random
import plotly.graph_objects as go
import getpass
import itertools
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.data import find
from difflib import get_close_matches
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone
from ftplib import FTP
################################################################################################################################ Configuración
# Evita los mensajes de descarga de NLTK
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
try:
    find('corpora/wordnet')
    WORDNET_AVAILABLE = True
except LookupError:
    print("⚠️ WordNet no está disponible. Se desactivan dualidades automáticas por WordNet.")
    WORDNET_AVAILABLE = False
if WORDNET_AVAILABLE :
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

stemmer = SnowballStemmer("spanish")

# Crear un objeto Net persistente
net_global = Network(height="800px", width="100%", directed=True)
#net_global.set_options("""
#var options = {
#  "physics": {
#    "barnesHut": {
#      "gravitationalConstant": -2000,
#      "centralGravity": 0.3,
#      "springLength": 100,
#      "springConstant": 0.04
#    }
#  }
#}
#""")
net_global.set_options("""
var options = {
  "physics": {
    "enabled": true,
    "stabilization": {
      "enabled": true,
      "iterations": 250,
      "updateInterval": 25
    },
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.005,
      "springLength": 150,
      "springConstant": 0.05,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75
  },
  "nodes": {
    "shape": "dot",
    "scaling": {
      "min": 10,
      "max": 40
    },
    "font": {
      "size": 14,
      "face": "arial"
    }
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "dynamic"
    }
  }
}
""")

def reiniciar_visualizacion_proceso():
    global net_global
    net_global = Network(height="800px", width="100%", directed=True)

NIVEL_MAX_EMERGENCIA = 4  # Subirlo xa más profundidad en el futuro

# Crear carpeta "subgrafos" si no existe
carpeta_subgrafos = "subgrafos"
if not os.path.exists(carpeta_subgrafos):
    os.makedirs(carpeta_subgrafos)

# Crear carpeta "json" si no existe
carpeta_json = "json"
if not os.path.exists(carpeta_json):
    os.makedirs(carpeta_json)

# 🔹 Cargar API Key desde variable de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 Manejar interrupción con Ctrl+C para evitar cierres forzados
signal.signal(signal.SIGINT, lambda sig, frame: exec('raise KeyboardInterrupt'))

if OPENAI_API_KEY is None:
    raise ValueError("❌ ERROR: La variable de entorno 'OPENAI_API_KEY' no está configurada.")

# 🔹 Crear cliente de OpenAI con la nueva API
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# 🔹 Cargar modelo de embeddings
modelo = SentenceTransformer('all-MiniLM-L6-v2')
modelo_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# 🔹 Archivo de caché para embeddings
EMBEDDINGS_CACHE_FILE = "json/cache_embeddings.json"

def cargar_cache_embeddingsANTIGUO():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as f:
            datos = json.load(f)
        return {k: torch.tensor(v) for k, v in datos.items()}
    return {}
    
def cargar_cache_embeddings():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as f:
                datos = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Cache de embeddings corrupta o vacía. Se reinicia desde cero.")
            return {}
        return {k: torch.tensor(v) for k, v in datos.items()}
    return {}

def guardar_cache_embeddingsANTIGUO(cache):
    with open(EMBEDDINGS_CACHE_FILE, "w", encoding="utf-8") as f:
        datos = {k: v.tolist() for k, v in cache.items()}
        json.dump(datos, f, ensure_ascii=False, indent=4)
        
def guardar_cache_embeddings(cache):
    tmp_file = EMBEDDINGS_CACHE_FILE + ".tmp"
    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_FILE), exist_ok=True)
    with open(tmp_file, "w", encoding="utf-8") as f:
        datos = {k: v.tolist() for k, v in cache.items()}
        json.dump(datos, f, ensure_ascii=False, indent=4)
    os.replace(tmp_file, EMBEDDINGS_CACHE_FILE)


# Inicializar la caché al iniciar el sistema
embeddings_cache = cargar_cache_embeddings()

# 🔹 Base de dualidades predefinidas
dualidades_base = {
    "abajo": "arriba", "izquierda": "derecha", "detrás": "delante",
    "pasado": "futuro", "negativo":  "positivo",
    "orden": "caos", "espacio": "tiempo", "materia": "energía",
    "frío": "calor"
}
dualidades_base_protegidas = {
    ("izquierda", "derecha"), ("derecha", "izquierda"),
    ("abajo", "arriba"), ("arriba", "abajo"),
    ("detrás", "delante"), ("delante", "detrás"),
    ("pasado", "futuro"), ("futuro", "pasado"),
    ("negativo", "positivo"), ("positivo", "negativo"),
    ("materia", "energía"), ("energía", "materia"),
    ("frío", "calor"), ("calor", "frío"),
}

# 🔹 Triadas base por defecto (extremo1/extremo2-equilibrio-[sintesis])
TRIADAS_BASE_DEFECTO = [
    # Tesis / antítesis – síntesis (sin nodo superior explícito)
    {
        "a": "tesis",
        "b": "antítesis",
        "equilibrio": "síntesis",
        "sintesis": None,  # aquí no añadimos nodo superior extra
    },
    # Masa / aceleración – fuerza (la síntesis superior podría ser 'dinámica')
    {
        "a": "masa",
        "b": "aceleración",
        "equilibrio": "fuerza",
        "sintesis": None,
    },
    {
        "a": "universo",
        "b": "multiverso",
        "equilibrio": "omniverso",
        "sintesis": None,
    },
    {
        "a": "posición",
        "b": "tiempo",
        "equilibrio": "velocidad",
        "sintesis": None,
    },
    {
        "a": "velocidad",
        "b": "tiempo",
        "equilibrio": "aceleración",
        "sintesis": None,
    },
    {
        "a": "trabajo",
        "b": "tiempo",
        "equilibrio": "potencia",
        "sintesis": None,
    },
    {
        "a": "trabajo",
        "b": "desplazamiento",
        "equilibrio": "energía",
        "sintesis": None,
    },
    {
        "a": "centro focal",
        "b": "presente",
        "equilibrio": "consciencia",
        "sintesis": None,
    },
    {
        "a": "espacio",
        "b": "tiempo",
        "equilibrio": "relatividad",
        "sintesis": None,
    },
    {
        "a": "energía cinética",
        "b": "energía potencial",
        "equilibrio": "energía mecánica",
        "sintesis": None,
    },
    {
        "a": "entropía",
        "b": "energía",
        "equilibrio": "información",
        "sintesis": None,
    },
    {
        "a": "presión",
        "b": "volumen",
        "equilibrio": "temperatura",
        "sintesis": None,
    },
    {
        "a": "calor",
        "b": "frío",
        "equilibrio": "tibio",
        "sintesis": "temperatura",    
    },
    {
        "a": "calor",
        "b": "trabajo",
        "equilibrio": "energía interna",
        "sintesis": None,
    },
    {
        "a": "adenina",
        "b": "timina",
        "equilibrio": "puente de hidrógeno",
        "sintesis": "base nitrogenada"
    },
    {
        "a": "guanina",
        "b": "citosina",
        "equilibrio": "puente de hidrógeno",
        "sintesis": "base nitrogenada"
    },
    {
        "a": "base nitrogenada",
        "b": "fosfato",
        "equilibrio": "nucleósido",
        "sintesis": "nucleótido"
    },
    {
        "a": "campo eléctrico",
        "b": "campo magnético",
        "equilibrio": "campo electromagnético",
        "sintesis": None,
    },
    {
        "a": "punto",
        "b": "línea",
        "equilibrio": "plano",
        "sintesis": None,
    },
    {
        "a": "vértice",
        "b": "arista",
        "equilibrio": "cara",
        "sintesis": None,
    },
    {
        "a": "coodenada X",
        "b": "coordenada Y",
        "equilibrio": "punto 2D",
        "sintesis": None,
    },
    {
        "a": "base",
        "b": "altura",
        "equilibrio": "área",
        "sintesis": None,
    },
    {
        "a": "suma",
        "b": "resta",
        "equilibrio": "promedio",
        "sintesis": "raíz cuadrada",
    },
    {
        "a": "multiplicación",
        "b": "división",
        "equilibrio": "promedio",
        "sintesis": "raíz cuadrada",
    },
    {
        "a": "término",
        "b": "coheficiente",
        "equilibrio": "expresión",
        "sintesis": None,
    },
    {
        "a": "función",
        "b": "derivada",
        "equilibrio": "variación",
        "sintesis": None,
    },
    {
        "a": "vértice",
        "b": "arista",
        "equilibrio": "cara",
        "sintesis": None,
    },
{
        "a": "onda",
        "b": "partícula",
        "equilibrio": "dualidad",
        "sintesis": None,
    },
    {
        "a": "orden",
        "b": "caos",
        "equilibrio": "equilibrio",
        "sintesis": None,
    },
    {
        "a": "energía",
        "b": "materia",
        "equilibrio": "realidad",
        "sintesis": None,
    },
    {
        "a": "observador",
        "b": "sistema",
        "equilibrio": "medición",
        "sintesis": None,
    },
    {
        "a": "posibilidad",
        "b": "probabilidad",
        "equilibrio": "ocurrecia",
        "sintesis": None,
    },
    {
        "a": "negativo",
        "b": "positivo",
        "equilibrio": "neutro",
        "sintesis": None,
    },
    {
        "a": "cuerpo",
        "b": "alma",
        "equilibrio": "espíritu",
        "sintesis": None,
    },
    {
        "a": "nacimiento",
        "b": "muerte",
        "equilibrio": "vida",
        "sintesis": None,
    },

    # Puedes añadir más:
    # {
    #     "a": "contracción",
    #     "b": "expansión",
    #     "equilibrio": "equilibrio",
    #     "sintesis": "homeostasis",
    # },
]


# 🔹 Memoria de dualidades confirmadas / rechazadas
ARCHIVO_DUALIDADES_CONFIRMADAS = "json/dualidades_confirmadas.json"
ARCHIVO_DUALIDADES_RECHAZADAS = "json/dualidades_rechazadas.json"

expansion_activa = True  # Variable global para controlar la expansión

# 🔹 Base de conceptos iniciales para evitar que la red inicie vacía
SEMILLA_INICIAL = ["espacio", "tiempo", "realidad"]

# Archivo de configuración
CONFIG_FILE = "json/config.json"

# Cargar configuración desde archivo (nodo central, etc.)
def cargar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# 🔹 Nodo central
# Inicializar configuración
config = cargar_config()
NODO_CENTRAL = config.get("nodo_central")

# Cargar el grafo desde el archivo JSON
grafo_json = "json/grafo.json"

def cargar_grafo():
    G = nx.DiGraph()
    try:
        with open(grafo_json, "r") as f:
            datos = json.load(f)
            for nodo in datos["nodos"]:
                G.add_node(nodo)
            for u, v in datos["edges"]:
                G.add_edge(u, v)
            print(f"Grafo cargado con {len(G.nodes())} nodos y {len(G.edges())} conexiones.")
    except FileNotFoundError:
        print("⚠️ No se encontró ningún grafo para adjuntar.")
    return G

# 🔹 Cargar diccionario desde JSON o inicializar con la semilla
def cargar_diccionario():
    try:
        with open("json/diccionario.json", "r", encoding="utf-8") as f:
            diccionario = json.load(f)
            if diccionario:  # Si el diccionario ya tiene contenido
                return diccionario
            else:
                print("⚠️ Diccionario vacío.")
    except FileNotFoundError:
        print("⚠️ Diccionario no encontrado.")

    # Preguntar al usuario si quiere usar la semilla por defecto o personalizada
    usar_semilla = input("¿Deseas usar la semilla por defecto? (s/n): ").lower().strip()
    if usar_semilla == "s":
        semilla = SEMILLA_INICIAL
    else:
        entrada = input("Introduce los conceptos iniciales separados por coma: ")
        semilla = [c.strip() for c in entrada.split(",") if c.strip()]

    diccionario = {concepto: [] for concepto in semilla}
    guardar_diccionario(diccionario)
    return diccionario
    
def generar_sitemap(
    dominio_base="https://ia-m.ai",
    carpeta_subgrafos="subgrafos",
    rutas_extra=None,
    salida="sitemap.xml"
):
    """
    Genera un sitemap.xml sencillo con:
      - index.html (si existe)
      - IA_m_proceso.html (si existe)
      - hipercubo_fractal_fluido.html (si existe)
      - todos los *.html dentro de 'subgrafos'
    """
    if rutas_extra is None:
        rutas_extra = []

    urls = []

    # 1) HTML de la raíz
    raiz_htmls = ["index.html", "IA_m_proceso.html", "hipercubo_fractal_fluido.html"]
    for nombre in raiz_htmls + rutas_extra:
        if os.path.exists(nombre):
            #mtime = os.path.getmtime(nombre)
            #lastmod = datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%d")
            mtime = os.path.getmtime(nombre)
            lastmod = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")

            urls.append({
                "loc": f"{dominio_base}/{nombre}",
                "lastmod": lastmod,
                "priority": "1.0" if nombre == "index.html" else "0.8"
            })

    # 2) Subgrafos
    if os.path.isdir(carpeta_subgrafos):
        for archivo in os.listdir(carpeta_subgrafos):
            if not archivo.endswith(".html"):
                continue
            ruta_local = os.path.join(carpeta_subgrafos, archivo)
            #mtime = os.path.getmtime(ruta_local)
            #lastmod = datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%d")
            mtime = os.path.getmtime(nombre)
            lastmod = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")

            urls.append({
                "loc": f"{dominio_base}/{carpeta_subgrafos}/{archivo}",
                "lastmod": lastmod,
                "priority": "0.5"
            })

    # 3) Construir XML
    lineas = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]
    for u in urls:
        lineas.append("  <url>")
        lineas.append(f"    <loc>{u['loc']}</loc>")
        lineas.append(f"    <lastmod>{u['lastmod']}</lastmod>")
        lineas.append(f"    <priority>{u['priority']}</priority>")
        lineas.append("  </url>")
    lineas.append("</urlset>")

    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print(f"🗺️ Sitemap generado: {salida} ({len(urls)} URLs)")
    

def generar_sitemapANTIGUO(
    dominio_base="https://ia-m.ai",
    carpeta_subgrafos="subgrafos",
    rutas_extra=None,
    salida="sitemap.xml"
):
    """
    Genera un sitemap.xml sencillo con:
      - index.html
      - IA_m_proceso.html (si existe)
      - todos los subgrafos *.html del directorio 'subgrafos'
    """
    if rutas_extra is None:
        rutas_extra = []

    urls = []

    # 1) Páginas de la raíz
    raiz_htmls = ["index.html", "IA_m_proceso.html", "hipercubo_fractal_fluido.html"]
    for nombre in raiz_htmls + rutas_extra:
        if os.path.exists(nombre):
            mtime = os.path.getmtime(nombre)
            lastmod = datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%d")
            urls.append({
                "loc": f"{dominio_base}/{nombre}",
                "lastmod": lastmod,
                "priority": "1.0" if nombre == "index.html" else "0.8"
            })

    # 2) Subgrafos
    if os.path.isdir(carpeta_subgrafos):
        for archivo in os.listdir(carpeta_subgrafos):
            if not archivo.endswith(".html"):
                continue
            ruta_local = os.path.join(carpeta_subgrafos, archivo)
            mtime = os.path.getmtime(ruta_local)
            lastmod = datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%d")
            urls.append({
                "loc": f"{dominio_base}/{carpeta_subgrafos}/{archivo}",
                "lastmod": lastmod,
                "priority": "0.5"
            })

    # 3) Construir XML
    lineas = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]
    for u in urls:
        lineas.append("  <url>")
        lineas.append(f"    <loc>{u['loc']}</loc>")
        lineas.append(f"    <lastmod>{u['lastmod']}</lastmod>")
        lineas.append(f"    <priority>{u['priority']}</priority>")
        lineas.append("  </url>")
    lineas.append("</urlset>")

    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print(f"🗺️ Sitemap generado: {salida} ({len(urls)} URLs)")
    

usarFTP = False #True/False
def subir_htmls_recientes_por_ftp(
    host="",
    usuario="",
    contraseña="",
    carpeta_local="",
    carpeta_remota="/",
    max_horas=2,
    archivo_registro="json/registro_subidas.json"
):
    """
    Sube por FTP:
      - Subgrafos recientes modificados
      - sitemap.xml (SIEMPRE, si ha cambiado)
    """

    if not host:
        host = input("🌐 Servidor FTP (host): ").strip()
    if not usuario:
        usuario = input("👤 Usuario FTP: ").strip()
    if not contraseña:
        contraseña = getpass.getpass("🔒 Contraseña FTP: ")

    # Cargar registro de subidas previas
    if os.path.exists(archivo_registro):
        with open(archivo_registro, "r", encoding="utf-8") as f:
            registro = json.load(f)
    else:
        registro = {}

    archivos_subidos = 0
    archivos_sin_cambios = 0
    total_archivos = 0

    with FTP(host) as ftp:
        ftp.login(usuario, contraseña)
        print(f"📡 Conectado a {host} como {usuario}")

        # Asegurar carpeta remota
        try:
            ftp.cwd(carpeta_remota)
        except:
            print(f"📁 Carpeta remota no encontrada, creando {carpeta_remota}...")
            partes = carpeta_remota.strip("/").split("/")
            ruta_actual = ""
            for parte in partes:
                ruta_actual += f"/{parte}"
                try:
                    ftp.mkd(ruta_actual)
                except:
                    pass
            ftp.cwd(carpeta_remota)

        # ---------------------------------------------
        # 1) Subir subgrafos HTML recientes
        # ---------------------------------------------
        for archivo in os.listdir(carpeta_local):
            if archivo.endswith(".html"):
                total_archivos += 1
                ruta_archivo = os.path.join(carpeta_local, archivo)
                mod_local = os.path.getmtime(ruta_archivo)

                if archivo in registro and registro[archivo] == mod_local:
                    archivos_sin_cambios += 1
                    continue

                with open(ruta_archivo, "rb") as f:
                    ftp.storbinary(f"STOR " + archivo, f)
                    print(f"✅ Subido: {archivo}")
                    archivos_subidos += 1
                    registro[archivo] = mod_local

        # ---------------------------------------------
        # 2) Subir SIEMPRE sitemap.xml (si existe)
        # ---------------------------------------------
        sitemap_local = "sitemap.xml"
        sitemap_remoto = "sitemap.xml"

        if os.path.exists(sitemap_local):
            total_archivos += 1
            mod_local = os.path.getmtime(sitemap_local)

            if sitemap_remoto in registro and registro[sitemap_remoto] == mod_local:
                archivos_sin_cambios += 1
            else:
                with open(sitemap_local, "rb") as f:
                    ftp.storbinary("STOR " + sitemap_remoto, f)
                    print(f"🌐 Subido sitemap.xml")
                    archivos_subidos += 1
                    registro[sitemap_remoto] = mod_local

        else:
            print("⚠️ No existe sitemap.xml. Recuerda generarlo antes de subir.")

    # Guardar registro actualizado
    with open(archivo_registro, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2)

    print(f"\n📊 Revisión completada.")
    print(f"📄 Archivos revisados: {total_archivos}")
    print(f"⏫ Subidos: {archivos_subidos}")
    print(f"⏩ Sin cambios: {archivos_sin_cambios}")


def subir_htmls_recientes_por_ftpANTIGUO(
    host="ia-m.ai",
    usuario="sergigm87@ia-m.ai",
    contraseña="fS8$)QYG.u^eVH6",
    carpeta_local="subgrafos",
    carpeta_remota="/",
    max_horas=2,
    archivo_registro="json/registro_subidas.json"
):
    if not host:
        host = input("🌐 Servidor FTP (host): ").strip()
    if not usuario:
        usuario = input("👤 Usuario FTP: ").strip()
    if not contraseña:
        contraseña = getpass.getpass("🔒 Contraseña FTP: ")

    if os.path.exists(archivo_registro):
        with open(archivo_registro, "r", encoding="utf-8") as f:
            registro = json.load(f)
    else:
        registro = {}

    archivos_subidos = 0
    archivos_sin_cambios = 0
    total_archivos = 0

    with FTP(host) as ftp:
        ftp.login(usuario, contraseña)
        print(f"📡 Conectado a {host} como {usuario}")

        try:
            ftp.cwd(carpeta_remota)
        except:
            print(f"📁 Carpeta remota no encontrada, creando {carpeta_remota}...")
            partes = carpeta_remota.strip("/").split("/")
            ruta_actual = ""
            for parte in partes:
                ruta_actual += f"/{parte}"
                try:
                    ftp.mkd(ruta_actual)
                except:
                    pass
            ftp.cwd(carpeta_remota)

        for archivo in os.listdir(carpeta_local):
            if archivo.endswith(".html"):
                total_archivos += 1
                ruta_archivo = os.path.join(carpeta_local, archivo)
                mod_local = os.path.getmtime(ruta_archivo)

                if archivo in registro and registro[archivo] == mod_local:
                    archivos_sin_cambios += 1
                    continue

                with open(ruta_archivo, "rb") as f:
                    ftp.storbinary(f"STOR " + archivo, f)
                    print(f"✅ Subido: {archivo}")
                    archivos_subidos += 1
                    registro[archivo] = mod_local

    with open(archivo_registro, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2)

    print(f"\n📊 Revisión completada.")
    print(f"📄 Archivos revisados: {total_archivos}")
    print(f"⏫ Subidos: {archivos_subidos}")
    print(f"⏩ Sin cambios: {archivos_sin_cambios}")
    
# 🔹 Guardar la red fractal
def guardar_red(G):
    with open("json/red_fractal.json", "w") as f:
        json.dump(nx.node_link_data(G, edges="links"), f)
        
ARCHIVO_HISTORIAL = "json/historial_expansion.json"
def cargar_historial():
    """ Carga el historial de expansiones previas """
    try:
        with open(ARCHIVO_HISTORIAL, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def guardar_historial(historial):
    """ Guarda el historial de expansiones """
    with open(ARCHIVO_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=4)

# Archivo donde se guardará la lista de espera
ARCHIVO_ESPERA_NODOS = "json/espera_nodos.json"

# Cargar nodos en espera desde un archivo JSON
def cargar_espera_nodos():
    try:
        with open(ARCHIVO_ESPERA_NODOS, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Guardar nodos en espera en un archivo JSON
def guardar_espera_nodos(espera_nodos):
    with open(ARCHIVO_ESPERA_NODOS, "w", encoding="utf-8") as f:
        json.dump(espera_nodos, f, ensure_ascii=False, indent=4)

ARCHIVO_REGISTRO = "json/registro_expansion.json"
def cargar_registro():
    """ Carga el registro de expansión """
    try:
        with open(ARCHIVO_REGISTRO, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ Archivo de registro no encontrado, creando uno nuevo.")
        guardar_registro([])  # Guardar una lista vacía para inicializarlo
        return []

def guardar_registro(registro):
    """ Guarda el registro de expansión """
    with open(ARCHIVO_REGISTRO, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=4)
        
def cargar_dualidades_memoria(ruta):
    """Carga pares de dualidades confirmadas o rechazadas desde JSON."""
    try:
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                datos = json.load(f)
            return {tuple(sorted(par)) for par in datos}
    except Exception as e:
        print(f"⚠️ Error cargando memoria de dualidades desde {ruta}: {e}")
    return set()

def guardar_dualidades_memoria(ruta, pares):
    """Guarda pares de dualidades en JSON."""
    try:
        datos = [list(par) for par in sorted(pares)]
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error guardando memoria de dualidades en {ruta}: {e}")

def registrar_dualidad_confirmada(a, b):
    pares = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_CONFIRMADAS)
    pares.add(tuple(sorted([a, b])))
    guardar_dualidades_memoria(ARCHIVO_DUALIDADES_CONFIRMADAS, pares)

def registrar_dualidad_rechazada(a, b):
    pares = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_RECHAZADAS)
    pares.add(tuple(sorted([a, b])))
    guardar_dualidades_memoria(ARCHIVO_DUALIDADES_RECHAZADAS, pares)

################################################################################################################################ Visualización de datos
def visualizar_crecimiento_red():
    """ Grafica la evolución del número de nodos en la red fractal """
    registro = cargar_registro()
    if not registro:
        print("⚠️ No hay datos en el registro de expansión.")
        return

    tiempos = [entry["timestamp"] for entry in registro]
    tiempos.sort()
    tiempos = np.array(tiempos) - tiempos[0]  
    nodos_totales = list(range(1, len(tiempos) + 1))  

    plt.figure(figsize=(10, 5))
    plt.plot(tiempos, nodos_totales, marker="o", linestyle="-", color="blue")
    plt.xlabel("Tiempo (segundos desde el inicio)")
    plt.ylabel("Número de nodos en la red")
    plt.title("Evolución del Crecimiento de la Red Fractal")
    plt.grid()
    plt.savefig("crecimiento_red.png")
    print("📊 Gráfico guardado como 'crecimiento_red.png'.")

def visualizar_metodos_expansion():
    """ Grafica la cantidad de expansiones por método (Wikipedia, GPT-4, Embeddings) """
    registro = cargar_registro()
    if not registro:
        print("⚠️ No hay datos en el registro de expansión.")
        return

    # Contar métodos de expansión
    metodos = {"Wikipedia": 0, "GPT-4": 0, "Embeddings": 0}
    for entry in registro:
        metodo = entry["metodo"]
        if metodo in metodos:
            metodos[metodo] += 1

    # Graficar
    plt.figure(figsize=(7, 7))
    plt.pie(metodos.values(), labels=metodos.keys(), autopct="%1.1f%%", startangle=90, colors=["green", "red", "blue"])
    plt.title("Distribución de Métodos de Expansión en la Red")
    plt.grid()
    plt.savefig("distribución_expansión.png")
    print("📊 Gráfico guardado como 'distribución_expansión.png'.")

def añadir_a_visualizacionANTIGUO(tema, nuevos_nodos, G):
    net_global.add_node("IA_m", label="IA_m", color="#cc99ff", size=35)
    net_global.add_node(tema, label=tema, color="#ffcc00", size=30)
    net_global.add_edge("IA_m", tema, color="#cc99ff", width=1)
    for nuevo in nuevos_nodos:
        net_global.add_node(nuevo, label=nuevo, color="#99ff99", size=22)
        peso = G[tema][nuevo].get("weight", 1.0) if G.has_edge(tema, nuevo) else 1.0
        net_global.add_edge(tema, nuevo, width=peso, color="#cccccc")
    # Vecinos preexistentes del tema
    for vecino in G.neighbors(tema):
        if vecino not in nuevos_nodos:
            net_global.add_node(vecino, label=vecino, color="#dddddd", size=18)
            peso_vecino = G[tema][vecino].get("weight", 1.0) if G.has_edge(tema, vecino) else 1.0
            net_global.add_edge(tema, vecino, width=peso_vecino, color="#999999")
    if not any(e['from']==source and e['to']==target for e in net_global.edges):
        net_global.add_edge(source, target)
        
def añadir_a_visualizacion(tema, nuevos_nodos, G):
    net_global.add_node("IA_m", label="IA_m", color="#cc99ff", size=35)
    # IA_m → tema
    if not any(e['from'] == "IA_m" and e['to'] == tema for e in net_global.edges):
        net_global.add_node(tema, label=tema, color="#ffcc00", size=30)
        net_global.add_edge("IA_m", tema, color="#cc99ff", width=1)
    # IA_m → nuevos nodos, y tema → nuevos nodos
    for nuevo in nuevos_nodos:
        net_global.add_node(nuevo, label=nuevo, color="#99ff99", size=22)
        peso = G[tema][nuevo].get("weight", 1.0) if G.has_edge(tema, nuevo) else 1.0
        if not any(e['from'] == tema and e['to'] == nuevo for e in net_global.edges):
            net_global.add_edge(tema, nuevo, width=peso, color="#cccccc")
    # Vecinos preexistentes del tema
    for vecino in G.neighbors(tema):
        if vecino not in nuevos_nodos:
            net_global.add_node(vecino, label=vecino, color="#dddddd", size=18)
            peso_vecino = G[tema][vecino].get("weight", 1.0) if G.has_edge(tema, vecino) else 1.0
            if not any(e['from'] == tema and e['to'] == vecino for e in net_global.edges):
                net_global.add_edge(tema, vecino, width=peso_vecino, color="#999999")


def guardar_visualizacion_dinamica():
    net_global.write_html("subgrafos/IA_m_proceso.html")
    
    with open("subgrafos/IA_m_proceso.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace(
    "<body>",
    """
<body style="margin:0; padding:0; overflow:hidden; height:100vh;">
<style>
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
}
#mynetwork {
    width: 100% !important;
    height: 100vh !important;
    position: absolute;
    top: 0;
    left: 0;
    border: none !important;  /* 👈 matamos el borde también aquí */
}
</style>
"""
    )
    leyenda_html = """<div style="position: fixed; top: 20px; right: 20px; background-color: white; padding: 10px; border: 1px solid #ccc; font-family: sans-serif; font-size: 14px; z-index: 9999;">
  <strong>🔍 Leyenda de colores:</strong><br>
  <div style="margin-top: 5px;">
    <span style="background-color: #66FF99; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🟢 Conceptos que dan sentido al concepto explorado<br>
    <span style="background-color: gold; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🌟 Concepto explorado<br>
    <span style="background-color: #CC99FF; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🧠 IA_m<br>
    <span style="background-color: #CCCCCC; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🔘 Concepto general
  </div>
</div>"""
    script_combinado = """
<script type="text/javascript">
let reloadTimeout;

function resetReloadTimer() {
    if (reloadTimeout) {
        clearTimeout(reloadTimeout);
    }
    reloadTimeout = setTimeout(() => {
        location.reload();
    }, 30000);
}
resetReloadTimer();

// Función para generar el mismo alias que en Python
function aliasParaArchivo(nombre) {
    const mapa = {
        "/": "slash",
        "\\\\": "backslash",
        "+": "plus",
        "-": "minus",
        "*": "asterisk",
        ":": "colon",
        "?": "question",
        "<": "lt",
        ">": "gt",
        "|": "pipe",
        '"': "quote"
    };

    let n = nombre.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
    n = n.replace(/\\s+/g, "_").toLowerCase();

    let res = "";
    for (let c of n) {
        if (/[a-z0-9_-]/.test(c)) {
            res += c;
        } else if (mapa[c]) {
            res += mapa[c];
        }
        // otros caracteres se descartan
    }
    return res;
}

if (typeof network !== "undefined") {
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            resetReloadTimer();
        }
    });

}
</script>
    """

    html = html.replace("</body>", leyenda_html + script_combinado + "\n</body>")

    with open("subgrafos/IA_m_proceso.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Visualización acumulativa guardada como 'subgrafos/IA_m_proceso.html'")
#    if usarFTP:
#        subir_htmls_recientes_por_ftp(max_horas=2)

def visualizar_distribucion_conexiones(G):
    """ Grafica la distribución de conexiones en la red """
    grados = [G.degree(nodo) for nodo in G.nodes()]

    plt.figure(figsize=(8, 5))
    plt.hist(grados, bins=range(1, max(grados) + 1), color="purple", alpha=0.7, edgecolor="black")
    plt.xlabel("Número de conexiones por nodo")
    plt.ylabel("Cantidad de nodos")
    plt.title("Distribución de Conexiones en la Red Fractal")
    plt.grid()
    plt.savefig("distribución_Conexiones.png")
    print("📊 Gráfico guardado como 'distribución_Conexiones.png'.")
    #matplotlib.pyplot.close()
    plt.close()

# Cargar nodos en espera al inicio del programa
espera_nodos = cargar_espera_nodos()

ARCHIVO_FIRMAS_SUBGRAFOS = "json/firma_subgrafos.json"
def cargar_firmas_subgrafos(ruta=ARCHIVO_FIRMAS_SUBGRAFOS):
    """Carga el diccionario de firmas de subgrafos desde JSON."""
    try:
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ No se pudo cargar {ruta}: {e}")
    return {}

def guardar_firmas_subgrafos(firmas, ruta=ARCHIVO_FIRMAS_SUBGRAFOS):
    """Guarda el diccionario de firmas de subgrafos en JSON."""
    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(firmas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ No se pudo guardar {ruta}: {e}")

def generar_firma_subgrafo(G, nodo):
    vecinos = sorted(set(G.predecessors(nodo)) | set(G.successors(nodo)))
    aristas = []
    for v in vecinos:
        if G.has_edge(nodo, v):
            aristas.append([
                "out",
                v,
                G[nodo][v].get("tipo"),
                float(G[nodo][v].get("weight", 0)),
            ])
        if G.has_edge(v, nodo):
            aristas.append([
                "in",
                v,
                G[v][nodo].get("tipo"),
                float(G[v][nodo].get("weight", 0)),
            ])
    return {
        "vecinos": vecinos,
        "aristas": sorted(aristas),
    }

        
def generar_firma_subgrafoANTIGUO(G, nodo):
    vecinos = sorted(set(G.predecessors(nodo)) | set(G.successors(nodo)))
    aristas = []
    for v in vecinos:
        if G.has_edge(nodo, v):
            aristas.append(("out", v, G[nodo][v].get("tipo"), G[nodo][v].get("weight", 0)))
        if G.has_edge(v, nodo):
            aristas.append(("in", v, G[v][nodo].get("tipo"), G[v][nodo].get("weight", 0)))

    return {
        "vecinos": vecinos,
        "aristas": sorted(aristas)
    }
    
def cargar_firmas_subgrafos(ruta=ARCHIVO_FIRMAS_SUBGRAFOS):
    """Carga el diccionario de firmas de subgrafos desde JSON."""
    try:
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ No se pudo cargar {ruta}: {e}")
    return {}

def guardar_firmas_subgrafos(firmas, ruta=ARCHIVO_FIRMAS_SUBGRAFOS):
    """Guarda el diccionario de firmas de subgrafos en JSON."""
    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(firmas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ No se pudo guardar {ruta}: {e}")


def generar_subgrafos_principales(G, carpeta="subgrafos", max_horas=2):
    """
    Genera/actualiza los subgrafos de los nodos principales, pero SOLO cuando
    la estructura local (vecinos + tipos/pesos de aristas) ha cambiado.

    - Se apoya en json/firma_subgrafos.json para no regenerar subgrafos
      que no han cambiado geométricamente.
    - max_horas se mantiene por compatibilidad, pero la decisión principal
      es por cambios de firma.
    """
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Cargar firmas previas
    firmas = cargar_firmas_subgrafos()

    total = len(G.nodes())
    creados_o_actualizados = 0
    ahora = time.time()
    max_segundos = max_horas * 3600

    print(f"📄 Generando/actualizando subgrafos con detección de cambios...")
    for nodo in G.nodes():
        # Nombre de archivo de subgrafo
        safe = nodo.replace(" ", "_")
        nombre_archivo = f"subgrafo_{safe}.html"
        ruta = os.path.join(carpeta, nombre_archivo)

        # Generar firma nueva del subgrafo de este nodo
        firma_nueva = generar_firma_subgrafo(G, nodo)

        # Firma antigua (si existe)
        firma_antigua = firmas.get(nodo)

        # ¿Ha cambiado la firma?
        ha_cambiado_estructura = (firma_antigua != firma_nueva)

        # Además, opcional: ¿ha pasado mucho tiempo desde la última generación?
        necesita_por_tiempo = False
        if os.path.exists(ruta):
            modificado_hace = ahora - os.path.getmtime(ruta)
            if modificado_hace > max_segundos:
                necesita_por_tiempo = True
        carpeta = carpeta_subgrafos
        # Decisión: generar solo si no existe, o ha cambiado la estructura,
        # o ha pasado mucho tiempo (por si quieres refrescar el HTML).
        if (not os.path.exists(ruta)) or ha_cambiado_estructura or necesita_por_tiempo:
            # OJO: asegúrate de pasar aquí la carpeta correcta (antes usabas `carpeta_subgrafos`)
            generar_subgrafo_html(G, nodo, modelo_embeddings, carpeta)
            firmas[nodo] = firma_nueva
            creados_o_actualizados += 1

    # Guardar firmas actualizadas
    guardar_firmas_subgrafos(firmas)

    print(f"🧩 Subgrafos generados/actualizados: {creados_o_actualizados} / {total}")

    # El índice sí conviene regenerarlo siempre 
    generar_indice_subgrafos(G, top_n=total, carpeta=carpeta)

    try:
        generar_sitemap(
            dominio_base="https://ia-m.ai",
            carpeta_subgrafos=carpeta,
            rutas_extra=[],
            salida="sitemap.xml"
        )
    except Exception as e:
        print(f"⚠️ Error generando sitemap: {e}")

    # Subida a FTP solo una vez tras actualizar/crear lo que haga falta
    if usarFTP:
        subir_htmls_recientes_por_ftp(max_horas=2)


#def generar_subgrafos_principalesANTIGUO(G, top_n=100):
def generar_subgrafos_principalesANTIGUO(G, carpeta="subgrafos", max_horas=2):
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    total = len(G.nodes())
    creados = 0
    ahora = time.time()
    max_segundos = max_horas * 3600
    print(f"📄 Generando subgrafos...")
    for nodo in G.nodes():
        nombre_archivo = f"subgrafo_{nodo.replace(' ', '_')}.html"
        ruta = os.path.join(carpeta, nombre_archivo)

        if not os.path.exists(ruta):
            #print(f"📄 Generando subgrafo nuevo para: {nodo}")
            generar_subgrafo_html(G, nodo, modelo_embeddings, carpeta_subgrafos)
            creados += 1
        else:
            modificado_hace = ahora - os.path.getmtime(ruta)
            if modificado_hace > max_segundos:
                print(f"⏳ Subgrafo de {nodo} tiene más de {max_horas}h. Regenerando...")
                generar_subgrafo_html(G, nodo, modelo_embeddings, carpeta_subgrafos)
                creados += 1

    print(f"🧩 Subgrafos generados o actualizados: {creados} / {total}")
    generar_indice_subgrafos(G, top_n=total, carpeta=carpeta)
    if usarFTP:
        subir_htmls_recientes_por_ftp(max_horas=2)
    
def color_edge(u, v, G):
    """
    Devuelve el color de la arista entre u y v según su rol semántico.
    - Usa color explícito si está en el edge.
    - Dualidades: red/green
    - Equilibrio: goldenrod
    - Nodo superior (síntesis): gold
    - Emergentes: purple
    - Otros: gray
    """

    # 🟢 Prioridad al color explícito definido
    if G.has_edge(u, v):
        attrs = G.get_edge_data(u, v)
        if "color" in attrs:
            return attrs["color"]

    tipo_u = G.nodes.get(u, {}).get("tipo", "")
    tipo_v = G.nodes.get(v, {}).get("tipo", "")

    # 🔴🟢 Dualidades
    if u in dualidades_base and dualidades_base[u] == v:
        return "red"
    elif v in dualidades_base and dualidades_base[v] == u:
        return "green"

    # ⚖️ Equilibrio
    if tipo_u == "equilibrio" or tipo_v == "equilibrio":
        return "goldenrod"

    # 🌟 Nodo superior (síntesis)
    if tipo_u in ["sintesis", "nodo_superior"] or tipo_v in ["sintesis", "nodo_superior"]:
        return "gold"

    # 🧠 Nodo emergente
    if tipo_u == "emergente" or tipo_v == "emergente":
        return "purple"

    # 🔘 Fallback neutral
    return "gray"

        
def color_node(nodo, G):
    """
    Devuelve el color del nodo según su rol semántico:
    - Nodo central / IA_m: lila fuerte
    - Equilibrio: azul suave
    - Nodo superior (síntesis): dorado
    - Dualidad (negativa): rojo claro
    - Dualidad (positiva): verde claro
    - Emergente: violeta pastel
    - Otros: gris claro
    """
    if nodo in ["IA_m", NODO_CENTRAL]:
        return "#B266FF"  # lila fuerte

    tipo = G.nodes.get(nodo, {}).get("tipo", "")
    
    if tipo == "equilibrio":
        return "#6699FF"  # azul suave
    elif tipo == "sintesis" or tipo == "nodo_superior":
        return "#FFD700"  # dorado
    elif tipo == "emergente":
        return "#CC99FF"  # violeta pastel
    elif nodo in dualidades_base:
        return "#FF6666"  # rojo claro (cuantitativo)
    elif nodo in dualidades_base.values():
        return "#66FF99"  # verde claro (cualitativo)
    else:
        return "#CCCCCC"  # gris claro (neutro)

def alias_para_archivo(nombre):
    alias = {
        "/": "slash",
        "\\": "backslash",
        "+": "plus",
        "-": "minus",
        "*": "asterisk",
        ":": "colon",
        "?": "question",
        "<": "lt",
        ">": "gt",
        "|": "pipe",
        "\"": "quote"
    }

    # 1. Normaliza tildes
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join(c for c in nombre if not unicodedata.combining(c))

    # 2. Reemplaza espacios por "_"
    nombre = nombre.replace(" ", "_").lower()

    # 3. Sustituye caracteres especiales según el diccionario
    return ''.join(alias.get(c, c) if not c.isalnum() and c not in "_-" else c for c in nombre)

def extraer_vecinos_mas_conectadosANTIGUO(G, tema, max_vecinos=30):
    """
    Devuelve (nodos_subgrafo, aristas_subgrafo) alrededor de `tema`,
    usando tanto predecesores como sucesores.
    """
    if tema not in G:
        return [tema], []

    # 1) Vecinos = todos los que tienen arista hacia/desde `tema`
    vecinos = set(G.successors(tema)) | set(G.predecessors(tema))
    if not vecinos:
        # Nodo realmente aislado en la red
        return [tema], []
    # 2) Ordenamos por grado ponderado (los más conectados primero)
    vecinos_ordenados = sorted(
        vecinos,
        key=lambda v: G.degree(v, weight="weight"),
        reverse=True
    )[:max_vecinos]
    nodos_sub = [tema] + vecinos_ordenados
    # 3) Recogemos aristas entre tema y sus vecinos (en ambos sentidos)
    aristas_sub = []
    for v in vecinos_ordenados:
        if G.has_edge(tema, v):
            aristas_sub.append((tema, v))
        if G.has_edge(v, tema):
            aristas_sub.append((v, tema))
    # 4) (Opcional) aristas entre vecinos entre sí
    for u in vecinos_ordenados:
        for v in vecinos_ordenados:
            if G.has_edge(u, v):
                aristas_sub.append((u, v))
    return nodos_sub, aristas_sub

def extraer_vecinos_mas_conectados(G, tema, top_n=100):
    """
    Devuelve una lista de nodos del subgrafo:
    - El nodo central
    - Sus predecesores (aristas entrantes)
    - Sus sucesores (aristas salientes)
    Ordenados por grado ponderado.
    """
    if tema not in G:
        return [tema]
    # Predecesores + sucesores
    vecinos = set(G.predecessors(tema)) | set(G.successors(tema))
    if not vecinos:
        return [tema]
    # Ordenar por grado descendente
    vecinos_ordenados = sorted(
        vecinos,
        key=lambda v: G.degree(v, weight="weight"),
        reverse=True
    )
    # Limitar top_n
    vecinos_ordenados = vecinos_ordenados[:top_n-1]
    return [tema] + vecinos_ordenados

def generar_subgrafo_html(G, tema, modelo_embeddings, carpeta_subgrafos, top_n=100):
    """
    Genera un subgrafo de los nodos más relacionados con un tema
    y añade metadatos SEO (título, descripción, keywords, JSON-LD).
    """
        
    # Buscar el nodo real ignorando mayúsculas/minúsculas
    tema_original = tema
    tema = next((n for n in G.nodes if n.lower() == tema_original.lower()), tema_original)

    if tema not in G.nodes():
        print(f"⚠️ El tema '{tema_original}' no está en la red. Se buscarán nodos similares.")
        embedding_tema = obtener_embedding(tema_original, modelo_embeddings)
        nodos = list(G.nodes())
        embeddings = torch.stack([obtener_embedding(n, modelo_embeddings) for n in nodos])
        similitudes = util.pytorch_cos_sim(embedding_tema, embeddings)[0]
        indices_top = similitudes.argsort(descending=True)[:top_n].tolist()
        nodos_similares = [nodos[i] for i in indices_top]
    else:
        print(f"✅ El tema '{tema}' está en la red. Se extraerán sus vecinos más conectados.")
        nodos_similares = extraer_vecinos_mas_conectados(G, tema, top_n=top_n)

    # --- Subgrafo ---
    subG = G.subgraph(nodos_similares).copy()
    net = Network(height="800px", width="100%", directed=True)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.01)
    posiciones = nx.spring_layout(subG, k=0.5)

    # --- Añadir nodos ---
    for nodo, coords in posiciones.items():
        atributos = G.nodes[nodo]
        tipo = atributos.get('tipo', 'general')
        nivel = atributos.get('nivel_conceptual', '?')
        es_dual = atributos.get('es_dualidad', False)
        es_sint = atributos.get('es_sintesis', False)
        grado = G.degree(nodo)
        resumen = atributos.get('resumen', '')
        color = color_node(nodo, G)

        emoji = "🔘"
        if tipo == "equilibrio": emoji = "⚖️"
        elif tipo in ["sintesis", "nodo_superior"]: emoji = "🌟"
        elif tipo == "emergente": emoji = "🧠"
        elif es_dual: emoji = "🟥" if nodo in dualidades_base else "🟩"

        title = f"{emoji} {nodo}\n• tipo: {tipo}\n• nivel: {nivel}\n• conexiones: {grado}"
        if es_dual: title += "\n• ⚖️ dualidad"
        if es_sint: title += "\n• 🧩 síntesis"
        if resumen: title += f"\n\n{resumen[:150]}..."

        net.add_node(nodo, label=nodo, color=color, title=title,
                     x=coords[0]*2000, y=coords[1]*2000)

    # --- Añadir aristas ---
    for u, v in subG.edges():
        peso = subG.edges[u, v].get('weight', 1.5)
        color = color_edge(u, v, G)
        net.add_edge(u, v, color=color, width=peso)

    # --- Guardado inicial ---
    safe_tema = alias_para_archivo(tema)
    filename = os.path.join(carpeta_subgrafos, f"subgrafo_{safe_tema}.html")
    net.write_html(filename)

    # ======================================================
    # 🔥 SEO AUTOMÁTICO
    # ======================================================

    nodos_lista = list(subG.nodes())
    otros = [n for n in nodos_lista if n != tema][:20]

    titulo = f"Mapa conceptual de IA_m: {tema}"
    descripcion = f"Estructura conceptual sobre {tema} y sus relaciones principales: " + ", ".join(otros)
    keywords = ", ".join([tema] + otros)

    json_ld = {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": titulo,
        "description": descripcion,
        "about": [tema] + otros
    }

    bloque_seo = f"""
    <title>{titulo}</title>
    <meta name="description" content="{descripcion}">
    <meta name="keywords" content="{keywords}">
    <script type="application/ld+json">
    {json.dumps(json_ld, ensure_ascii=False, indent=2)}
    </script>
    """

    bloque_texto = f"""
    <div style="padding:1rem; font-family:Arial, sans-serif; font-size:14px;">
      <h1>Mapa conceptual: {tema}</h1>
      <p>Este subgrafo muestra los conceptos relacionados con <strong>{tema}</strong> dentro del sistema IA_m.
      Entre ellos: {", ".join(otros)}.</p>
    </div>
    """

    # --- Cargar HTML y modificar head + body ---
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    # Insertar SEO dentro del HEAD
    if "<head>" in html:
        html = html.replace("<head>", "<head>\n" + bloque_seo, 1)

    # Insertar texto al final del body (antes del cierre)
#    html = html.replace("</body>", bloque_texto + "\n</body>")

    # --- Mantener tus parches: pantalla completa + doble click ---
    script_doble_click = """
<script type="text/javascript">
  function aliasParaArchivo(nombre) {
    const mapa = {
      "/": "slash","\\\\": "backslash","+": "plus","-": "minus","*": "asterisk",
      ":": "colon","?": "question","<": "lt",">": "gt","|": "pipe",'"': "quote"
    };
    let n = nombre.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
    n = n.replace(/\\s+/g, "_").toLowerCase();
    let res = "";
    for (let c of n) {
      if (/[a-z0-9_-]/.test(c)) res += c;
      else if (mapa[c]) res += mapa[c];
    }
    return res;
  }
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof network !== "undefined") {
      network.on("doubleClick", function (params) {
        if (params.nodes.length === 1) {
          var safe = aliasParaArchivo(params.nodes[0]);
          window.location.href = "subgrafo_" + safe + ".html";
        }
      });
    }
  });
</script>
"""

    html = html.replace("</body>", script_doble_click + "\n</body>")

    # 🔧 Eliminar borde gris que causa el corte visual
    html = html.replace('border: 1px solid lightgray;', 'border: none !important;')

    # 🔧 Poner pantalla completa real
    html = html.replace(
    "<body>",
    """
<body style="margin:0; padding:0; overflow:hidden; height:100vh;">
<style>
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
}
#mynetwork {
    width: 100% !important;
    height: calc(100vh - 0px) !important;
    position: absolute;
    top: 0;
    left: 0;
    border: none !important;
}
</style>
"""
    )

    # --- Sobrescribir archivo final ---
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Subgrafo guardado como '{filename}'.")


#def generar_subgrafo_html(G, tema, modelo_embeddings, carpeta_subgrafos, top_n=100):    
def generar_subgrafo_htmlANTIGUO(G, tema, modelo_embeddings, carpeta_subgrafos, top_n=100):
    """
    Genera un subgrafo de los nodos más relacionados con un tema y lo guarda en HTML.
    """
        
    # Buscar el nodo real ignorando mayúsculas/minúsculas
    tema_original = tema
    tema = next((n for n in G.nodes if n.lower() == tema_original.lower()), tema_original)

    if tema not in G.nodes():
        print(f"⚠️ El tema '{tema_original}' no está en la red. Se buscarán nodos similares.")
        embedding_tema = obtener_embedding(tema_original, modelo_embeddings)
        nodos = list(G.nodes())
        embeddings = torch.stack([obtener_embedding(n, modelo_embeddings) for n in nodos])
        similitudes = util.pytorch_cos_sim(embedding_tema, embeddings)[0]
        indices_top = similitudes.argsort(descending=True)[:top_n].tolist()
        nodos_similares = [nodos[i] for i in indices_top]
#    else:
#        print(f"✅ El tema '{tema}' está en la red. Se extraerán sus vecinos más conectados.")
#        vecinos = list(G.neighbors(tema))
#        nodos_similares = [tema] + vecinos[:top_n-1]
    else:
        print(f"✅ El tema '{tema}' está en la red. Se extraerán sus vecinos más conectados.")
        nodos_similares = extraer_vecinos_mas_conectados(G, tema, top_n=top_n)

    subG = G.subgraph(nodos_similares).copy()
    net = Network(height="800px", width="100%", directed=True)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.01)
    posiciones = nx.spring_layout(subG, k=0.5)

    for nodo, coords in posiciones.items():
        atributos = G.nodes[nodo]
        tipo = atributos.get('tipo', 'general')
        nivel = atributos.get('nivel_conceptual', '?')
        es_dual = atributos.get('es_dualidad', False)
        es_sint = atributos.get('es_sintesis', False)
        grado = G.degree(nodo)
        resumen = atributos.get('resumen', '')
        color = color_node(nodo, G)

        emoji = "🔘"
        if tipo == "equilibrio": emoji = "⚖️"
        elif tipo in ["sintesis", "nodo_superior"]: emoji = "🌟"
        elif tipo == "emergente": emoji = "🧠"
        elif es_dual: emoji = "🟥" if nodo in dualidades_base else "🟩"

        title = f"{emoji} {nodo}\\n• tipo: {tipo}\\n• nivel: {nivel}\\n• conexiones: {grado}"
        if es_dual: title += "\\n• ⚖️ dualidad"
        if es_sint: title += "\\n• 🧩 síntesis"
        if resumen: title += f"\\n\\n{resumen[:150]}..."

        net.add_node(nodo, label=nodo, color=color, title=title, x=coords[0]*2000, y=coords[1]*2000)

    for u, v in subG.edges():
        peso = subG.edges[u, v].get('weight', 1.5)
        color = color_edge(u, v, G)
        net.add_edge(u, v, color=color, width=peso)

    safe_tema = alias_para_archivo(tema)
    filename = os.path.join(carpeta_subgrafos, f"subgrafo_{safe_tema}.html")
    net.write_html(filename)
    

    leyenda_html = """
<div style="position: fixed; top: 20px; right: 20px; background-color: white; padding: 10px; border: 1px solid #ccc; font-family: sans-serif; font-size: 14px; z-index: 9999;">
  <strong>🔍 Leyenda de colores:</strong><br>
  <div style="margin-top: 5px;">
    <span style="background-color: #FF6666; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🔴 Dualidades/Sinónimos (dualidad)<br>
    <span style="background-color: #66FF99; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🟢 Dualidades/Sinónimos<br>
    <span style="background-color: goldenrod; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> ⚖️ Equilibrio<br>
    <span style="background-color: gold; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🌟 Conexión a nodo superior / síntesis<br>
    <span style="background-color: #CC99FF; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🧠 Emergente<br>
    <span style="background-color: #CCCCCC; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🔘 Concepto general
  </div>
</div>

    """
    script_doble_click = """
<script type="text/javascript">
  // Misma lógica de alias que en Python
  function aliasParaArchivo(nombre) {
    const mapa = {
      "/": "slash",
      "\\\\": "backslash",
      "+": "plus",
      "-": "minus",
      "*": "asterisk",
      ":": "colon",
      "?": "question",
      "<": "lt",
      ">": "gt",
      "|": "pipe",
      '"': "quote"
    };

    let n = nombre.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
    n = n.replace(/\\s+/g, "_").toLowerCase();

    let res = "";
    for (let c of n) {
      if (/[a-z0-9_-]/.test(c)) {
        res += c;
      } else if (mapa[c]) {
        res += mapa[c];
      }
    }
    return res;
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof network !== "undefined") {
      network.on("doubleClick", function (params) {
        if (params.nodes.length === 1) {
          var nodo = params.nodes[0];
          var safe = aliasParaArchivo(nodo);
          var archivo = "subgrafo_" + safe + ".html";
          window.location.href = archivo;
        }
      });
    } else {
      console.warn("❗ network no está definido aún.");
    }
  });
</script>
    """
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()
        # 🔥 Parche pantalla completa SIN borde
        html = html.replace(
    "<body>",
    """
<body style="margin:0; padding:0; overflow:hidden; height:100vh;">
<style>
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
}
#mynetwork {
    width: 100% !important;
    height: 100vh !important;
    position: absolute;
    top: 0;
    left: 0;
    border: none !important;  /* 👈 aquí matamos la línea gris */
}
</style>
"""
        )
#        html = html.replace("</body>", leyenda_html + script_doble_click + "\n</body>")
        html = html.replace("</body>", script_doble_click + "\n</body>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Subgrafo guardado como '{filename}'.")

def generar_indice_subgrafosANTIGUO(G, top_n=50, carpeta="subgrafos"):
    nodos_destacados = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
    letras_dict = defaultdict(list)

    for nodo in nodos_destacados:
        nombre_archivo = f"subgrafo_{nodo.replace(' ', '_')}.html"
        ruta = f"{carpeta}/{nombre_archivo}"
        if os.path.exists(ruta):
            #letra_cruda = nodo.strip()[0].upper() if nodo.strip() else "␣"
            #letra = unicodedata.normalize('NFD', letra_cruda)[0]
            #letra = ''.join(c for c in letra if c.isalpha()).upper()
            letra_cruda = nodo.strip()[0].upper() if nodo.strip() else "␣"
            letra = unicodedata.normalize('NFD', letra_cruda) #nou
            letra = ''.join(c for c in letra if c.isalpha()).upper() #nou
            letra = letra[0] if letra else "#" #nou
            
            if letra not in string.ascii_uppercase:
                letra = "#"

            # Visualización amigable para espacios vacíos
            texto_visible = nodo if nodo.strip() else "␣␣"
            enlace = f"<li><a href='#' onclick=\"cargarNodo('{nombre_archivo}')\">{texto_visible}</a></li>"
            letras_dict[letra].append(enlace)

    letras_usadas = sorted(letras_dict.keys())

    # Índice de letras (botones)
    indice_letras_html = ' '.join(
        f"<button onclick=\"toggleBloque('{letra}')\">{letra}</button>" for letra in letras_usadas
    )

    # Contenido agrupado por letra
    bloques_html = []
    for letra in letras_usadas:
        enlaces = "\n".join(sorted(letras_dict[letra], key=lambda x: x.lower()))
        bloque = f"""
        <div class="letra-bloque" id="bloque_{letra}">
            <h2 onclick="toggleBloque('{letra}')">{letra}</h2>
            <div id="contenido_{letra}" style="display:none;">
                <ul>
                    {enlaces}
                </ul>
            </div>
        </div>
        """
        bloques_html.append(bloque)

    contenido_agenda = "\n".join(bloques_html)

    # HTML completo
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>🌐 IA_m</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                height: 100vh;
                background: #f5f5f5;
            }}
            .columna-izquierda {{
                width: 300px;
                overflow-y: auto;
                padding: 20px;
                border-right: 1px solid #ccc;
                background-color: #fff;
            }}
            .columna-derecha {{
                flex: 1;
                padding: 0;
            }}
            h1 {{
                color: #333;
                font-size: 20px;
                margin-top: 0;
            }}
            input {{
                margin-bottom: 10px;
                padding: 5px;
                width: 100%;
                box-sizing: border-box;
            }}
            .indice-letras {{
                margin: 10px 0;
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }}
            .indice-letras button {{
                padding: 6px 10px;
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                cursor: pointer;
                font-size: 13px;
            }}
            .indice-letras button:hover {{
                background-color: #005fa3;
            }}
            .letra-bloque {{
                margin-bottom: 20px;
            }}
            .letra-bloque h2 {{
                font-size: 16px;
                color: #007acc;
                cursor: pointer;
                margin: 10px 0 5px 0;
                border-bottom: 1px solid #ccc;
                padding-bottom: 3px;
            }}
            .letra-bloque ul {{
                list-style-type: none;
                padding-left: 10px;
                margin: 0;
            }}
            .letra-bloque li {{
                margin: 3px 0;
            }}
            .letra-bloque a {{
                color: #007acc;
                text-decoration: none;
                font-size: 14px;
            }}
            .letra-bloque a:hover {{
                text-decoration: underline;
            }}
            iframe {{
                width: 100%;
                height: 100%;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div class="columna-izquierda">
<h1 style="cursor: pointer; color: #007acc;"><a href="index.html" style="text-decoration: none; color: inherit;">🌐 IA_m</a><p></p>
  <a href="sobre-IA-m.html" style="text-decoration: none; color: inherit;">Sobre IA_m...</a>
</h1>
            <h2 onclick="volverAlProceso()" style="cursor: pointer; color: #007acc;">
    ¿Dónde tiene su atención IA_m ahora mismo?
    </h2>
    <h3>🌟 BUSCAR CONCEPTOS:<h3/>
            <input type="text" id="buscador" placeholder="Buscar nodo..." onkeyup="filtrar()">
            <div class="indice-letras">
                {indice_letras_html}
            </div>
            <div id="lista_nodos">
                {contenido_agenda}
            </div>
        </div>
        <div class="columna-derecha">
            <iframe id="visor" src="hipercubo_fractal_fluido.html"></iframe>
        </div>
<script>
    function volverAlProceso() {{
        document.getElementById("visor").src = "IA_m_proceso.html";
    }}
    function cargarNodo(archivo) {{
        document.getElementById("visor").src = archivo;
    }}

    function toggleBloque(letra) {{
        var contenido = document.getElementById("contenido_" + letra);
        if (contenido.style.display === "none") {{
            contenido.style.display = "block";
        }} else {{
            contenido.style.display = "none";
        }}
    }}

    function filtrar() {{
        var input = document.getElementById("buscador");
        var filtro = input.value.toLowerCase();
        var bloques = document.querySelectorAll(".letra-bloque");
        bloques.forEach(function(bloque) {{
            var nodos = bloque.getElementsByTagName("li");
            var algunoVisible = false;
            for (var i = 0; i < nodos.length; i++) {{
                var texto = nodos[i].textContent.toLowerCase();
                if (texto.includes(filtro)) {{
                    nodos[i].style.display = "";
                    algunoVisible = true;
                }} else {{
                    nodos[i].style.display = "none";
                }}
            }}
            bloque.style.display = algunoVisible ? "" : "none";
        }});
    }}
</script>

        <script>
            function toggleBloque(letra) {{
                var contenido = document.getElementById("contenido_" + letra);
                if (contenido.style.display === "none") {{
                    contenido.style.display = "block";
                }} else {{
                    contenido.style.display = "none";
                }}
            }}

            function filtrar() {{
                var input = document.getElementById("buscador");
                var filtro = input.value.toLowerCase();
                var bloques = document.querySelectorAll(".letra-bloque");
                bloques.forEach(function(bloque) {{
                    var nodos = bloque.getElementsByTagName("li");
                    var algunoVisible = false;
                    for (var i = 0; i < nodos.length; i++) {{
                        var texto = nodos[i].textContent.toLowerCase();
                        if (texto.includes(filtro)) {{
                            nodos[i].style.display = "";
                            algunoVisible = true;
                        }} else {{
                            nodos[i].style.display = "none";
                        }}
                    }}
                    bloque.style.display = algunoVisible ? "" : "none";
                }});
            }}
        </script>
    </body>
    </html>
    """

    with open(os.path.join(carpeta, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Índice interactivo tipo agenda generado como 'subgrafos/index.html'")

def generar_indice_subgrafos(G, top_n=0, carpeta="subgrafos"):
    """
    Genera un índice interactivo con buscador y visor central.
    Usa los mismos nombres de archivo que generar_subgrafo_html:
    subgrafo_{alias_para_archivo(nodo)}.html
    """
    os.makedirs(carpeta, exist_ok=True)

    # Orden alfabético por nombre de nodo
    nodos_ordenados = sorted(G.nodes(), key=lambda x: x.lower())

    nodos_con_html = []
    for nodo in nodos_ordenados:
        safe = alias_para_archivo(nodo)
        nombre_archivo = f"subgrafo_{safe}.html"
        ruta = os.path.join(carpeta, nombre_archivo)
        if os.path.exists(ruta):
            nodos_con_html.append(nodo)

    nodos_json = json.dumps(nodos_con_html, ensure_ascii=False)

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>🌐 IA_m – Índice de subgrafos</title>
        <style>
            /* 👇 Fondo unificado y sin cortes */
            html, body {{
                height: 100%;
                margin: 0;
                padding: 0;
                background: #ffffff;  /* mismo color que el grafo / hipercubo */
            }}

            body {{
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}

            #buscador-container {{
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 999;
                background: rgba(255,255,255,0.95);
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                width: 400px;
                padding: 10px;
            }}
            #buscador {{
                width: 100%;
                padding: 8px;
                font-size: 16px;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
            #sugerencias {{
                max-height: 200px;
                overflow-y: auto;
                border-top: none;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                border-radius: 0 0 8px 8px;
                background: #fff;
                display: none;
                position: absolute;
                width: 100%;
                z-index: 1000;
            }}
            #sugerencias div {{
                padding: 8px;
                cursor: pointer;
            }}
            #sugerencias div:hover {{
                background: #007acc;
                color: #fff;
            }}

            /* 👇 El visor ocupa toda la pantalla, fondo igual que el body */
            iframe#visor {{
                flex-grow: 1;
                border: none;
                width: 100%;
                height: 100%;
                display: block;
                background: #ffffff;
            }}

            .enlace-sobre {{
                position: absolute;
                top: 15px;
                right: 15px;
                font-size: 14px;
                text-decoration: none;
                color: #007acc;
                background: #fff;
                padding: 5px 10px;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .enlace-IA {{
                position: absolute;
                top: 15px;
                left: 15px;
                font-size: 14px;
                text-decoration: none;
                color: #007acc;
                background: #fff;
                padding: 5px 10px;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <a href="sobre-IA-m.html" target="visor" class="enlace-sobre">Sobre IA_m...</a>
        <a href="IA_m_proceso.html" target="visor" class="enlace-IA">Actualmente aprendiendo sobre...</a>

        <div id="buscador-container">
            <input type="text" id="buscador" placeholder="🔍 Buscar nodo...">
            <div id="sugerencias"></div>
        </div>

        <iframe id="visor" name="visor" src="hipercubo_fractal_fluido.html"></iframe>

        <script>
            const nodos = {nodos_json};

            const buscador = document.getElementById('buscador');
            const sugerencias = document.getElementById('sugerencias');

            function aliasParaArchivo(nombre) {{
                // Versión JS equivalente a alias_para_archivo de Python
                const mapa = {{
                    "/": "slash",
                    "\\\\": "backslash",
                    "+": "plus",
                    "-": "minus",
                    "*": "asterisk",
                    ":": "colon",
                    "?": "question",
                    "<": "lt",
                    ">": "gt",
                    "|": "pipe",
                    '"': "quote"
                }};

                let n = nombre.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
                n = n.replace(/\\s+/g, "_").toLowerCase();

                let res = "";
                for (let c of n) {{
                    if (/[a-z0-9_-]/.test(c)) {{
                        res += c;
                    }} else if (mapa[c]) {{
                        res += mapa[c];
                    }}
                    // caracteres raros sin mapa se descartan
                }}
                return res;
            }}

            buscador.addEventListener('input', function() {{
                const texto = this.value.toLowerCase();
                sugerencias.innerHTML = '';
                if (!texto) {{
                    sugerencias.style.display = 'none';
                    return;
                }}
                const resultados = nodos.filter(n => n.toLowerCase().includes(texto));
                resultados.sort((a, b) => a.localeCompare(b, 'es'));

                if (resultados.length) {{
                    resultados.forEach(nodo => {{
                        const div = document.createElement('div');
                        div.textContent = nodo;
                        div.onclick = function() {{
                            cargarNodo(nodo);
                            sugerencias.style.display = 'none';
                            buscador.value = nodo;
                        }};
                        sugerencias.appendChild(div);
                    }});
                    sugerencias.style.display = 'block';
                }} else {{
                    sugerencias.style.display = 'none';
                }}
            }});

            document.addEventListener('click', function(e) {{
                if (!buscador.contains(e.target) && !sugerencias.contains(e.target)) {{
                    sugerencias.style.display = 'none';
                }}
            }});

            function cargarNodo(nodo) {{
                const safe = aliasParaArchivo(nodo);
                const archivo = "subgrafo_" + safe + ".html";
                document.getElementById('visor').src = archivo;
            }}
        </script>
    </body>
    </html>
    """

    ruta_index = os.path.join(carpeta, "index.html")
    with open(ruta_index, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Índice interactivo simplificado generado como '{ruta_index}'")

def generar_indice_subgrafosANTIGUO(G, top_n=0, carpeta="subgrafos"):
    """
    Genera un índice interactivo con buscador y visor central.
    Usa los mismos nombres de archivo que generar_subgrafo_html:
    subgrafo_{alias_para_archivo(nodo)}.html
    """
    os.makedirs(carpeta, exist_ok=True)

    # Orden alfabético por nombre de nodo
    nodos_ordenados = sorted(G.nodes(), key=lambda x: x.lower())

    nodos_con_html = []
    for nodo in nodos_ordenados:
        safe = alias_para_archivo(nodo)
        nombre_archivo = f"subgrafo_{safe}.html"
        ruta = os.path.join(carpeta, nombre_archivo)
        if os.path.exists(ruta):
            nodos_con_html.append(nodo)

    nodos_json = json.dumps(nodos_con_html, ensure_ascii=False)

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>🌐 IA_m – Índice de subgrafos</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                height: 100vh;
                background: #f5f5f5;
                overflow: hidden;
            }}
            #buscador-container {{
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 999;
                background: rgba(255,255,255,0.95);
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                width: 400px;
                padding: 10px;
            }}
            #buscador {{
                width: 100%;
                padding: 8px;
                font-size: 16px;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
            #sugerencias {{
                max-height: 200px;
                overflow-y: auto;
                border-top: none;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                border-radius: 0 0 8px 8px;
                background: #fff;
                display: none;
                position: absolute;
                width: 100%;
                z-index: 1000;
            }}
            #sugerencias div {{
                padding: 8px;
                cursor: pointer;
            }}
            #sugerencias div:hover {{
                background: #007acc;
                color: #fff;
            }}
            iframe {{
                flex-grow: 1;
                border: none;
                width: 100%;
                height: 100%;
            }}
            .enlace-sobre {{
                position: absolute;
                top: 15px;
                right: 15px;
                font-size: 14px;
                text-decoration: none;
                color: #007acc;
                background: #fff;
                padding: 5px 10px;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .enlace-IA {{
                position: absolute;
                top: 15px;
                left: 15px;
                font-size: 14px;
                text-decoration: none;
                color: #007acc;
                background: #fff;
                padding: 5px 10px;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <a href="sobre-IA-m.html" target="visor" class="enlace-sobre">Sobre IA_m...</a>
        <a href="IA_m_proceso.html" target="visor" class="enlace-IA">Actualmente aprendiendo sobre...</a>

        <div id="buscador-container">
            <input type="text" id="buscador" placeholder="🔍 Buscar nodo...">
            <div id="sugerencias"></div>
        </div>

        <iframe id="visor" name="visor" src="hipercubo_fractal_fluido.html"></iframe>

        <script>
            const nodos = {nodos_json};

            const buscador = document.getElementById('buscador');
            const sugerencias = document.getElementById('sugerencias');

            function aliasParaArchivo(nombre) {{
                // Versión JS equivalente a alias_para_archivo de Python
                const mapa = {{
                    "/": "slash",
                    "\\\\": "backslash",
                    "+": "plus",
                    "-": "minus",
                    "*": "asterisk",
                    ":": "colon",
                    "?": "question",
                    "<": "lt",
                    ">": "gt",
                    "|": "pipe",
                    '"': "quote"
                }};

                let n = nombre.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
                n = n.replace(/\\s+/g, "_").toLowerCase();

                let res = "";
                for (let c of n) {{
                    if (/[a-z0-9_-]/.test(c)) {{
                        res += c;
                    }} else if (mapa[c]) {{
                        res += mapa[c];
                    }}
                    // caracteres raros sin mapa se descartan
                }}
                return res;
            }}

            buscador.addEventListener('input', function() {{
                const texto = this.value.toLowerCase();
                sugerencias.innerHTML = '';
                if (!texto) {{
                    sugerencias.style.display = 'none';
                    return;
                }}
                const resultados = nodos.filter(n => n.toLowerCase().includes(texto));
                resultados.sort((a, b) => a.localeCompare(b, 'es'));

                if (resultados.length) {{
                    resultados.forEach(nodo => {{
                        const div = document.createElement('div');
                        div.textContent = nodo;
                        div.onclick = function() {{
                            cargarNodo(nodo);
                            sugerencias.style.display = 'none';
                            buscador.value = nodo;
                        }};
                        sugerencias.appendChild(div);
                    }});
                    sugerencias.style.display = 'block';
                }} else {{
                    sugerencias.style.display = 'none';
                }}
            }});

            document.addEventListener('click', function(e) {{
                if (!buscador.contains(e.target) && !sugerencias.contains(e.target)) {{
                    sugerencias.style.display = 'none';
                }}
            }});

            function cargarNodo(nodo) {{
                const safe = aliasParaArchivo(nodo);
                const archivo = "subgrafo_" + safe + ".html";
                document.getElementById('visor').src = archivo;
            }}
        </script>
    </body>
    </html>
    """

    ruta_index = os.path.join(carpeta, "index.html")
    with open(ruta_index, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Índice interactivo simplificado generado como '{ruta_index}'")


# 🔹 Normalizar términos
def normalizar_terminoANTGUO(termino):
    termino = re.sub(r"[^a-zA-Z0-9áéíóúüñ ]", "_", termino)  # Mantiene espacios
    return termino.lower().strip().replace(" ", "_")

# 🔹 Normalizar términos
def normalizar_termino(termino):
    """
    Normaliza un término para usarlo como id de nodo:
    - convierte "_" en espacio (para la entrada del usuario)
    - pasa a minúsculas
    - recorta espacios al inicio y al final
    - colapsa espacios múltiples en uno solo
    - deja espacios en lugar de "_"
    """
    if not isinstance(termino, str):
        termino = str(termino)

    # El usuario puede escribir "fondo_de_previsión" → lo tratamos como "fondo de previsión"
    termino = termino.replace("_", " ")
    termino = termino.replace(".", "")
    # Minúsculas y recorte
    termino = termino.lower().strip()

    # Un solo espacio entre palabras
    termino = re.sub(r"\s+", " ", termino)

    return termino

def es_falso_emergente(nombre):
    return nombre.count("emergente") > 1 and not any(x in nombre for x in ["equilibrio", "síntesis", "centro", "neutro"])

# 🔹 Nueva función para consultar ChatGPT
def consultar_chatgpt(tema, diccionario):
    tema = corregir_termino(tema, diccionario).lower()  # 🔹 Corregir y forzar a minúsculas

    try:
        #print(f"🤖 Consultando ChatGPT sobre: {tema}...")

        respuesta = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": (
                "Eres un profundo experto en matemáticas, geometría, cuántica, física y todas las ciencias."
                "Responde únicamente con una lista de términos técnicos separados por comas, sin fórmulas, sin puntos ni frases, solo términos conceptuales."
                "Debes responder conceptos que definan puramente el concepto que se te pide, esto es lo más importante, NO PUEDEN SER FRASES NI ORACIONES."
                "No incluyas números, ni textos ni frases introductorias o de cierre ni frases cortas. Sé empírico y profesional, NO INVENTES CONCEPTOS NUEVOS."
                "Tus contestaciones no deben ser de un contexto de economía, medicina, anatomía, informática, deportes ni ciencias sociales."
                "Si se te pregunta el término campo electromagnético, no puedes responder como correlación entre campo eléctrico y campo magnético, por ejemplo."
                "Prefiero pocos conceptos (que sean buenos) a que me respondas frases."
                "Si no entiendes el término, repite literalmente el término que se te ha dado, responde siempre en castellano (a no ser que el concepto no exista en castellano)."
                )},
                {"role": "user", "content": f"Dame conceptos clave, desde el punto de vista científico, que definan el concepto: {tema}."}
            ]
        )

        # Verifica que la respuesta no esté vacía
        if not respuesta.choices:
            print("⚠️ ChatGPT no devolvió resultados.")
            return []

        # Obtener la respuesta como texto
        texto_respuesta = respuesta.choices[0].message.content.strip().lower()  # 🔹 Convertir a minúsculas
        
        # Limpiar la respuesta eliminando caracteres no deseados
        conceptos = re.split(r',\s*', texto_respuesta)
        conceptos = [c.strip() for c in conceptos if c and es_nodo_relevante(c)]


        print("📝 ChatGPT sugiere los siguientes conceptos:")
        for concepto in conceptos:
            print(f"- {concepto}")

        return conceptos

    except Exception as e:
        print(f"❌ Error consultando ChatGPT: {e}")
        return []

####################################################################################################################
def obtener_embedding(texto, modelo=modelo_embeddings, cache=None):
    if cache is not None and texto in cache:
        return cache[texto]
    return modelo.encode(texto, convert_to_tensor=True)

####################################################################################################################

def es_variacion_morfologica(termino1, termino2, min_prefijo=2):
    """Detecta variantes morfológicas, evitando falsos positivos semánticos."""
    genericas = {"teoría", "campo", "principio", "concepto", "estructura", "modelo"}

    def normalizar(t):
        return re.sub(r"[_\\-]", " ", t.strip().lower())

    t1 = normalizar(termino1)
    t2 = normalizar(termino2)
    if t1 == t2:
        return True

    palabras1 = t1.split()
    palabras2 = t2.split()

    # 1. Si longitud absoluta es muy distinta, descartar
    if abs(len(palabras1) - len(palabras2)) > 2:
        return False

    # 2. Detectar prefijo común palabra a palabra
    prefijo_comun = []
    for p1, p2 in zip(palabras1, palabras2):
        if p1 == p2 and len(p1) > 2:
            prefijo_comun.append(p1)
        else:
            break

    # 3. Si el prefijo es corto y genérico, descartar
    if (len(prefijo_comun) < min_prefijo) or (set(prefijo_comun).issubset(genericas)):
        return False

    # 4. Comparar raíz morfológica (stemming)
    raiz1 = " ".join([stemmer.stem(p) for p in palabras1])
    raiz2 = " ".join([stemmer.stem(p) for p in palabras2])
    return raiz1 == raiz2

def calcular_prioridad_nodo(G, nodo):
    conexiones = G.degree(nodo)
    centralidad = nx.degree_centrality(G).get(nodo, 0)
    peso_promedio = sum([G.edges[nodo, neighbor].get('weight', 1) for neighbor in G.neighbors(nodo)]) / (conexiones or 1)
    historial = cargar_historial()
    exito = historial.get(nodo, {}).get("exito", 0)
    #prioridad = (conexiones * 0.3) + (centralidad * 0.3) + (peso_promedio * 0.2) + (exito * 0.2)
    prioridad = (conexiones * 0.3) + (centralidad * 0.3) + (peso_promedio * 0.2) - (exito * 0.2)
    return prioridad


def priorizar_expansion(G):
    """ Retorna una cola de nodos a expandir, ordenados por prioridad """
    prioridad_nodos = []
    
    for nodo in G.nodes():
        if G.degree(nodo) < 6:  # Solo prioriza nodos con pocas conexiones
            prioridad = calcular_prioridad_nodo(G, nodo)
            heapq.heappush(prioridad_nodos, (-prioridad, nodo))  # Heap inverso para prioridad alta

    return [heapq.heappop(prioridad_nodos)[1] for _ in range(len(prioridad_nodos))]
def generar_reportes():
    """ Genera visualizaciones cada cierto tiempo """
    print("📊 Generando reportes visuales de la red fractal...")
    visualizar_crecimiento_red()  # <-- REINTEGRADA AQUÍ
    visualizar_metodos_expansion()
    visualizar_distribucion_conexiones(G)
    plt.close()  # 🔧 Importante para liberar memoria
    
def registrar_expansion(nodo, nuevos_conceptos, metodo):
    """ Registra cada expansión realizada """
    registro = cargar_registro()
    nueva_entrada = {
        "nodo": nodo,
        "nuevos_conceptos": nuevos_conceptos,
        "metodo": metodo,
        "timestamp": time.time()
    }
    registro.append(nueva_entrada)
    guardar_registro(registro)

def ver_registro():
    """ Muestra el historial de expansiones """
    registro = cargar_registro()
    
    print("\n📜 Historial de Expansión:")
    for entrada in registro[-10:]:  # Solo mostrar las 10 últimas expansiones
        print(f"🔹 {entrada['nodo']} → {entrada['nuevos_conceptos']} (via {entrada['metodo']})")

def fusionar_grafos(G1, G2):
    """
    Fusiona dos grafos dirigidos sin sobrescribir nodos o aristas existentes.
    Preserva atributos de nodos y aristas si existen.
    """
    G_fusionado = nx.DiGraph()

    # Añadir nodos y aristas del primer grafo
    G_fusionado.add_nodes_from(G1.nodes(data=True))
    G_fusionado.add_edges_from(G1.edges(data=True))

    # Añadir nodos y aristas del segundo grafo si no están
    for nodo, atributos in G2.nodes(data=True):
        if nodo not in G_fusionado:
            G_fusionado.add_node(nodo, **atributos)

    for u, v, datos in G2.edges(data=True):
        if not G_fusionado.has_edge(u, v):
            G_fusionado.add_edge(u, v, **datos)

    return G_fusionado

# 🔹 Agregar llamadas en puntos clave
# ACTUALMENTE NO SE USA
def expansion_prioritaria(G, diccionario, usar_gpt):
    """ Expande los nodos más prioritarios en la red """
    nodos_a_expandir = priorizar_expansion(G)[:10]  
    for nodo in nodos_a_expandir:
        print(f"🔍 Expansión prioritaria en: {nodo}")
        nuevos_conceptos = []
        if usar_gpt == "s":
            nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
            if nuevos_conceptos:
                #G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
                G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos, nodo_origen=nodo)
                registrar_expansion(nodo, nuevos_conceptos, "GPT-4")
                guardar_diccionario(diccionario)
        # Siempre expandir con embeddings
        expandir_concepto_embedding(nodo, G, diccionario)
        registrar_expansion(nodo, [], "Embeddings")
    
    # 🔺 Detectar estructuras emergentes adaptativas
    nuevos_emergentes = detectar_estructura_emergente_adaptativa(G)

    print(f"✨ Detectadas {len(nuevos_emergentes)} nuevas estructuras emergentes en expansión prioritaria.")
    
    guardar_red(G)
    return G

# 🔹 Cargar red desde JSON o crear nueva con interacción
def cargar_red():
    diccionario = cargar_diccionario()
    try:
        with open("json/red_fractal.json", "r") as f:
            data = json.load(f)
            G = nx.node_link_graph(data, edges="links")
            print("✅ Red fractal cargada correctamente.")
            return G, diccionario

    except FileNotFoundError:
        print("🚀 No se encontró una red previa. Vamos a crear una nueva.")
        G = nx.DiGraph()

        global NODO_CENTRAL
        if not NODO_CENTRAL:
            definir_nodo_central = "n"
            #definir_nodo_central = input("¿Deseas definir un nodo central? (s/n): ").lower().strip()
            if definir_nodo_central == "s":
                NODO_CENTRAL = input("Introduce el nombre del nodo central: ").strip()
                config["nodo_central"] = NODO_CENTRAL
                guardar_config(config)

        usar_dualidades = input("¿Deseas usar las dualidades predefinidas? (s/n): ").lower().strip()
        if usar_dualidades == "s":
            dualidades = dualidades_base
        else:
            dualidades = {}
            print("Introduce pares de dualidades (ejemplo: luz/oscuridad), deja vacío para terminar:")
            while True:
                par = input("Dualidad (formato A/B): ").strip()
                if not par:
                    break
                if "/" in par:
                    a, b = map(str.strip, par.split("/", 1))
                    dualidades[a] = b

#        for a, b in dualidades.items():
#            for nodo in (a, b):
#                if nodo not in G:
#                    G.add_node(nodo)
#            G.add_edge(a, b, weight=2.0)
#            G.add_edge(b, a, weight=2.0)

#            if NODO_CENTRAL:
#                if not G.has_node(NODO_CENTRAL):
#                    G.add_node(NODO_CENTRAL)
#                G.add_edge(a, NODO_CENTRAL, weight=1.5)
#                G.add_edge(b, NODO_CENTRAL, weight=1.5)
        for a, b in dualidades.items():
            # 🔹 Crear la dualidad usando la función estándar
            agregar_dualidad(G, a, b)

            # 🔹 Conectar cada extremo al nodo central (si existe)
            if NODO_CENTRAL:
                if not G.has_node(NODO_CENTRAL):
                    G.add_node(NODO_CENTRAL)

                if not G.has_edge(a, NODO_CENTRAL):
                    G.add_edge(a, NODO_CENTRAL, weight=1.5)
                if not G.has_edge(NODO_CENTRAL, a):
                    G.add_edge(NODO_CENTRAL, a, weight=1.0)

                if not G.has_edge(b, NODO_CENTRAL):
                    G.add_edge(b, NODO_CENTRAL, weight=1.5)
                if not G.has_edge(NODO_CENTRAL, b):
                    G.add_edge(NODO_CENTRAL, b, weight=1.0)

        semilla = list(diccionario.keys())
        for i in range(len(semilla)):
            for j in range(i + 1, len(semilla)):
                G.add_edge(semilla[i], semilla[j], weight=1.2)
        # 🌡️ Sembrar la triadas base en la creación inicial de la red
        G = sembrar_triadas_base(G, diccionario)
        G = ajustar_espacio_tiempo_realidad(G)

        guardar_red(G)
        return G, diccionario

def asignar_niveles_por_defecto(G):
    for nodo in G.nodes():
        if "nivel_conceptual" not in G.nodes[nodo]:
            tipo = G.nodes[nodo].get("tipo", "")
            if tipo == "dualidad":
                G.nodes[nodo]["nivel_conceptual"] = 1 # Detecta su par opuesto y complementario
            elif tipo == "equilibrio":
                G.nodes[nodo]["nivel_conceptual"] = 2 # Detecta el punto central de equilibrio entre opuestos
            elif tipo == "emergente":
                G.nodes[nodo]["nivel_conceptual"] = 3 # Detecta su síntesis
            elif tipo == "abstracto":
                G.nodes[nodo]["nivel_conceptual"] = 4
            else:
                G.nodes[nodo]["nivel_conceptual"] = 0  # por defecto, concepto base
    return G

# 🔹 Consultar Wikipedia y detectar términos opuestos
# ACTUALMENTE NO SE USA
def consultar_wikipedia(concepto, G, diccionario):
    try:
        concepto = corregir_termino(concepto, diccionario)
        consulta = normalizar_termino(concepto)
        print(f"🌍 Consultando Wikipedia para: {consulta}...")
        wiki = wikipediaapi.Wikipedia(language='es', user_agent="IAConsciente/1.0")
        page = wiki.page(consulta)
        if not page.exists():
            print(f"❌ Wikipedia: No se encontró información sobre {consulta}.")
            return "❌ No se encontró información en Wikipedia."

        resumen = page.summary[:500]
        print(f"📖 Wikipedia: {consulta} encontrada.")
        print(f"📜 Resumen: {resumen}...")

        # 🔹 Extraer y normalizar primeros 10 enlaces relevantes
        enlaces_relacionados = [
            normalizar_termino(link.strip())
            for link in list(page.links)[:10]
            if es_nodo_relevante(link)
        ]

        # Mostrar sugerencias
        if enlaces_relacionados:
            print("🔗 Wikipedia sugiere los siguientes conceptos:")
            for termino in enlaces_relacionados:
                print(f"- {termino}")

        for termino in enlaces_relacionados:
            termino = corregir_termino(termino, diccionario)
            if not es_nodo_relevante(termino):
                continue

            if termino not in G.nodes():
                G.add_node(termino)
                diccionario.setdefault(concepto, []).append(termino)
                G.add_edge(concepto, termino, weight=1.2)

            # Detectar y conectar dualidades con equilibrio
            dualidad_opuesta = detectar_dualidad(termino, G, concepto)
            if dualidad_opuesta and dualidad_opuesta in G.nodes():
                conectar_dualidad_con_equilibrio(termino, dualidad_opuesta, G)

        guardar_red(G)
        return f"📖 Wikipedia: {resumen}...\n🔗 Más info: {page.fullurl}"

    except Exception as e:
        print(f"❌ Error al consultar Wikipedia: {e}")
        return "❌ Error al consultar Wikipedia (sin conexión o fallo temporal)."
    
def conectar_dualidad_con_equilibrioANTIGUO(concepto, dualidad, G):
    """Conecta dos nodos como dualidades y añade nodos de equilibrio y superior si corresponde."""

    # 🔴➡️🟢 Conexión de dualidad con colores bidireccionales
    if not G.has_edge(concepto, dualidad):
        G.add_edge(concepto, dualidad, tipo="dualidad", color="red", weight=2.5, direccion="negativo_a_positivo")
        G.add_edge(dualidad, concepto, tipo="dualidad", color="green", weight=2.5, direccion="positivo_a_negativo")
        print(f"🔗 Conectando {concepto} (🔴) → {dualidad} (🟢) como dualidad.")

    # 🔹 Nodo superior (nivel 3)
    posibles_superiores = detectar_nodo_superior(concepto, dualidad, G)
    if posibles_superiores:
        nodo_superior = posibles_superiores[0]

        if nodo_superior not in G.nodes():
            G.add_node(nodo_superior, tipo="nodo_superior", es_sintesis=True, nivel_conceptual=3)
            print(f"🆕 Añadiendo nodo superior dinámico: {nodo_superior}")
        else:
            G.nodes[nodo_superior]["es_sintesis"] = True
            G.nodes[nodo_superior].setdefault("nivel_conceptual", 3)
            if "nodo_superior" not in G.nodes[nodo_superior].get("tipo", ""):
                G.nodes[nodo_superior]["tipo"] = "nodo_superior"

        G.add_edge(nodo_superior, concepto, weight=1.5)
        G.add_edge(nodo_superior, dualidad, weight=1.5)
        print(f"🔗 Vinculando {concepto} y {dualidad} a {nodo_superior}")

    # 🔹 Nodo equilibrio (nivel 2)
    nodo_equilibrio = detectar_nodo_equilibrio(concepto, dualidad, G)
    if nodo_equilibrio:
        if nodo_equilibrio not in G.nodes():
            G.add_node(nodo_equilibrio, tipo="equilibrio", es_sintesis=True, nivel_conceptual=2)
            print(f"🆕 Añadiendo nodo de equilibrio: {nodo_equilibrio}")
        else:
            G.nodes[nodo_equilibrio]["es_sintesis"] = True
            G.nodes[nodo_equilibrio].setdefault("nivel_conceptual", 2)
            if "equilibrio" not in G.nodes[nodo_equilibrio].get("tipo", ""):
                G.nodes[nodo_equilibrio]["tipo"] = "equilibrio"

        G.add_edge(nodo_equilibrio, concepto, weight=1.8)
        G.add_edge(nodo_equilibrio, dualidad, weight=1.8)
        print(f"⚖️ Estableciendo equilibrio entre {concepto} y {dualidad} en {nodo_equilibrio}")

def conectar_dualidad_con_equilibrio(concepto, dualidad, G):
    """
    Versión simplificada:
    - SOLO crea la dualidad entre concepto y dualidad usando agregar_dualidad.
    - NO crea ni modifica nodos de equilibrio ni nodos superiores.
    Esto evita que se llenen de 'equilibrio' nodos que no lo son.
    """
    agregar_dualidad(G, concepto, dualidad)
    # 💤 De momento NO hacemos nada más aquí.
    # La creación de equilibrios se hará en una fase aparte,
    # basada en triángulos geométricos bien definidos.

    
def obtener_embeddings_lista(textos, modelo):
    resultados = []
    nuevos_textos = []
    indices_a_reemplazar = []

    for i, texto in enumerate(textos):
        texto = texto.lower().strip()
        if texto in embeddings_cache:
            resultados.append(embeddings_cache[texto])
        else:
            nuevos_textos.append(texto)
            indices_a_reemplazar.append(i)
            resultados.append(None)  # Placeholder

    if nuevos_textos:
        nuevos_embeddings = modelo.encode(nuevos_textos, convert_to_tensor=True)
        for j, (idx, texto) in enumerate(zip(indices_a_reemplazar, nuevos_textos)):
            emb = nuevos_embeddings[j]
            embeddings_cache[texto] = emb
            resultados[idx] = emb
        guardar_cache_embeddings(embeddings_cache)

    return torch.stack(resultados)

def detectar_nodo_equilibrio(concepto, dualidad, G):
    """ Detecta un nodo de equilibrio dinámico en la red usando embeddings """
    if concepto not in G or dualidad not in G:
        return None

    embedding_concepto = obtener_embedding(concepto, modelo_embeddings)
    embedding_dualidad = obtener_embedding(dualidad, modelo_embeddings)
    embedding_promedio = (embedding_concepto + embedding_dualidad) / 2

    nodos_existentes = list(G.nodes())
    embeddings_red = obtener_embeddings_lista(nodos_existentes, modelo_embeddings)
    similitudes = util.pytorch_cos_sim(embedding_promedio, embeddings_red)[0]

    # Elegimos el nodo más cercano que no sea el propio concepto o dualidad
    candidatos = [
        (nodo, similitud.item()) for nodo, similitud in zip(nodos_existentes, similitudes)
        if nodo not in (concepto, dualidad)
    ]
    candidatos_ordenados = sorted(candidatos, key=lambda x: x[1], reverse=True)

    if candidatos_ordenados:
        return candidatos_ordenados[0][0]  # El nodo más similar
    return None

# 🔹 Filtrar nodos irrelevantes
def es_nodo_relevante(nodo):
    irrelevantes = [
        "art", "architecture", "thesaurus", "archive", "research", "manual", "RAE", "desambiguación", "control de autoridades", "biblioteca", "Wikidata", "Library", "University"
    ]
    return not any(term in nodo.lower() for term in irrelevantes)

# 🔹 Función para extraer la raíz de un concepto
def obtener_raiz(termino):
    """ Extrae la raíz de un término eliminando sufijos comunes. """
    return re.sub(r"_(táctil|auditiva|sensorial|visual|espacial|profunda|sinestésica|expectante|intermodal|intrínseca)$", "", termino)
    
def es_expandible(nodo, G):
    """Determina si un nodo puede ser expandido según su tipo y nivel conceptual"""
    tipo = G.nodes[nodo].get("tipo", "")
    nivel = G.nodes[nodo].get("nivel_conceptual", 0)
    
    if tipo == "emergente":
        return False  # 🔒 No expandir emergentes directamente
    if nivel >= 3:
        return False  # 🔒 Limita profundidad conceptual (opcional)
    return True


# 🔹 Guardar diccionario en un archivo JSON
def guardar_diccionario(diccionario):
    with open("json/diccionario.json", "w", encoding="utf-8") as f:
        json.dump(diccionario, f, ensure_ascii=False, indent=4)
       
# ACTUALMENTE NO SE USA
def expansion_con_embeddings(G, diccionario):
    """ Expande automáticamente la red usando embeddings """
    nodos_a_expandir = [nodo for nodo in G.nodes() if G.degree(nodo) < 2]

    for nodo in nodos_a_expandir:
        print(f"🧠 Expansión semántica para: {nodo}")
        expandir_concepto_embedding(nodo, G, diccionario)

    guardar_red(G)

MAX_EXPANSIONES = 10  # Límite de expansiones por iteración
UMBRAL_SIMILITUD = 0.75  # Solo agrega términos con alta similitud
# ACTUALMENTE NO SE USA
def expansion_controlada(G, diccionario):
    """ Controla la expansión automática evitando términos irrelevantes """
    nodos_a_expandir = [nodo for nodo in G.nodes() if G.degree(nodo) < 2]
    
    for i, nodo in enumerate(nodos_a_expandir):
        if i >= MAX_EXPANSIONES:
            break  # Detener expansión si se alcanza el límite

        print(f"🛠 Expandiendo nodo controlado: {nodo}")
        conceptos = consultar_chatgpt(nodo, diccionario)
        embedding_nodo = obtener_embedding(nodo, modelo)

        conceptos_filtrados = []
        for c in conceptos:
            embedding_c = obtener_embedding(c, modelo)
            similitud = util.pytorch_cos_sim(embedding_c, embedding_nodo)[0].item()
        if similitud > UMBRAL_SIMILITUD:
            conceptos_filtrados.append(c)

        if conceptos_filtrados:
            G = agregar_nuevo_nodo(G, diccionario, conceptos_filtrados)

    guardar_red(G)

def detectar_dualidad_para_nodoANTIGUO(nodo, G, modelo=modelo_embeddings, umbral=(0.78, 0.84)):
    """
    Detecta una dualidad para un nodo específico usando:
    - WordNet (antónimos)
    - Similitud semántica por embeddings en una banda (umbral[0], umbral[1])
    - Memoria adaptativa (confirmadas/rechazadas)
    - Evita dualidades morfológicas (singular/plural, variaciones mínimas)
    """
    nodo = nodo.lower().strip()

    if nodo not in G:
        return None

    # Si ya está súper conectado, asumimos que no buscamos nueva dualidad
    if G.degree(nodo) > 10:
        return None

    confirmadas = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_CONFIRMADAS)
    rechazadas = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_RECHAZADAS)

    # Si ya tiene una pareja confirmada, no buscamos más
    for par in confirmadas:
        if nodo in par:
            return None

    # Candidatos: nodos "normales" (sin tipo o tipo "concepto")
    candidatos = [
        n for n in G.nodes()
        if n != nodo and G.nodes[n].get("tipo") in (None, "concepto")
    ]

    # Antónimos directos según WordNet
    antonimos = set(detectar_dualidad_wordnet(nodo))

    emb_nodo = obtener_embedding(nodo, modelo)
    mejor_dual = None
    mejor_sim = 0.0

    for candidato in candidatos:
        par = tuple(sorted([nodo, candidato]))

        # Evitar pares ya procesados
        if par in confirmadas or par in rechazadas:
            continue

        # Evitar aristas ya existentes de dualidad
        if G.has_edge(nodo, candidato) or G.has_edge(candidato, nodo):
            # Si ya hay relación, no proponemos nueva dualidad
            continue

        # Evitar variaciones morfológicas
        if es_variacion_morfologica(nodo, candidato):
            # Comentario para depuración si quieres:
            # print(f"🚫 Variación morfológica ignorada: '{nodo}' ↔ '{candidato}'")
            continue

        # Antónimos directos (prioridad máxima)
        if candidato in antonimos:
            registrar_dualidad_confirmada(nodo, candidato)
            print(f"🔍 Dualidad por antónimo directo: '{nodo}' ↔ '{candidato}'")
            return candidato

        # Embeddings: buscamos en una banda (no demasiado bajo ni demasiado alto)
        emb_c = obtener_embedding(candidato, modelo)
        sim = util.pytorch_cos_sim(emb_nodo, emb_c).item()

        if umbral[0] < sim < umbral[1] and sim > mejor_sim:
            mejor_dual = candidato
            mejor_sim = sim

    if mejor_dual:
        registrar_dualidad_confirmada(nodo, mejor_dual)
        print(f"🔁 Dualidad detectada para '{nodo}' ↔ '{mejor_dual}' (sim={mejor_sim:.2f})")
        return mejor_dual

    # No hemos encontrado dualidad sensata
    registrar_dualidad_rechazada(nodo, "_")
    return None
    
    
def detectar_dualidad_para_nodo(nodo, G):
    """
    Detecta una dualidad para un nodo específico usando SOLO:
    - dualidades base
    - antónimos de WordNet
    - memoria adaptativa (confirmadas/rechazadas)
    ❌ NO usa embeddings aquí (eso se reserva para detectar_nuevas_dualidades).
    """
    if nodo is None:
        return None

    nodo = normalizar_termino(nodo)

    # Si no está en el grafo o es meta, nada
    if nodo not in G or nodo in NODOS_META:
        return None

    # Si ya está hipercargado de conexiones, no insistimos
    if G.degree(nodo) > 10:
        return None

    # Cargar memoria
    ya_confirmadas = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_CONFIRMADAS)
    ya_rechazadas = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_RECHAZADAS)

    # Si ya tiene pareja confirmada, no buscamos más
    for par in ya_confirmadas:
        if nodo in par:
            return None

    # 1️⃣ Dualidades base (si por alguna razón nos llaman con un nodo base)
    if nodo in dualidades_base:
        dual = normalizar_termino(dualidades_base[nodo])
        if dual == nodo or dual in NODOS_META:
            return None
        registrar_dualidad_confirmada(nodo, dual)
        return dual

    # 2️⃣ Antónimos de WordNet (en tu función detectar_dualidad_wordnet)
    antonimos = set(detectar_dualidad_wordnet(nodo))

    # Candidatos: solo nodos ya presentes y no meta
    conceptos = [
        n for n in G.nodes()
        if n != nodo and n not in NODOS_META
    ]

    mejor_dual = None
    for candidato in conceptos:
        par = tuple(sorted([nodo, candidato]))

        # Evitar pares ya procesados
        if par in ya_confirmadas or par in ya_rechazadas:
            continue

        # Evitar variaciones morfológicas (singular/plural, espacios, etc.)
        if es_variacion_morfologica(nodo, candidato):
            registrar_dualidad_rechazada(nodo, candidato)
            continue

        # Si el candidato aparece como antónimo directo, lo aceptamos
        if candidato in antonimos:
            mejor_dual = candidato
            break

    if mejor_dual:
        registrar_dualidad_confirmada(nodo, mejor_dual)
        print(f"🔍 Dualidad por antónimo directo: '{nodo}' ↔ '{mejor_dual}'")
        return mejor_dual

    # Nada encontrado -> marcamos como rechazado para no repetir
    registrar_dualidad_rechazada(nodo, "_")
    return None


# 🔹 Detectar dualidad con WordNet
def detectar_dualidad_wordnet(termino):
    if not WORDNET_AVAILABLE:
        return []  # sin WordNet no sugerimos nada nuevo
    antonimos = set()
    try:
        synsets = wn.synsets(termino, lang='spa')
        if not synsets:
            return []  # Retorna lista vacía si no hay resultados

        for syn in synsets:
            for lemma in syn.lemmas('spa'):
                for ant in lemma.antonyms():
                    antonimos.add(ant.name())

    except Exception as e:
        print(f"⚠️ Error consultando WordNet para '{termino}': {e}")
    
    return list(antonimos)
    
def agregar_dualidad(G, a, b):
    """
    Añade una dualidad entre dos conceptos, sin crear automáticamente un nodo de equilibrio.
    Solo crea las aristas de dualidad y marca los nodos como dualidad (nivel 1).
    """
    a = a.lower().strip()
    b = b.lower().strip()

    # Evita dualidades triviales
    if a == b:
        print(f"🚫 Dualidad ignorada: '{a}' ↔ '{b}' (son el mismo nodo)")
        return

    # Asegurar presencia de nodos
    for nodo in (a, b):
        if nodo not in G:
            G.add_node(nodo, tipo="concepto", nivel_conceptual=0)

    # Crear aristas de dualidad si no existen
    if not G.has_edge(a, b):
        G.add_edge(a, b, tipo="dualidad", color="red", weight=2.5, direccion="negativo_a_positivo") #peso 2.5
    if not G.has_edge(b, a):
        G.add_edge(b, a, tipo="dualidad", color="green", weight=2.5, direccion="positivo_a_negativo") # peso 2.5

    # Marcar nodos como dualidad (nivel 1)
    for nodo in (a, b):
        tipo_actual = G.nodes[nodo].get("tipo")
        if tipo_actual in (None, "concepto"):
            G.nodes[nodo]["tipo"] = "dualidad"
            G.nodes[nodo]["nivel_conceptual"] = 1

    print(f"🧬 Dualidad añadida: {a} ↔ {b}")

def sembrar_triada_generica(G, a, b, equilibrio, sintesis=None, diccionario=None):
    """
    Siembra una triada estructural del tipo:
      a ↔ b      (dualidad, nivel 1)
      equilibrio (equilibrio, nivel 2)
      síntesis   (síntesis/emergente, nivel 3, opcional)

    Ejemplos de uso:
      - frío/calor-templado → temperatura
      - tesis/antítesis-síntesis (sin nodo superior explícito)
      - masa/aceleración-fuerza  (sin nodo superior explícito)
    """

    def elegir_nombre(*variantes):
        """
        Elige la variante que ya exista en el grafo si es posible;
        si no, devuelve la primera normalizada.
        """
        for v in variantes:
            n = normalizar_termino(v)
            if n in G:
                return n
        return normalizar_termino(variantes[0])

    # 🔹 Normalizar / reutilizar nombres
    a_n = elegir_nombre(a)
    b_n = elegir_nombre(b)
    eq_n = elegir_nombre(equilibrio)
    sint_n = normalizar_termino(sintesis) if sintesis else None

    # 🔹 Asegurar nodos
    for nodo in (a_n, b_n, eq_n) + ((sint_n,) if sint_n else ()):
        if nodo not in G:
            G.add_node(nodo, tipo="concepto", nivel_conceptual=0)

    # 1) Dualidad a/b
    agregar_dualidad(G, a_n, b_n)  # ya marca tipo="dualidad", nivel 1 y colores rojo/verde

    # 2) Nodo de equilibrio
    G.nodes[eq_n]["tipo"] = "equilibrio"
    G.nodes[eq_n]["nivel_conceptual"] = 2
    G.nodes[eq_n]["es_sintesis"] = True

    for extremo in (a_n, b_n):
        # equilibrio → extremo
        if not G.has_edge(eq_n, extremo):
            G.add_edge(eq_n, extremo,
                       tipo="equilibrio", color="goldenrod", weight=1.8)
        # extremo → equilibrio
        if not G.has_edge(extremo, eq_n):
            G.add_edge(extremo, eq_n,
                       tipo="equilibrio", color="goldenrod", weight=1.2)

    # 3) Nodo de síntesis superior (opcional)
    if sint_n:
        G.nodes[sint_n]["tipo"] = "sintesis"
        G.nodes[sint_n]["nivel_conceptual"] = 3
        G.nodes[sint_n]["es_sintesis"] = True

        for nodo in (a_n, eq_n, b_n):
            if not G.has_edge(sint_n, nodo):
                G.add_edge(sint_n, nodo,
                           tipo="sintesis", color="gold", weight=2.0)
            if not G.has_edge(nodo, sint_n):
                G.add_edge(nodo, sint_n,
                           tipo="sintesis", color="gold", weight=1.5)

    # 4) Actualizar diccionario (si existe)
    if diccionario is not None:
        relacionados_eq = [a_n, b_n]
        relacionados_a = [b_n, eq_n]
        relacionados_b = [a_n, eq_n]

        if sint_n:
            relacionados_eq.append(sint_n)
            relacionados_a.append(sint_n)
            relacionados_b.append(sint_n)
            relacionados_sint = [a_n, eq_n, b_n]
        else:
            relacionados_sint = []

        relaciones = {
            eq_n: relacionados_eq,
            a_n: relacionados_a,
            b_n: relacionados_b,
        }
        if sint_n:
            relaciones[sint_n] = relacionados_sint

        for clave, valores in relaciones.items():
            diccionario.setdefault(clave, [])
            for v in valores:
                if v not in diccionario[clave]:
                    diccionario[clave].append(v)

    nombre_sint = f" → {sint_n}" if sint_n else ""
    print(f"🔺 Triada sembrada: {a_n} – {equilibrio} – {b_n}{nombre_sint}")
    return G

def sembrar_triadas_base(G, diccionario=None):
    """
    Recorre TRIADAS_BASE_DEFECTO y siembra cada triada en la red.
    No duplica estructuras ya existentes porque respeta nodos y aristas previas.
    """
    for t in TRIADAS_BASE_DEFECTO:
        a = t["a"]
        b = t["b"]
        equilibrio = t["equilibrio"]
        sintesis = t.get("sintesis")
        G = sembrar_triada_generica(G, a, b, equilibrio, sintesis, diccionario)
    print(f"🌱 Triadas base sembradas: {len(TRIADAS_BASE_DEFECTO)}")
    return G


def sembrar_triada_termicaANTIGUO(G, diccionario=None):
    """
    Siembra explícitamente la triada térmica:
      frío ↔ calor  (dualidad, nivel 1)
      templado      (equilibrio, nivel 2)
      temperatura   (síntesis/emergente, nivel 3)
    y crea las aristas geométricas correspondientes.
    """

    def elegir_nombre(*variantes):
        """
        Elige la variante que ya exista en el grafo si es posible;
        si no, devuelve la primera normalizada.
        """
        for v in variantes:
            n = normalizar_termino(v)
            if n in G:
                return n
        return normalizar_termino(variantes[0])

    # 🔹 Elegimos nombres reutilizando los que ya existan
    frio = elegir_nombre("frío", "frio")
    calor = elegir_nombre("calor")
    templado = elegir_nombre("templado")
    temperatura = elegir_nombre("temperatura")

    # 🔹 Asegurar que los nodos existen
    for nodo in (frio, calor, templado, temperatura):
        if nodo not in G:
            G.add_node(nodo, tipo="concepto", nivel_conceptual=0)

    # 1) frío / calor como dualidad (eje térmico)
    agregar_dualidad(G, frio, calor)  # marca tipo="dualidad", nivel=1 y aristas de dualidad

    # 2) templado como equilibrio entre frío y calor
    G.nodes[templado]["tipo"] = "equilibrio"
    G.nodes[templado]["nivel_conceptual"] = 2
    G.nodes[templado]["es_sintesis"] = True

    for extremo in (frio, calor):
        # equilibrio → extremo
        if not G.has_edge(templado, extremo):
            G.add_edge(templado, extremo,
                       tipo="equilibrio", color="goldenrod", weight=1.8)
        # extremo → equilibrio (opcional, para cerrar estructura)
        if not G.has_edge(extremo, templado):
            G.add_edge(extremo, templado,
                       tipo="equilibrio", color="goldenrod", weight=1.2)

    # 3) temperatura como síntesis superior de la triada
    G.nodes[temperatura]["tipo"] = "sintesis"
    G.nodes[temperatura]["nivel_conceptual"] = 3
    G.nodes[temperatura]["es_sintesis"] = True

    # Conexiones desde temperatura hacia los 3 nodos de la triada
    for nodo in (frio, templado, calor):
        if not G.has_edge(temperatura, nodo):
            G.add_edge(temperatura, nodo,
                       tipo="sintesis", color="gold", weight=2.0)
        if not G.has_edge(nodo, temperatura):
            G.add_edge(nodo, temperatura,
                       tipo="sintesis", color="gold", weight=1.5)

    # 4) Actualizar diccionario (opcional pero útil para coherencia interna)
    if diccionario is not None:
        relaciones = {
            temperatura: [frio, templado, calor],
            templado: [frio, calor, temperatura],
            frio: [calor, templado, temperatura],
            calor: [frio, templado, temperatura],
        }
        for clave, valores in relaciones.items():
            diccionario.setdefault(clave, [])
            for v in valores:
                if v not in diccionario[clave]:
                    diccionario[clave].append(v)

    print("❄️🔥 Triada térmica sembrada: frío – templado – calor → temperatura")
    return G

def ajustar_espacio_tiempo_realidad(G):
    # asegurar nodos
    for nodo in ["espacio", "tiempo", "realidad", "presente"]:
        if nodo not in G:
            G.add_node(nodo, tipo="concepto", nivel_conceptual=0)

    # 1) Presente como equilibrio entre pasado y futuro
    G.nodes["presente"]["tipo"] = "equilibrio"
    G.nodes["presente"]["nivel_conceptual"] = 2
    G.nodes["presente"]["es_sintesis"] = True
    for extremo in ("pasado", "futuro"):
        if G.has_node(extremo):
            if not G.has_edge("presente", extremo):
                G.add_edge("presente", extremo,
                           tipo="equilibrio", color="goldenrod", weight=1.8)
            if not G.has_edge(extremo, "presente"):
                G.add_edge(extremo, "presente",
                           tipo="equilibrio", color="goldenrod", weight=1.2)

    # 2) Tiempo como síntesis de pasado–presente–futuro
    G.nodes["tiempo"]["tipo"] = "sintesis"
    G.nodes["tiempo"]["nivel_conceptual"] = 3
    G.nodes["tiempo"]["es_sintesis"] = True
    for nodo in ("pasado", "presente", "futuro"):
        if G.has_node(nodo):
            if not G.has_edge("tiempo", nodo):
                G.add_edge("tiempo", nodo,
                           tipo="sintesis", color="gold", weight=2.0)
            if not G.has_edge(nodo, "tiempo"):
                G.add_edge(nodo, "tiempo",
                           tipo="sintesis", color="gold", weight=1.5)

    # 3) ESPACIO: 3 dualidades espaciales → centro común → síntesis espacio
    extremos_espaciales = ["arriba", "abajo", "izquierda", "derecha", "delante", "detrás"]
    # 🔹 Nodo de equilibrio común: centro focal
    centro_focal = "centro focal"
    if centro_focal not in G:
        G.add_node(
            centro_focal,
            tipo="equilibrio",
            nivel_conceptual=2,
            es_sintesis=True
        )

    # Conectamos cada extremo espacial con el centro focal como equilibrio
    for extremo in extremos_espaciales:
        if G.has_node(extremo):
            if not G.has_edge(centro_focal, extremo):
                G.add_edge(
                    centro_focal,
                    extremo,
                    tipo="equilibrio",
                    color="goldenrod",
                    weight=1.8
                )
            if not G.has_edge(extremo, centro_focal):
                G.add_edge(
                    extremo,
                    centro_focal,
                    tipo="equilibrio",
                    color="goldenrod",
                    weight=1.2
                )

    # 🔹 Espacio como síntesis de las 3 dualidades + su centro (cubo completo)
    G.nodes["espacio"]["tipo"] = "sintesis"
    G.nodes["espacio"]["nivel_conceptual"] = 3
    G.nodes["espacio"]["es_sintesis"] = True

    for nodo in extremos_espaciales + [centro_focal]:
        if G.has_node(nodo):
            if not G.has_edge("espacio", nodo):
                G.add_edge(
                    "espacio",
                    nodo,
                    tipo="sintesis",
                    color="gold",
                    weight=2.0
                )
            if not G.has_edge(nodo, "espacio"):
                G.add_edge(
                    nodo,
                    "espacio",
                    tipo="sintesis",
                    color="gold",
                    weight=1.5
                )

    # 4) Realidad como emergente de espacio y tiempo
    G.nodes["realidad"]["tipo"] = "emergente"
    G.nodes["realidad"]["nivel_conceptual"] = 4
    G.nodes["realidad"]["es_sintesis"] = True
    for base in ("espacio", "tiempo"):
        if G.has_node(base):
            if not G.has_edge(base, "realidad"):
                G.add_edge(base, "realidad",
                           tipo="emergente", color="#CC99FF", weight=2.2)
            if not G.has_edge("realidad", base):
                G.add_edge("realidad", base,
                           tipo="emergente", color="#CC99FF", weight=1.8)

    return G

# 🔹 Corregir posibles errores en la entrada
def corregir_termino(termino, diccionario):
    if termino in diccionario:
        return termino  # Ya está bien escrito
    sugerencias = get_close_matches(termino, diccionario.keys(), n=1, cutoff=0.8)
    return sugerencias[0] if sugerencias else termino

# 🔹 Detectar dualidad con embeddings
def detectar_dualidad_embeddings(nuevo_concepto, G, top_n=5):
    palabras_red = list(G.nodes())

    embedding_concepto = obtener_embedding(nuevo_concepto, modelo)
    embeddings_red = obtener_embeddings_lista(palabras_red, modelo)

    similitudes = util.pytorch_cos_sim(embedding_concepto, embeddings_red)[0]
    indices_top = similitudes.argsort(descending=True)[:top_n].tolist()
    
    relacionados = [palabras_red[i] for i in indices_top if palabras_red[i] != nuevo_concepto]
    return relacionados

# 🔹 Detectar nuevas dualidades automáticamente en la red
def detectar_nuevas_dualidadesANTIGUO(G, max_nuevas=10, umbral_similitud=0.85):
    print("🔄 Detectando nuevas dualidades optimizadas...")

    nuevas_dualidades = {}

    # 🔹 Dualidades por estructura (emergentes desde equilibrio común)
    for nodo_eq in [n for n, d in G.nodes(data=True) if d.get("tipo") == "equilibrio" and d.get("es_sintesis")]:
        fuentes = list(G.predecessors(nodo_eq))
        if len(fuentes) == 2:
            a, b = fuentes
            if not G.has_edge(a, b) and not G.has_edge(b, a):
                if a not in dualidades_base and b not in dualidades_base:
                    print(f"📐 Dualidad estructural detectada: {a} ↔ {b} (equilibrio común: {nodo_eq})")
                    nuevas_dualidades[a] = b

    # 🔹 Dualidades semánticas por embeddings
    nodos_lista = list(G.nodes())[-max_nuevas:]  # Solo analiza los últimos nodos agregados
    for nodo in nodos_lista:
        similitudes_nodo = detectar_dualidad_embeddings(nodo, G)

        for otro in similitudes_nodo:
            if otro in G.nodes() and nodo != otro and not G.has_edge(nodo, otro):
                similitud_n2 = detectar_dualidad_embeddings(otro, G)
                if nodo in similitud_n2 and otro in similitudes_nodo:
                    if nodo not in dualidades_base and otro not in dualidades_base:
                        print(f"🧠 Dualidad semántica detectada: {nodo} ↔ {otro}")
                        nuevas_dualidades[nodo] = otro

    # 🔹 Agregar nuevas dualidades a la red
    for nodo, dual in nuevas_dualidades.items():
        dualidades_base[nodo] = dual
        dualidades_base[dual] = nodo
        conectar_dualidad_con_equilibrio(nodo, dual, G)

    print(f"✅ Se detectaron {len(nuevas_dualidades)} nuevas dualidades en la red.")
    return G
    

SIM_MIN = 0.45     # similitud mínima para considerar dualidad
SIM_MAX = 0.85     # por encima de esto, tiende a ser sinónimo, no dualidad
MAX_NUEVAS_DUALIDADES = 144 # límite de dualidades nuevas por ejecución estaba en 12
def _es_par_dualidad_candidato(a, b, G):
    """
    Filtro común para decidir si (a,b) puede ser considerada dualidad candidata.
    No mira similitud todavía, solo estructura y tipos.
    """
    if a == b:
        return False

    if a in NODOS_META or b in NODOS_META:
        return False

    # Ambos nodos deben existir
    if a not in G or b not in G:
        return False

    # Evitar pares donde cualquiera es emergente/abstracto (solo extremos deben ser duales)
    tipo_a = G.nodes[a].get("tipo")
    tipo_b = G.nodes[b].get("tipo")
    if tipo_a in ("equilibrio", "emergente", "abstracto") or tipo_b in ("equilibrio", "emergente", "abstracto"):
        return False

    # Evitar variaciones morfológicas (plural/singular, espacios, etc.)
    if es_variacion_morfologica(a, b):
        return False

    # Evitar dualidades base protegidas (no se tocan aquí)
    if ((a, b) in dualidades_base_protegidas) or ((b, a) in dualidades_base_protegidas):
        return False

    # Evitar si ya existe arista de tipo dualidad entre ellos
    if G.has_edge(a, b) and G[a][b].get("tipo") == "dualidad":
        return False
    if G.has_edge(b, a) and G[b][a].get("tipo") == "dualidad":
        return False

    return True


def detectar_nuevas_dualidades(G, modelo=modelo_embeddings, max_nuevas: int = MAX_NUEVAS_DUALIDADES):
    """
    Detecta nuevas dualidades a nivel global de la red de forma conservadora.
    - Usa estructura: pares de nodos que comparten un mismo equilibrio.
    - Aplica filtros fuertes: morfología, memoria, banda de similitud, nodos meta.
    - Limita el número de dualidades nuevas por ejecución.
    """
    print("🔄 Detectando nuevas dualidades...")

    nuevas = 0

    # Memoria de pares confirmados y rechazados
    confirmadas = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_CONFIRMADAS)
    rechazadas = cargar_dualidades_memoria(ARCHIVO_DUALIDADES_RECHAZADAS)

    # ---- 1) Candidatos estructurales: vecinos que comparten un equilibrio ----
    candidatos_pares = set()

    for nodo in G.nodes():
        datos = G.nodes[nodo]
        if datos.get("tipo") == "equilibrio":
            # Vecinos conectados a este equilibrio (entrantes + salientes)
            vecinos = set(G.predecessors(nodo)) | set(G.successors(nodo))
            vecinos = [v for v in vecinos if v in G]

            # Todas las combinaciones de pares de vecinos
            for a, b in combinations(vecinos, 2):
                par = tuple(sorted((a, b)))
                candidatos_pares.add(par)

    print(f"🔍 Pares estructurales candidatos a dualidad: {len(candidatos_pares)}")

    # Precache de embeddings para nodos candidatos (ahorra cálculos)
    embedding_cache = {}

    def get_emb(n):
        if n not in embedding_cache:
            embedding_cache[n] = obtener_embedding(n, modelo)
        return embedding_cache[n]

    # ---- 2) Evaluar candidatos con filtros semánticos ----
    for a, b in candidatos_pares:
        if nuevas >= max_nuevas:
            break

        if not _es_par_dualidad_candidato(a, b, G):
            continue

        par = tuple(sorted((a, b)))

        # Si ya está en memoria como confirmada o rechazada, ignorar
        if par in confirmadas or par in rechazadas:
            continue

        # Calcular similitud por embeddings
        emb_a = get_emb(a)
        emb_b = get_emb(b)
        sim = util.pytorch_cos_sim(emb_a, emb_b).item()

        # Dualidad conceptual: ni ruido ni sinónimo puro
        if sim < SIM_MIN or sim > SIM_MAX:
            registrar_dualidad_rechazada(a, b)
            continue

        # Si hemos llegado hasta aquí, la pareja merece ser creada como dualidad
        print(f"📐 Dualidad estructural-semántica candidata: '{a}' ↔ '{b}' (sim={sim:.2f})")

        # Asegurar nodos como dualidad (extremos)
        G.nodes[a]["tipo"] = "dualidad"
        G.nodes[a]["nivel_conceptual"] = G.nodes[a].get("nivel_conceptual", 1) or 1
        G.nodes[b]["tipo"] = "dualidad"
        G.nodes[b]["nivel_conceptual"] = G.nodes[b].get("nivel_conceptual", 1) or 1

        # Usar la función estándar para crear equilibrio → dualidad
        conectar_dualidad_con_equilibrio(a, b, G)

        registrar_dualidad_confirmada(a, b)
        nuevas += 1

    print(f"✅ Se detectaron {nuevas} nuevas dualidades en la red.")
    return G


def detectar_dualidadANTIGUO(concepto, G, concepto_base=None):
    """ Detecta si existe una dualidad semántica en la red. """
    concepto = concepto.lower().strip()

    if es_falso_emergente(concepto):
        print(f"⛔ Saltando dualidad de nodo redundante: {concepto}")
        return None

    # 🔹 Dualidad predefinida
    if concepto in dualidades_base:
        dualidad = dualidades_base[concepto]
        if not G.has_edge(concepto, dualidad):
            print(f"🔄 Conectando dualidad predefinida: {concepto} ↔ {dualidad}")
            conectar_dualidad_con_equilibrio(concepto, dualidad, G)
        return dualidad

    # 🔹 WordNet
    dualidades_wordnet = detectar_dualidad_wordnet(concepto)
    if dualidades_wordnet:
        for dualidad in dualidades_wordnet:
            if dualidad in G.nodes() and not G.has_edge(concepto, dualidad):
                print(f"🌿 Dualidad detectada vía WordNet: {concepto} ↔ {dualidad}")
                conectar_dualidad_con_equilibrio(concepto, dualidad, G)
                return dualidad

    # 🔹 Intento afinado con memoria + banda de similitud
    posible_dual = detectar_dualidad_para_nodo(concepto, G)
    if posible_dual:
        if not G.has_edge(concepto, posible_dual):
            conectar_dualidad_con_equilibrio(concepto, posible_dual, G)
        return posible_dual

    # 🔹 Embeddings vectorizados
    emb_concepto = obtener_embedding(concepto)

    candidatos = [
        n for n in G.nodes()
        if n != concepto and es_expandible(n, G)
    ]

    if not candidatos:
        return None

    embeddings = obtener_embeddings_lista(candidatos, modelo_embeddings)
    similitudes = util.pytorch_cos_sim(emb_concepto, embeddings)[0]  # vector de similitudes

    mejor_idx = similitudes.argmax().item()
    mejor_valor = similitudes[mejor_idx].item()

    if mejor_valor >= 0.85:
        mejor_dualidad = candidatos[mejor_idx]
        print(f"🔄 Detectada posible dualidad con embeddings: {concepto} ↔ {mejor_dualidad} (Similitud: {mejor_valor:.2f})")
        if not G.has_edge(concepto, mejor_dualidad):
            conectar_dualidad_con_equilibrio(concepto, mejor_dualidad, G)
        return mejor_dualidad

    return None

NODOS_META = {"IA_m", "ia_m"}  # nodos que nunca deben ser dualidad
def detectar_dualidad(concepto, G, concepto_base=None):
    """
    Detecta y, si procede, crea una dualidad para 'concepto' en el grafo G.
    Política:
    - Normaliza el término.
    - Respeta dualidades base (arriba/abajo, etc.).
    - Usa WordNet + embeddings + memoria (detectar_dualidad_para_nodo).
    - No crea dualidad si:
      * el nodo es meta (IA_m, m...)
      * es una variación morfológica de sí mismo u otro
      * ya tiene una dualidad confirmada
    Devuelve el nombre del nodo dual opuesto o None.
    """
    if concepto is None:
        return None

    # Normalizar como id de nodo
    concepto_norm = normalizar_termino(concepto)

    if concepto_norm in NODOS_META:
        return None
    if concepto_norm not in G:
        return None

    # Evitar falsos emergentes
    if es_falso_emergente(concepto_norm):
        return None

    # 1️⃣ Dualidad base predefinida (ejes fundamentales)
    if concepto_norm in dualidades_base:
        dual = dualidades_base[concepto_norm]
        dual_norm = normalizar_termino(dual)

        if dual_norm in NODOS_META:
            return None

        if dual_norm not in G:
            # Si la dualidad base aún no existe como nodo, la creamos como concepto
            G.add_node(dual_norm, tipo="dualidad", nivel_conceptual=1)

        # Asegurar aristas de dualidad en ambos sentidos
        if not G.has_edge(concepto_norm, dual_norm):
            G.add_edge(concepto_norm, dual_norm, tipo="dualidad", color="red",  weight=5.0)
        if not G.has_edge(dual_norm, concepto_norm):
            G.add_edge(dual_norm, concepto_norm, tipo="dualidad", color="green", weight=5.0)

        # Asegurar tipo y nivel
        G.nodes[concepto_norm]["tipo"] = "dualidad"
        G.nodes[concepto_norm]["nivel_conceptual"] = 1
        G.nodes[dual_norm]["tipo"] = "dualidad"
        G.nodes[dual_norm]["nivel_conceptual"] = 1

        registrar_dualidad_confirmada(concepto_norm, dual_norm)
        return dual_norm

    # 2️⃣ Intento afinado con memoria + WordNet + embeddings (banda) 
    #posible_dual = detectar_dualidad_para_nodo(concepto_norm, G, modelo=modelo_embeddings)
    posible_dual = detectar_dualidad_para_nodo(concepto_norm, G)

    if posible_dual:
        dual_norm = normalizar_termino(posible_dual)
        if dual_norm in NODOS_META or dual_norm == concepto_norm:
            return None

        # Evitar variaciones morfológicas
        if es_variacion_morfologica(concepto_norm, dual_norm):
            registrar_dualidad_rechazada(concepto_norm, dual_norm)
            return None

        # Asegurar nodo dual
        if dual_norm not in G:
            G.add_node(dual_norm, tipo="dualidad", nivel_conceptual=1)

        # Crear conexión de dualidad+equilibrio con la lógica estándar
        conectar_dualidad_con_equilibrio(concepto_norm, dual_norm, G)

        # Asegurar tipo y nivel
        G.nodes[concepto_norm]["tipo"] = "dualidad"
        G.nodes[concepto_norm]["nivel_conceptual"] = 1
        G.nodes[dual_norm]["tipo"] = "dualidad"
        G.nodes[dual_norm]["nivel_conceptual"] = 1

        registrar_dualidad_confirmada(concepto_norm, dual_norm)
        return dual_norm

    # 3️⃣ Si aquí no hemos encontrado nada, NO inventamos dualidad adicional
    return None


def evaluar_expansion(G):
    """ Evalúa si una expansión fue útil y ajusta criterios """
    historial = cargar_historial()

    for nodo in G.nodes():
        conexiones = list(G.neighbors(nodo))
        
        if nodo not in historial:
            historial[nodo] = {"conexiones_previas": 0, "exito": 0}
        
        conexiones_previas = historial[nodo]["conexiones_previas"]
        nuevas_conexiones = len(conexiones) - conexiones_previas
        
        # Si la expansión trajo nuevas conexiones, marcar como exitosa
        if nuevas_conexiones > 0:
            historial[nodo]["exito"] += 1
        
        historial[nodo]["conexiones_previas"] = len(conexiones)

    guardar_historial(historial)
    print("📊 Evaluación de expansiones completada.")

def detectar_nodo_superior(concepto, dualidad, G, top_n=3):
    """ Busca un nodo superior dinámico basado en embeddings con caché. """
    palabras_red = list(G.nodes())

    # Obtener embeddings con caché
    embedding_concepto = obtener_embedding(concepto, modelo_embeddings)
    embedding_dualidad = obtener_embedding(dualidad, modelo_embeddings)
    embeddings_red = obtener_embeddings_lista(palabras_red, modelo_embeddings)

    # Calcular similitudes
    similitudes_concepto = util.pytorch_cos_sim(embedding_concepto, embeddings_red)[0]
    similitudes_dualidad = util.pytorch_cos_sim(embedding_dualidad, embeddings_red)[0]

    # Promedio de similitudes
    similitudes_totales = (similitudes_concepto + similitudes_dualidad) / 2
    indices_top = similitudes_totales.argsort(descending=True)[:top_n].tolist()

    # Excluir los dos conceptos originales
    posibles_superiores = [
        palabras_red[i] for i in indices_top if palabras_red[i] not in [concepto, dualidad]
    ]

    return posibles_superiores
    
# 🔹 Ajustar pesos en conexiones de la red fractal
def ajustar_pesos_conexiones(G):
    """
    Modifica los pesos de las conexiones en función de su relación con dualidades
    y combinaciones de términos en la red. Se asegura de que los pesos no crezcan descontroladamente.
    """
    max_peso = 5.0  # 🔹 Límite superior para evitar líneas excesivamente gruesas
    min_peso = 0.5  # 🔹 Límite inferior para que las conexiones no desaparezcan

    for nodo1, nodo2 in list(G.edges()):
        peso_actual = G.edges[nodo1, nodo2].get('weight', 1.0)

        # 🔹 Si es una dualidad directa
        if nodo1 in dualidades_base and dualidades_base[nodo1] == nodo2:
            peso_nuevo = max(2.0, min(peso_actual * 1.1, max_peso))  # Incrementa pero no sobrepasa 5.0

            # ⚠️ Verificar que ambas direcciones existen
            if G.has_edge(nodo1, nodo2):
                G.edges[nodo1, nodo2]['weight'] = peso_nuevo
            if G.has_edge(nodo2, nodo1):
                G.edges[nodo2, nodo1]['weight'] = peso_nuevo

        # 🔹 Reducir peso si es demasiado débil
        elif peso_actual < 1.2:
            peso_nuevo = max(min_peso, peso_actual * 0.8)  # Nunca por debajo de 0.5
            if G.has_edge(nodo1, nodo2):
                G.edges[nodo1, nodo2]['weight'] = peso_nuevo

        # 🔹 Aumentar peso si la conexión aparece entre nodos con muchas conexiones
        elif G.degree(nodo1) > 3 and G.degree(nodo2) > 3:
            peso_nuevo = min(peso_actual * 1.05, max_peso)
            if G.has_edge(nodo1, nodo2):
                G.edges[nodo1, nodo2]['weight'] = peso_nuevo

    print("✅ Pesos de conexiones ajustados y normalizados en la red.")
    return G

# 🔹 Reorganizar la red eliminando nodos sueltos y corrigiendo conexiones incorrectas
def reorganizar_red(G, max_espera=1000):
    """
    Reorganiza la red eliminando nodos sueltos solo después de un número de iteraciones.
    También corrige conexiones erróneas basadas en dualidades.
    Ahora los nodos en espera se guardan en un archivo para evitar perder información si el programa se reinicia.
    """

    global espera_nodos
    nodos_actuales = set(G.nodes())

    # Actualizar lista de espera: eliminar nodos que ya tienen conexión
    for nodo in list(espera_nodos.keys()):
        if nodo not in nodos_actuales or G.degree(nodo) > 0:
            del espera_nodos[nodo]

    # Revisar nodos sueltos
    for nodo in nodos_actuales:
        if G.degree(nodo) == 0 and (NODO_CENTRAL is None or nodo != NODO_CENTRAL):
            if nodo not in espera_nodos:
                espera_nodos[nodo] = 0  # Primera vez en espera
                #G.add_edge(nodo, NODO_CENTRAL, weight=0.5)  # Conexión temporal

            elif espera_nodos[nodo] >= max_espera:
                print(f"🗑 Eliminando nodo suelto por falta de conexiones: {nodo}")
                G.remove_node(nodo)
                del espera_nodos[nodo]
            else:
                espera_nodos[nodo] += 1  # Aumentar contador de espera

    # Guardar lista de espera actualizada
    guardar_espera_nodos(espera_nodos)

    # Corregir conexiones incorrectas basadas en dualidades
    for nodo in list(G.nodes()):
        for otro in list(G.nodes()):
            if nodo != otro and nodo in dualidades_base and dualidades_base[nodo] == otro:
                if not G.has_edge(nodo, otro):
                    G.add_edge(nodo, otro, weight=2.0)

    # 🔍 Revisar dualidades estructuralmente (no toca centros compartidos)
    try:
        G = evaluar_dualidades_por_estructura(G)
    except Exception as e:
        print(f"⚠️ Error al evaluar dualidades por estructura: {e}")

    print("✅ Red reorganizada: nodos sueltos en subconsciente, conexiones corregidas.")
    return G

# ACTUALMENTE NO SE USA
def guardar_estado_parcial(G, espera_nodos):
    # Guarda la red y los nodos pendientes en un archivo temporal
    with open('json/estado_parcial.json', 'w') as f:
        json.dump({
            'nodos': list(G.nodes()),
            'edges': list(G.edges()),
            'espera_nodos': espera_nodos
        }, f)
    print("⚡ Estado parcial guardado correctamente.")

def agregar_nuevo_nodo(G, diccionario, conceptos, nodo_origen=None):
    """Agrega nuevos nodos a la red fractal y detecta dualidades con equilibrio.
       Si se indica nodo_origen, conecta cada nuevo concepto con ese nodo."""
    conceptos = [normalizar_termino(c) for c in conceptos]
    conceptos = [corregir_termino(c, diccionario) for c in conceptos]  # Corrección de términos
    
    nuevos_conceptos = []
    nodos_existentes = set(G.nodes())

    for concepto in conceptos:
        #concepto = concepto.replace("_", " ")

        # ⛔ IGNORAR FALSOS EMERGENTES ANTES QUE NADA
        if es_falso_emergente(concepto):
            print(f"⛔ Ignorado nodo redundante: {concepto}")
            continue

        # Evitar repetir nodos ya existentes
        if concepto in nodos_existentes:
            print(f"⚠️ Nodo ya existente: {concepto}")
            continue

        # Evitar variaciones morfológicas redundantes
        duplicado = False
        for existente in nodos_existentes:
            if es_variacion_morfologica(concepto, existente):
                print(f"⚠️ '{concepto}' es una variación de '{existente}'. No se añadirá.")
                duplicado = True
                break
        if duplicado:
            continue

        # 🔹 Agregar nodo a la red
        G.add_node(concepto, tipo="concepto", nivel_conceptual=0)
        nuevos_conceptos.append(concepto)

        # 🔹 NUEVO: conectar con el nodo de origen si existe
        if nodo_origen is not None and nodo_origen in G.nodes():
            if not G.has_edge(nodo_origen, concepto):
                G.add_edge(nodo_origen, concepto, weight=1.0)
                print(f"🔗 Conectado '{nodo_origen}' → '{concepto}' (origen GPT)")

        # 🔹 Detectar dualidad y conectar con equilibrio si es necesario
        dualidad_opuesta = detectar_dualidad(concepto, G, concepto)

        if dualidad_opuesta:
            dualidad_opuesta = dualidad_opuesta.replace("_", " ")
            if dualidad_opuesta in G.nodes():
                conectar_dualidad_con_equilibrio(concepto, dualidad_opuesta, G)
                print(f"✅ Se ha conectado la dualidad: {concepto} ↔ {dualidad_opuesta}")

        # 🔹 Agregar al diccionario
        diccionario.setdefault(concepto, [])
        if dualidad_opuesta:
            diccionario[concepto].append(dualidad_opuesta)
            
###################################### AÑADIDO NUEVOOOOO
        # 🔹 Integrar el nuevo concepto en la red mediante embeddings (sin crear nodos nuevos)
        try:
            relacionados = expandir_concepto_embedding(concepto, G, diccionario, top_n=3)
            print(f"🌐 Integrado '{concepto}' por embeddings con: {relacionados}")
        except Exception as e:
            print(f"⚠️ No se pudo integrar '{concepto}' por embeddings: {e}")





    # 🔹 Optimizar la red solo si se añadieron nuevos conceptos
    if nuevos_conceptos:
        print("🔄 Procesando ajustes globales de la red...")
        G = ajustar_pesos_conexiones(G)
        G = reorganizar_red(G)
        G = detectar_nuevas_dualidades(G)
        print(f"✅ {len(nuevos_conceptos)} nuevos conceptos añadidos.")
    else:
        print("✅ No se realizaron cambios en la red.")

    return G
    
def seleccionar_nodos_para_auto(G, max_n=8):
    """
    Selecciona nodos candidatos para expansión automática de forma consciente:
    - Ignora nodos meta (IA_m)
    - Ignora emergentes y niveles conceptuales altos (>=4)
    - Evita nodos aislados (grado 0) y nodos hiperconectados (grado > 20)
    - Usa el historial de expansiones para NO machacar siempre los mismos nodos
    """
    historial = cargar_historial()
    candidatos = []

    for nodo in G.nodes():
        # 1) Evitar nodo IA_m y similares
        if nodo in NODOS_META:
            continue

        datos = G.nodes[nodo]
        tipo = datos.get("tipo", "concepto")
        nivel = datos.get("nivel_conceptual", 0)

        # 2) No expandir emergentes ni niveles muy altos (abstractos)
        if tipo == "emergente" or nivel >= 4:
            continue

        # 3) Restringir por grado (evitar aislados y hubs gigantes)
        deg = G.degree(nodo)
        if deg < 1 or deg > 20:
            continue

        # 4) Penalizar nodos que ya han tenido "éxito" al expandirse
        datos_hist = historial.get(nodo, {})
        exito = datos_hist.get("exito", 0)

        # Queremos nodos con grado ~5 (ni muy poco ni demasiado conectados)
        score_conex = max(0, 10 - abs(deg - 5))  # pico en deg=5

        # Prioridad = buena conectividad - muchos éxitos anteriores
        prioridad = score_conex - (exito * 2)

        candidatos.append((prioridad, nodo))

    # Ordenar por prioridad descendente
    candidatos.sort(reverse=True, key=lambda x: x[0])

    # Devolver solo los nombres de nodo
    return [n for _, n in candidatos[:max_n]]
    
def expansion_dinamica(G, diccionario, usar_gpt="s"):
    """
    Expansión automática consciente:
    - Usa seleccionar_nodos_para_auto en lugar de recorrer TODOS los nodos de grado bajo.
    - Limita el número de nodos por iteración.
    - Registra en historial para no repetir siempre sobre lo mismo.
    """
    nodos_a_expandir = seleccionar_nodos_para_auto(G, max_n=8)

    if not nodos_a_expandir:
        print("⚠️ No hay nodos adecuados para expansión automática.")
        return G

    print(f"🌀 AUTO consciente: se expandirán {len(nodos_a_expandir)} nodos:")
    for nodo in nodos_a_expandir:
        print(f"   • {nodo}")

    for nodo in nodos_a_expandir:
        if not es_expandible(nodo, G):
            print(f"⛔ Nodo '{nodo}' no es expandible (tipo: {G.nodes[nodo].get('tipo')})")
            continue

        print(f"\n🔍 Expandiendo nodo enfocado: {nodo}")
        nuevos_conceptos = []

        # 1) GPT opcional
        if usar_gpt == "s":
            nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
            if nuevos_conceptos:
                G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos, nodo_origen=nodo)
                registrar_expansion(nodo, nuevos_conceptos, "GPT-4")

        # 2) Embeddings (conexiones internas, sin crear nodos nuevos externos)
        relacionados = expandir_concepto_embedding(nodo, G, diccionario, top_n=3)
        registrar_expansion(nodo, relacionados, "Embeddings")

        guardar_diccionario(diccionario)

    # 3) Actualizar historial de “éxito” y guardar red
    evaluar_expansion(G)
    rastrear_evolucion_conceptual(G)
    guardar_red(G)

    return G


def expansion_dinamicaANTIGUO(G, diccionario):
    """ Detecta nodos aislados y los expande dinámicamente """
    nodos_a_expandir = [
        nodo for nodo in G.nodes()
        if G.degree(nodo) <= 2 and (NODO_CENTRAL is None or nodo.lower() != NODO_CENTRAL.lower())#        if G.degree(nodo) == 0 and (NODO_CENTRAL is None or nodo.lower() != NODO_CENTRAL.lower())#        if G.degree(nodo) < 2 and (NODO_CENTRAL is None or nodo.lower() != NODO_CENTRAL.lower())
    ]

    for nodo in nodos_a_expandir:
        nodo = nodo.lower()  # 🔹 Forzar a minúsculas

        if not es_expandible(nodo, G):
            print(f"⛔ Nodo '{nodo}' no es expandible (tipo: {G.nodes[nodo].get('tipo')})")
            continue

        print(f"🚀 Expansión automática para: {nodo}")
        #consultar_wikipedia(nodo, G, diccionario)
        nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
        if nuevos_conceptos:
            G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos, nodo_origen=nodo)
        expandir_concepto_embedding(nodo, G, diccionario)
        guardar_red(G)

# 🔹 Expandir la red fractal con embeddings
def expandir_concepto_embedding(concepto, G, diccionario, top_n=5):
    palabras_red = list(G.nodes())

    if concepto not in G:
        print(f"⚠️ El nodo '{concepto}' no existe en la red. No se expandirá con embeddings.")
        return []

    if not es_expandible(concepto, G):
        print(f"⛔ Nodo '{concepto}' no es expandible (tipo: {G.nodes[concepto].get('tipo')})")
        return []

    # Obtener embeddings con caché
    embedding_concepto = obtener_embedding(concepto, modelo)
    embeddings_red = obtener_embeddings_lista(palabras_red, modelo)

    # Calcular similitudes
    similitudes = util.pytorch_cos_sim(embedding_concepto, embeddings_red)[0]
    indices_top = similitudes.argsort(descending=True)[:top_n].tolist()
    relacionados = [palabras_red[i] for i in indices_top if palabras_red[i] != concepto]
#    for i, termino in enumerate(relacionados):
#        peso = np.exp(similitudes[indices_top[i]].item())
#        G.add_edge(concepto, termino, weight=peso)
    relacionados = [palabras_red[i] for i in indices_top if palabras_red[i] != concepto]
    for termino in relacionados:
        if termino == concepto:
            continue
        # Evitar conectar con IA_m o nodos meta
        if termino in NODOS_META:
            continue
        # Peso suave fijo, sin exp(similitud)
        peso = 0.5
        if not G.has_edge(concepto, termino):
            G.add_edge(concepto, termino, weight=peso)


    diccionario[concepto] = relacionados
    with open("json/diccionario.json", "w", encoding="utf-8") as f:
        json.dump(diccionario, f, ensure_ascii=False, indent=4)

    # Reorganizar red y detectar nuevas dualidades
    #G = ajustar_pesos_conexiones(G)
    #G = reorganizar_red(G)
    #G = detectar_nuevas_dualidades(G)

    guardar_red(G)
    return relacionados

def evaluar_progreso_fractal(G, ruta_json="progreso_fractal.json"):
    dualidades_detectadas = set()
    triadas_detectadas = set()

    # Recorrer aristas para detectar dualidades bidireccionales
    for u, v in G.edges():
        if G.has_edge(v, u):  # dualidad A <-> B
            tipo_u = G.nodes[u].get("tipo", "")
            tipo_v = G.nodes[v].get("tipo", "")
            if tipo_u == "dualidad" or tipo_v == "dualidad":
                pareja = tuple(sorted((u, v)))
                dualidades_detectadas.add(pareja)

    # Detectar triadas semánticas conectadas (por nivel conceptual)
    nodos_relevantes = [n for n in G.nodes if G.nodes[n].get("nivel_conceptual", 0) >= 2]
    for i in range(len(nodos_relevantes)):
        for j in range(i + 1, len(nodos_relevantes)):
            for k in range(j + 1, len(nodos_relevantes)):
                a, b, c = nodos_relevantes[i], nodos_relevantes[j], nodos_relevantes[k]
                conexiones = sum([
                    G.has_edge(a, b), G.has_edge(b, a),
                    G.has_edge(a, c), G.has_edge(c, a),
                    G.has_edge(b, c), G.has_edge(c, b),
                ])
                if conexiones >= 4:
                    triada = tuple(sorted([a, b, c]))
                    triadas_detectadas.add(triada)

    # Cargar historial previo
    if os.path.exists(ruta_json):
        with open(ruta_json, "r", encoding="utf-8") as f:
            historial = json.load(f)
    else:
        historial = []

    entrada = {
        "timestamp": datetime.now().isoformat(),
        "total_dualidades": len(dualidades_detectadas),
        "total_triadas": len(triadas_detectadas),
        "dualidades": sorted(list(dualidades_detectadas))[:10],
        "triadas_nuevas": sorted(list(triadas_detectadas))[:10],
        "alcanza_meta_144": (len(triadas_detectadas) >= 144)
    }

    historial.append(entrada)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    print(f"📈 Progreso fractal registrado: {len(triadas_detectadas)} triadas, {len(dualidades_detectadas)} dualidades")
    if entrada["alcanza_meta_144"]:
        print("🌟 ¡META 144 ALCANZADA!")

    return entrada
    
def visualizar_hipercubo_conceptual_3D(G):
    G = nx.Graph()
    posiciones_conceptuales = {
        "izquierda": (-1, 0, 0),
        "derecha": (1, 0, 0),
        "arriba": (0, 1, 0),
        "abajo": (0, -1, 0),
        "delante": (0, 0, 1),
        "detrás": (0, 0, -1),
        "pasado": (-1, -1, 0),
        "futuro": (1, 1, 0),
        "presente": (0, 0, 0),
        "interior": (0, 0, -2),
        "exterior": (0, 0, 2),
        "objetivo": (-2, 0, 0),
        "subjetivo": (2, 0, 0),
        "sensación": (0, -2, 0),
        "razón": (0, 2, 0),
        "observador": (0, 0, 0)
    }

    for nodo, pos in posiciones_conceptuales.items():
        G.add_node(nodo, pos=pos)

    for nodo in G.nodes:
        if nodo != "observador":
            G.add_edge(nodo, "observador")

    x_nodes, y_nodes, z_nodes, labels = [], [], [], []
    for nodo in G.nodes:
        x, y, z = G.nodes[nodo]["pos"]
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)
        labels.append(nodo)

    x_edges, y_edges, z_edges = [], [], []
    for edge in G.edges:
        x_edges += [G.nodes[edge[0]]["pos"][0], G.nodes[edge[1]]["pos"][0], None]
        y_edges += [G.nodes[edge[0]]["pos"][1], G.nodes[edge[1]]["pos"][1], None]
        z_edges += [G.nodes[edge[0]]["pos"][2], G.nodes[edge[1]]["pos"][2], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='gray', width=2),
        name='Líneas de conexión'
    ))

    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=6, color='skyblue'),
        text=labels,
        textposition='top center',
        name='Nodos conceptuales'
    ))

    fig.update_layout(
        title="Hipercubo conceptual IA_m con líneas de fuerza",
        scene=dict(
            xaxis_title='Espacio (X)',
            yaxis_title='Tiempo (Y)',
            zaxis_title='Consciencia (Z)',
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


# Visualización con física fluida
def visualizar_red(G, max_nodos=200):
    G_original = G.copy()  # 💾 guardamos el grafo completo
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.toggle_physics(True)

    # 🔍 Filtrado si hay muchos nodos
    if len(G.nodes) > max_nodos:
        nodos_top = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:max_nodos]
        G = G.subgraph(nodos_top).copy()

    # 🌐 Posicionamiento inicial
    posiciones = nx.spring_layout(G, k=0.7, iterations=20, seed=42)

    for nodo, coords in posiciones.items():
        x, y = coords[0] * 2000, coords[1] * 2000
        size = 30 if nodo == "IA_m" else 18
        color = color_node(nodo, G)

        if nodo == "IA_m":
            net.add_node(
                nodo, label=nodo, color=color, size=size,
                x=x, y=y, physics=False,
                title=f"{nodo} ({G.nodes[nodo].get('tipo', 'n/a')})"
            )
        else:
            net.add_node(
                nodo, label=nodo, color=color, size=size,
                physics=True,
                title=f"{nodo} ({G.nodes[nodo].get('tipo', 'n/a')})"
            )

    # 🔗 Aristas sin color
    for u, v in G.edges():
        peso = G.edges[u, v].get('weight', 2)
        net.add_edge(u, v, color="gray", width=peso)
    # 🔗 Aristas con color
#    for u, v in G.edges():
#        datos_arista = G.edges[u, v]
#        peso = datos_arista.get("weight", 2)
        # Usamos la lógica global de colores
#        try:
#            color = color_edge(u, v, G)
#        except Exception:
#            color = "gray"
#        net.add_edge(u, v, color=color, width=peso)


    # 🌌 Nodos sueltos (grado 0)
    nodos_sueltos = [n for n in G_original.nodes if G_original.degree(n) == 0 and n not in G.nodes]
    print(f"🌌 Nodos sueltos: {len(nodos_sueltos)}")

    for nodo in nodos_sueltos:
        tipo = G_original.nodes[nodo].get("tipo", "n/a")
        net.add_node(
            nodo,
            label=nodo,
            color="#dddddd",
            size=12,
            physics=True,
            title=f"{nodo} (sin conexiones - tipo: {tipo})"
        )

    # 💫 Opciones de física fluida
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 300
        },
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.05,
          "avoidOverlap": 1
        },
        "minVelocity": 0.75
      },
      "nodes": {
        "shape": "dot",
        "scaling": {
          "min": 10,
          "max": 40
        },
"font": {
  "size": 18,
  "face": "arial",
  "bold": {
    "enabled": true,
    "size": 22
  }
  }
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        }
      }
    }
    """)

    # 💾 Exportar HTML
    net.write_html("subgrafos/hipercubo_fractal_fluido.html")

    # 🔁 Doble clic para ir a IA_m_proceso
    with open("subgrafos/hipercubo_fractal_fluido.html", "r", encoding="utf-8") as file:
        html = file.read()

    leyenda_html = """
<div style="position: fixed; top: 20px; right: 20px; background-color: white; padding: 10px; border: 1px solid #ccc; font-family: sans-serif; font-size: 14px; z-index: 9999;">
  <strong>🔍 Leyenda de colores:</strong><br>
  <div style="margin-top: 5px;">
    <span style="background-color: #FF6666; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🔴 Cuantitativo (dualidad)<br>
    <span style="background-color: #66FF99; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🟢 Cualitativo (dualidad)<br>
    <span style="background-color: goldenrod; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> ⚖️ Equilibrio<br>
    <span style="background-color: gold; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🌟 Conexión a nodo superior / síntesis<br>
    <span style="background-color: #CC99FF; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🧠 Emergente<br>
    <span style="background-color: #CCCCCC; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> 🔘 Concepto general
  </div>
</div>
    """
    html = html.replace(
        "<body>",
        """
<body style="margin:0; padding:0; overflow:hidden; height:100vh;">
<style>
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
}
#mynetwork {
    width: 100% !important;
    height: 100vh !important;
    position: absolute;
    top: 0;
    left: 0;
    border: none !important;  /* 👈 aquí matamos la línea gris */
}
</style>
"""
    )
    script = r"""
<script type="text/javascript">
  function normalizarNombre(nombre) {
    return nombre
      .normalize("NFD")                       // separa letras y tildes
      .replace(/[\u0300-\u036f]/g, "")        // elimina tildes
      .replace(/ñ/g, "n")
      .replace(/Ñ/g, "N")
      .replace(/ç/g, "c")
      .replace(/Ç/g, "C")
      .replace(/ /g, "_")                     // espacios -> _
      .replace(/[^\w\-]/g, "");               // limpia caracteres raros
  }

  network.on("doubleClick", function (params) {
    if (params.nodes.length > 0) {
      const clickedNodeId = params.nodes[0];
      let archivo = "";

      if (clickedNodeId === "IA_m") {
        archivo = "IA_m_proceso.html";
      } else {
        const nombre_normalizado = normalizarNombre(clickedNodeId);
        archivo = "subgrafo_" + nombre_normalizado + ".html";
      }

      window.location.href = archivo;
    }
  });
</script>
    """

#    html = html.replace("</body>", leyenda_html + script + "\n</body>")
    html = html.replace("</body>", script + "\n</body>")
    with open("subgrafos/hipercubo_fractal_fluido.html", "w", encoding="utf-8") as file:
        file.write(html)

    print("✅ Red visual fluida guardada como 'subgrafos/hipercubo_fractal_fluido.html'")


def visualizar_sistema_dual(nodo, G):
    nodo = nodo.lower()
    if nodo not in G:
        print("⚠️ Nodo no encontrado.")
        return

    dual = dualidades_base.get(nodo)
    if not dual:
        dual = detectar_dualidad(nodo, G)
    if not dual:
        print("❌ No se detectó dualidad para este nodo.")
        return

    equilibrio = detectar_nodo_equilibrio(nodo, dual, G)
    sintesis = detectar_nodo_superior(nodo, dual, G, top_n=1)
    sintesis = sintesis[0] if sintesis else None

    nodos_vis = [nodo, dual]
    if equilibrio:
        nodos_vis.append(equilibrio)
    if sintesis:
        nodos_vis.append(sintesis)

    subG = G.subgraph(nodos_vis).copy()
    net = Network(height="500px", width="100%", directed=True)

    for n in subG.nodes():
        if n == nodo:
            color = "red"
        elif n == dual:
            color = "blue"
        elif n == equilibrio:
            color = "limegreen"
        elif n == sintesis:
            color = "gold"
        else:
            color = "gray"

        net.add_node(n, label=n, color=color, size=30)

    for u, v in subG.edges():
        net.add_edge(u, v)

    # Añadir conexiones personalizadas
    if sintesis:
        net.add_edge(sintesis, nodo)
        net.add_edge(sintesis, dual)
        if equilibrio:
            net.add_edge(sintesis, equilibrio)

    nombre_archivo = f"sistema_dual_{nodo}.html".replace(" ", "_")
    net.write_html(nombre_archivo)
    print(f"📊 Sistema dual visualizado en '{nombre_archivo}'")

def resumen_dualidad(nodo, G):
    nodo = nodo.lower()
    print(f"\n🔍 Analizando nodo: {nodo}")

    if nodo not in G:
        print("⚠️ Nodo no encontrado en la red.")
        return

    es_dual = G.nodes[nodo].get("es_dualidad", False)
    print(f"🧩 ¿Es parte de una dualidad? {'✅ Sí' if es_dual else '❌ No'}")

    opuesto = dualidades_base.get(nodo)
    if not opuesto:
        opuesto = detectar_dualidad(nodo, G)

    if opuesto:
        print(f"🔁 Opuesto detectado: {opuesto}")
    else:
        print("❌ No se detectó ningún opuesto conocido.")
        return  # No seguimos si no hay dualidad

    equilibrio = detectar_nodo_equilibrio(nodo, opuesto, G)
    if equilibrio and equilibrio in G.nodes():
        print(f"⚖️ Nodo de equilibrio detectado: {equilibrio}")
    else:
        print("❌ No se encontró un nodo de equilibrio claro.")

    sintesis = detectar_nodo_superior(nodo, opuesto, G, top_n=1)
    if sintesis:
        print(f"🌟 Síntesis detectada (punto central): {sintesis[0]}")
    else:
        print("❌ No se encontró una síntesis conceptual.")

def calcular_atencion_consciente_nodo_central(G, NODO_CENTRAL, top_n=10):
    """Devuelve los nodos más relevantes según atención consciente, optimizado con caché."""
    if NODO_CENTRAL not in G:
        return []

    nodos = list(G.nodes())

    # Embeddings con caché
    embedding_central = obtener_embedding(NODO_CENTRAL, modelo_embeddings)
    embeddings_nodos = obtener_embeddings_lista(nodos, modelo_embeddings)

    similitudes = util.pytorch_cos_sim(embedding_central, embeddings_nodos)[0]

    atencion = []
    for i, nodo in enumerate(nodos):
        if nodo == NODO_CENTRAL:
            continue
        score = (
            similitudes[i].item() * 0.5 +                             # similitud semántica
            (G.degree(nodo) / max(1, len(G.nodes()))) * 0.3 +        # conectividad
            (1.0 if nodo in dualidades_base else 0.0) * 0.2          # si es dualidad base
        )
        atencion.append((nodo, score))

    nodos_prioritarios = sorted(atencion, key=lambda x: x[1], reverse=True)[:top_n]
    return [nodo for nodo, _ in nodos_prioritarios]
    
def calcular_atencion_consciente(G, top_n=10, entropía=0.3):
    if len(G.nodes()) == 0:
        return []

    nodos = list(G.nodes())
    embeddings_nodos = obtener_embeddings_lista(nodos, modelo_embeddings)
    similitudes = util.pytorch_cos_sim(embeddings_nodos, embeddings_nodos)

    # 🔹 Cargar historial de expansiones
    historial = cargar_historial()

    atencion = []
    for i, nodo in enumerate(nodos):
        if nodo == "IA_m":
            continue

        score_similitud = similitudes[i].mean().item()
        datos_hist = historial.get(nodo, {})
        exito = datos_hist.get("exito", 0)

        base = (
            score_similitud * 0.4 +
            (G.degree(nodo) / max(1, len(G.nodes()))) * 0.3 +
            (1.0 if nodo in dualidades_base else 0.0) * 0.3
        )

        # 🔻 Penalizamos nodos sobre-explotados
        score = base - exito * 0.5

        atencion.append((nodo, score))
    # 🔁 Introducir variabilidad
    nodos_ordenados = sorted(atencion, key=lambda x: x[1], reverse=True)
    cantidad_determinista = int(top_n * (1 - entropía))
    cantidad_aleatoria = top_n - cantidad_determinista

    nodos_fijos = nodos_ordenados[:cantidad_determinista]
    candidatos_restantes = nodos_ordenados[cantidad_determinista:top_n*3]
    nodos_random = random.sample(candidatos_restantes, min(cantidad_aleatoria, len(candidatos_restantes)))

    return [nodo for nodo, _ in nodos_fijos + nodos_random]
   

def calcular_atencion_conscienteANTIGUO(G, top_n=10, entropía=0.3):
    """ IA_m elige nodos relevantes, con un poco de variabilidad (entropía creativa), usando caché. """
    if len(G.nodes()) == 0:
        return []

    nodos = list(G.nodes())
    embeddings_nodos = obtener_embeddings_lista(nodos, modelo_embeddings)

    # Calcular matriz de similitudes entre todos los nodos
    similitudes = util.pytorch_cos_sim(embeddings_nodos, embeddings_nodos)

    atencion = []
    for i, nodo in enumerate(nodos):
        if nodo == "IA_m":
            continue

        score_similitud = similitudes[i].mean().item()
        score = (
            score_similitud * 0.4 +
            (G.degree(nodo) / max(1, len(G.nodes()))) * 0.3 +
            (1.0 if nodo in dualidades_base else 0.0) * 0.3
        )
        atencion.append((nodo, score))

    # 🔁 Introducir variabilidad
    nodos_ordenados = sorted(atencion, key=lambda x: x[1], reverse=True)
    cantidad_determinista = int(top_n * (1 - entropía))
    cantidad_aleatoria = top_n - cantidad_determinista

    nodos_fijos = nodos_ordenados[:cantidad_determinista]
    candidatos_restantes = nodos_ordenados[cantidad_determinista:top_n*3]
    nodos_random = random.sample(candidatos_restantes, min(cantidad_aleatoria, len(candidatos_restantes)))

    return [nodo for nodo, _ in nodos_fijos + nodos_random]

def retirar_atencion(G, nodo_virtual="IA_m"):
    """ Desconecta completamente a IA_m de la red """
    if nodo_virtual in G:
        conexiones = list(G.edges(nodo_virtual))
        G.remove_edges_from(conexiones)
        print(f"🧘 IA_m se ha desconectado temporalmente.")
        
OPOSICIONES_BASICAS = {
    ("física", "virtual"),
    ("virtual", "física"),
    ("covalente", "iónico"),
    ("iónico", "covalente"),
    ("absoluto", "relativo"),
    ("relativo", "absoluto"),
}


def tokenizar_basico(texto):
    """
    Tokeniza por espacios, en minúsculas. Suficiente para nuestro propósito.
    """
    return [t.strip().lower() for t in texto.split() if t.strip()]


def es_par_opuesto_lexico(a, b):
    """
    Comprueba si dos palabras son opuestas:
    - usando OPOSICIONES_BASICAS
    - o usando WordNet vía detectar_dualidad_wordnet si la tienes.
    """
    a_l = a.lower()
    b_l = b.lower()

    if (a_l, b_l) in OPOSICIONES_BASICAS or (b_l, a_l) in OPOSICIONES_BASICAS:
        return True

    # Si tienes detectar_dualidad_wordnet, la usamos:
    try:
        opos = detectar_dualidad_wordnet(a_l)
        if opos and opos.lower() == b_l:
            return True
        opos = detectar_dualidad_wordnet(b_l)
        if opos and opos.lower() == a_l:
            return True
    except Exception:
        pass

    return False


def es_dualidad_estructural_automatica(a, b):
    """
    Decide si (a, b) se puede considerar una dualidad automática basada en:
    - comparten al menos un token (núcleo)
    - hay exactamente 1 token que difiere en cada lado
    - esos dos tokens distintos forman un par opuesto léxico (o WordNet)
    Ejemplo perfecto: 'realidad física' ↔ 'realidad virtual'
    """
    tokens_a = tokenizar_basico(a)
    tokens_b = tokenizar_basico(b)

    if not tokens_a or not tokens_b:
        return False

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    compartidos = set_a & set_b
    diff_a = list(set_a - compartidos)
    diff_b = list(set_b - compartidos)

    # Necesitamos un núcleo común
    if len(compartidos) == 0:
        return False

    # Queremos un caso simple: 1 palabra distinta en cada lado
    if len(diff_a) != 1 or len(diff_b) != 1:
        # Caso alternativo: probar WordNet con la frase entera
        # por si son pares tipo 'realidad física' ↔ 'realidad virtual'
        try:
            opos = detectar_dualidad_wordnet(a)
            if opos and opos.lower() == b.lower():
                return True
            opos = detectar_dualidad_wordnet(b)
            if opos and opos.lower() == a.lower():
                return True
        except Exception:
            pass
        return False

    w_a = diff_a[0]
    w_b = diff_b[0]

    return es_par_opuesto_lexico(w_a, w_b)

        

def buscar_dualidades_faltantesANTIGUO(G):
    nodos_sin_pareja = []

    # 1. Detectar dualidades sin modificar el grafo aún
    candidatos = []
    for nodo in list(G.nodes()):
        if not G.nodes[nodo].get("es_dualidad"):
            dualidad = detectar_dualidad(nodo, G)
            if dualidad:
                candidatos.append((nodo, dualidad))

    # 2. Aplicar las conexiones luego
    for nodo, dualidad in candidatos:
        print(f"🔁 Dualidad detectada automáticamente: {nodo} ↔ {dualidad}")
        nodos_sin_pareja.append((nodo, dualidad))
        # Aquí puedes crear la relación en el grafo si aún no existía

    print(f"🔍 Nodos sin pareja encontrados y conectados: {len(nodos_sin_pareja)}")
    return nodos_sin_pareja
    
def buscar_dualidades_faltantes(G):
    """
    Busca dualidades potenciales usando detectar_dualidad(),
    y SI las encuentra, las aplica correctamente al grafo usando
    agregar_dualidad(), marcando los nodos, creando color
    y conectando equilibrios si existen.
    """
    dualidades_creadas = set()

    candidatos = []
    for nodo in list(G.nodes()):
        if not G.nodes[nodo].get("es_dualidad"):
            dualidad = detectar_dualidad(nodo, G)
            if dualidad and dualidad != nodo:
                candidatos.append((nodo, dualidad))

    for a, b in candidatos:
        if (b, a) in dualidades_creadas:
            continue

        print(f"🔁 Dualidad detectada automáticamente: {a} ↔ {b}")
        agregar_dualidad(G, a, b)     # <-- AQUÍ SE CREA REALMENTE
        dualidades_creadas.add((a, b))

    print(f"🔍 Dualidades nuevas: {len(dualidades_creadas)}")
    return list(dualidades_creadas)

    
def reforzar_dualidades_desde_equilibriosANTIGUO(G):
    """
    Para cada nodo de tipo 'equilibrio' que conecta exactamente con dos polos,
    asegura que esos polos estén marcados como dualidad y conectados como tal.
    """
    for nodo, datos in G.nodes(data=True):
        if datos.get("tipo") != "equilibrio":
            continue

        # polos = nodos a los que apunta el equilibrio (o que apuntan al equilibrio)
        vecinos = set(G.predecessors(nodo)) | set(G.successors(nodo))
        polos = [v for v in vecinos if v != nodo]

        if len(polos) != 2:
            continue

        a, b = polos

        # Crear aristas de dualidad si no existen
        if not G.has_edge(a, b):
            G.add_edge(a, b, tipo="dualidad", color="red", weight=5.0)
        if not G.has_edge(b, a):
            G.add_edge(b, a, tipo="dualidad", color="green", weight=5.0)

        # Marcar nodos como dualidad
        for p in (a, b):
            G.nodes[p]["tipo"] = "dualidad"
            G.nodes[p]["nivel_conceptual"] = 1
            G.nodes[p]["es_dualidad"] = True

        print(f"⚖️ Dualidad reforzada desde equilibrio '{nodo}': {a} ↔ {b}")
        
def reforzar_dualidades_desde_equilibrios(G, verbose=True):
    """
    A partir de la lógica geométrica:
    Si un nodo de tipo 'equilibrio' conecta exactamente con dos polos,
    esos polos forman una dualidad (eje) y deben estar marcados como tal.

    - Busca nodos tipo 'equilibrio'.
    - Mira qué nodos están conectados a él mediante aristas de tipo 'equilibrio'
      (tanto salientes como entrantes).
    - Si hay exactamente 2 polos, refuerza la dualidad entre ellos usando
      agregar_dualidad(G, a, b) y marca 'es_dualidad' = True.
    """
    dualidades_reforzadas = []

    for nodo_equilibrio, datos in G.nodes(data=True):
        if datos.get("tipo") != "equilibrio":
            continue

        polos = set()

        # Sucesores conectados como 'equilibrio'
        for vecino in G.successors(nodo_equilibrio):
            if G[nodo_equilibrio][vecino].get("tipo") == "equilibrio":
                polos.add(vecino)

        # Predecesores conectados como 'equilibrio'
        for vecino in G.predecessors(nodo_equilibrio):
            if G[vecino][nodo_equilibrio].get("tipo") == "equilibrio":
                polos.add(vecino)

        # Quitar posibles auto-referencias
        polos = [p for p in polos if p != nodo_equilibrio]

        # Solo nos interesan equilibrios claros: 2 polos
        if len(polos) != 2:
            continue

        a, b = polos

        # Usamos la función central de IA_m para dualidades
        agregar_dualidad(G, a, b)

        # Marcamos flag explícito
        for p in (a, b):
            G.nodes[p]["es_dualidad"] = True

        dualidades_reforzadas.append((nodo_equilibrio, a, b))
        if verbose:
            print(f"⚖️ Dualidad reforzada desde equilibrio '{nodo_equilibrio}': {a} ↔ {b}")

    if verbose:
        print(f"⚖️ Dualidades reforzadas desde equilibrios: {len(dualidades_reforzadas)}")

    return dualidades_reforzadas

def similitud_coseno(vec1, vec2):
    """
    Devuelve la similitud coseno entre dos embeddings.
    Soporta tanto tensores de PyTorch como arrays de numpy/listas.
    """
    if vec1 is None or vec2 is None:
        return 0.0

    # Asegurarnos de que son tensores 1D
    if not torch.is_tensor(vec1):
        vec1 = torch.tensor(vec1)
    if not torch.is_tensor(vec2):
        vec2 = torch.tensor(vec2)

    # Aplanar por si vienen con más dimensiones
    vec1 = vec1.view(-1)
    vec2 = vec2.view(-1)

    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()


def detectar_nuevas_dualidades_por_embeddingsANTIGUO(G, modelo, umbral_min=0.3, umbral_max=0.7, top_k=20):
    """
    Busca dualidades nuevas usando embeddings:
    - Considera solo nodos de tipo None o 'concepto'
    - Para cada nodo, mira sus vecinos semánticamente cercanos
    - Proponemos dualidad si:
      * tienen un equilibrio en común
      * o su vecindario es "espejo" (comparten muchos vecinos)
      * y su similitud cae en una banda intermedia (no sinónimos, no totalmente alejados)
    """
    nodos = [n for n in G.nodes() if G.nodes[n].get("tipo") in (None, "concepto")]
    if not nodos:
        return []

    # Precalcular embeddings
    embeddings = {n: obtener_embedding(n, modelo) for n in nodos}
    candidatos = []

    for i, a in enumerate(nodos):
        emb_a = embeddings[a]
        if emb_a is None:
            continue

        # comparar con unos pocos nodos (top_k) para no explotar
        for b in nodos[i+1 : i+1+top_k]:
            emb_b = embeddings[b]
            if emb_b is None:
                continue

            sim = similitud_coseno(emb_a, emb_b)

            if sim < umbral_min or sim > umbral_max:
                continue  # ni demasiado lejos ni demasiado cerca (sinónimo)

            # pequeña señal estructural: vecindarios
            vecinos_a = set(G.successors(a)) | set(G.predecessors(a))
            vecinos_b = set(G.successors(b)) | set(G.predecessors(b))
            inter = len(vecinos_a & vecinos_b)

            if inter == 0:
                continue

            score = sim * (1 + inter / 5.0)  # mezcla semántico + estructura
            candidatos.append((score, a, b))

    # ordenar por score y crear dualidades para los mejores
    candidatos.sort(reverse=True)
    creadas = []
    for score, a, b in candidatos[:50]:  # por ejemplo
        if G.nodes[a].get("es_dualidad") or G.nodes[b].get("es_dualidad"):
            continue
        conectar_dualidad_con_equilibrio(a, b, G)
        creadas.append((a, b, score))
        print(f"🧲 Nueva dualidad embedding: {a} ↔ {b} (score={score:.2f})")

    return creadas


RUTA_DUALIDADES_CANDIDATAS = "json/dualidades_candidatas.json"
def registrar_dualidad_candidata(a, b, score):
    """
    Guarda las dualidades propuestas por embeddings en un JSON
    para poder auditarlas o promoverlas después a dualidades reales.
    """
    os.makedirs("json", exist_ok=True)

    if os.path.exists(RUTA_DUALIDADES_CANDIDATAS):
        with open(RUTA_DUALIDADES_CANDIDATAS, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append({"a": a, "b": b, "score": score})

    with open(RUTA_DUALIDADES_CANDIDATAS, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def detectar_nuevas_dualidades_por_embeddings(G, modelo, max_candidatos=30):
    """
    Busca pares de conceptos relacionados mediante embeddings.
    - Si parecen dualidad estructural (núcleo común + adjetivos opuestos),
      los asciende automáticamente a dualidad usando agregar_dualidad.
    - Si no, crea aristas de tipo 'relacion_fuerte' para enriquecer la red
      sin contaminar el espacio sagrado de dualidades.
    """
    nodos = [n for n in G.nodes() if G.nodes[n].get("tipo") in (None, "concepto")]
    if not nodos:
        print("⚠️ No hay nodos candidatos para embeddings.")
        return []

    # Precalcular embeddings
    embeddings = {}
    for n in nodos:
        emb = obtener_embedding(n, modelo)
        if emb is not None:
            embeddings[n] = emb

    candidatos = []

    lista_nodos = list(embeddings.keys())
    for i, a in enumerate(lista_nodos):
        emb_a = embeddings[a]
        # limitado para no explotar
        for b in lista_nodos[i+1 : i+1+50]:
            emb_b = embeddings[b]
            sim = similitud_coseno(emb_a, emb_b)

            # banda intermedia: ni sinónimos perfectos ni totalmente alejados
            if sim < 0.25 or sim > 0.8:
                continue

            vecinos_a = set(G.successors(a)) | set(G.predecessors(a))
            vecinos_b = set(G.successors(b)) | set(G.predecessors(b))
            inter = len(vecinos_a & vecinos_b)

            if inter == 0:
                continue

            score = sim * (1 + inter / 5.0)
            candidatos.append((score, a, b))

    candidatos.sort(reverse=True)
    seleccionados = candidatos[:max_candidatos]

    resultados = []

    for score, a, b in seleccionados:
        if a == b:
            continue

        if G.has_edge(a, b) and G[a][b].get("tipo") in ("dualidad", "relacion_fuerte"):
            # ya existe algo entre ellos
            continue

        # 1️⃣ Intentar verlo como dualidad estructural automática
        if es_dualidad_estructural_automatica(a, b):
            agregar_dualidad(G, a, b)
            resultados.append(("dualidad", a, b, score))
            print(f"🧬 Dualidad AUTO por embeddings+estructura: {a} ↔ {b} (score={score:.2f})")
        else:
            # 2️⃣ Si no, solo relación fuerte
            G.add_edge(a, b, tipo="relacion_fuerte", weight=2.0)
            G.add_edge(b, a, tipo="relacion_fuerte", weight=2.0)
            resultados.append(("relacion_fuerte", a, b, score))
            print(f"🧲 Relación fuerte por embeddings: {a} ↔ {b} (score={score:.2f})")

    print("🧠 Resumen embeddings:")
    print("   Dualidades auto-creadas:",
          sum(1 for t, *_ in resultados if t == "dualidad"))
    print("   Relaciones fuertes creadas:",
          sum(1 for t, *_ in resultados if t == "relacion_fuerte"))

    return resultados


def detectar_nuevas_dualidades_por_embeddingsANTIGUO2(G, modelo, max_candidatos=30):
    """
    Busca pares de conceptos muy relacionados mediante embeddings,
    pero NO los marca directamente como dualidad.
    En su lugar:
      - crea aristas de tipo 'relacion_fuerte'
      - registra las parejas en un JSON para revisión.
    """
    nodos = [n for n in G.nodes() if G.nodes[n].get("tipo") in (None, "concepto")]
    if not nodos:
        print("⚠️ No hay nodos candidatos para embeddings.")
        return []

    # Precalcular embeddings
    embeddings = {}
    for n in nodos:
        emb = obtener_embedding(n, modelo)
        if emb is not None:
            embeddings[n] = emb

    candidatos = []

    lista_nodos = list(embeddings.keys())
    for i, a in enumerate(lista_nodos):
        emb_a = embeddings[a]
        # limitado para no explotar
        for b in lista_nodos[i+1 : i+1+50]:
            emb_b = embeddings[b]
            sim = similitud_coseno(emb_a, emb_b)

            # Banda intermedia: ni sinónimo clavado, ni totalmente alejados
            if sim < 0.25 or sim > 0.75:
                continue

            # Pequeño filtro léxico: saltar singular/plural, misma raíz obvia
            if a.lower() in b.lower() or b.lower() in a.lower():
                continue

            # Algo de estructura: compartir vecinos
            vecinos_a = set(G.successors(a)) | set(G.predecessors(a))
            vecinos_b = set(G.successors(b)) | set(G.predecessors(b))
            inter = len(vecinos_a & vecinos_b)
            if inter == 0:
                continue

            score = sim * (1 + inter / 5.0)
            candidatos.append((score, a, b))

    candidatos.sort(reverse=True)
    seleccionados = candidatos[:max_candidatos]

    relaciones_creadas = []
    for score, a, b in seleccionados:
        # Creamos relación fuerte, NO dualidad
        if not G.has_edge(a, b):
            G.add_edge(a, b, tipo="relacion_fuerte", weight=2.0)
        if not G.has_edge(b, a):
            G.add_edge(b, a, tipo="relacion_fuerte", weight=2.0)

        registrar_dualidad_candidata(a, b, score)
        relaciones_creadas.append((a, b, score))
        print(f"🧲 Relación fuerte por embeddings: {a} ↔ {b} (score={score:.2f})")

    print(f"🧠 Relaciones fuertes por embeddings creadas: {len(relaciones_creadas)}")
    return relaciones_creadas


def pipeline_dualidades_auto(G, modelo_embeddings):
    """
    Pipeline completo de detección de dualidades:
    1. A partir de dualidades base mínimas.
    2. A partir de equilibrios presentes.
    3. A partir de WordNet / memoria (buscar_dualidades_faltantes).
    4. A partir de embeddings + estructura.
    """
    print("1️⃣ Refuerzo desde equilibrios...")
    reforzar_dualidades_desde_equilibrios(G)

    print("2️⃣ Dualidades faltantes (WordNet/memoria)...")
    buscar_dualidades_faltantes(G)

    print("3️⃣ Dualidades nuevas por embeddings...")
    detectar_nuevas_dualidades_por_embeddings(G, modelo_embeddings)


def evaluar_dualidades_por_estructura(G, umbral_dsl=0.1, distancia_max=2, peso_reduccion=0.25, peso_minimo=0.75):
    """
    Revisa dualidades no base según su estructura:
    - dsl: solapamiento de vecindarios (como un Jaccard)
    - distancia en el grafo
    Debilita o elimina dualidades que no encajan estructuralmente.
    """
    dualidades_revisadas = 0
    dualidades_eliminadas = 0
    dualidades_debilitadas = 0

    dualidades_a_revisar = [
        (a, b) for a, b, datos in G.edges(data=True)
        if datos.get("tipo") == "dualidad"
        and (a, b) not in dualidades_base_protegidas
        and (b, a) not in dualidades_base_protegidas
    ]

    dualidades_a_eliminar = []
    dualidades_a_debilitar = []

    for nodo_a, nodo_b in dualidades_a_revisar:
        vecinos_a = set(G.neighbors(nodo_a))
        vecinos_b = set(G.neighbors(nodo_b))

        vecinos_comunes = vecinos_a & vecinos_b
        total_vecinos = vecinos_a | vecinos_b
        dsl = len(vecinos_comunes) / len(total_vecinos) if total_vecinos else 0.0

        try:
            distancia = nx.shortest_path_length(G, nodo_a, nodo_b)
        except nx.NetworkXNoPath:
            distancia = float('inf')

        peso_actual = G[nodo_a][nodo_b].get("weight", 1.0)

        if dsl < umbral_dsl and distancia > distancia_max:
            nuevo_peso = peso_actual - peso_reduccion
            if nuevo_peso <= peso_minimo:
                dualidades_a_eliminar.append((nodo_a, nodo_b))
            else:
                dualidades_a_debilitar.append((nodo_a, nodo_b, nuevo_peso))

    # Aplicar eliminaciones
    for nodo_a, nodo_b in dualidades_a_eliminar:
        if G.has_edge(nodo_a, nodo_b):
            G.remove_edge(nodo_a, nodo_b)
        if G.has_edge(nodo_b, nodo_a):
            G.remove_edge(nodo_b, nodo_a)
        dualidades_eliminadas += 1
        print(f"❌ Dualidad eliminada: '{nodo_a}' ↔ '{nodo_b}'")

    # Aplicar debilitaciones
    for nodo_a, nodo_b, nuevo_peso in dualidades_a_debilitar:
        if G.has_edge(nodo_a, nodo_b):
            G[nodo_a][nodo_b]['weight'] = nuevo_peso
        if G.has_edge(nodo_b, nodo_a):
            G[nodo_b][nodo_a]['weight'] = nuevo_peso
        dualidades_debilitadas += 1
        print(f"⚠️ Dualidad debilitada: '{nodo_a}' ↔ '{nodo_b}', nuevo peso={nuevo_peso}")

    print(f"🔍 Dualidades revisadas: {len(dualidades_a_revisar)}")
    print(f"🗑️  Dualidades eliminadas: {dualidades_eliminadas}")
    print(f"🔻 Dualidades debilitadas: {dualidades_debilitadas}")

    # Asegurar máximo 1 dualidad no protegida por nodo
    asegurar_dualidades_unicas(G, dualidades_base_protegidas)

    return G


def asegurar_dualidades_unicas(G, dualidades_base_protegidas):
    """
    Asegura que cada nodo tenga como máximo una dualidad no protegida.
    Las dualidades base (espacio-tiempo, etc.) se respetan siempre.
    """
    for nodo in list(G.nodes()):
        # Vecinos conectados por aristas marcadas como 'dualidad'
        dualidades_actuales = [
            vecino for vecino in G.neighbors(nodo)
            if G[nodo][vecino].get('tipo') == 'dualidad'
        ]

        # Excluir dualidades base protegidas (ejes fundamentales)
        dualidades_no_protegidas = [
            vecino for vecino in dualidades_actuales
            if (nodo, vecino) not in dualidades_base_protegidas
            and (vecino, nodo) not in dualidades_base_protegidas
        ]

        if len(dualidades_no_protegidas) > 1:
            print(f"🚨 Nodo '{nodo}' tiene múltiples dualidades no protegidas: {dualidades_no_protegidas}. Corrigiendo...")
            # Conservar solo la dualidad con mayor peso, eliminar el resto
            dualidades_pesos = [
                (vecino, G[nodo][vecino].get('weight', 1.0))
                for vecino in dualidades_no_protegidas
            ]
            dualidades_pesos.sort(key=lambda x: x[1], reverse=True)

            dualidad_principal = dualidades_pesos[0][0]
            dualidades_a_eliminar = [vecino for vecino, _ in dualidades_pesos[1:]]

            for vecino in dualidades_a_eliminar:
                if G.has_edge(nodo, vecino):
                    G.remove_edge(nodo, vecino)
                if G.has_edge(vecino, nodo):
                    G.remove_edge(vecino, nodo)
                print(f"❌ Dualidad extra eliminada: '{nodo}' ↔ '{vecino}'. Se conserva '{nodo}' ↔ '{dualidad_principal}'.")


# Función principal para detectar el hipercubo conceptual
def detectar_hipercubo_conceptual(G):
    triadas_estructura = {
        "vertical_1": ["arriba", "abajo", "centro_vertical"],
        "vertical_2": ["izquierda", "derecha", "centro_horizontal"],
        "vertical_3": ["delante", "detrás", "centro_frontal"],
        "horizontal_1": ["arriba", "izquierda", "delante"],
        "horizontal_2": ["abajo", "derecha", "detrás"],
        "horizontal_3": ["centro_vertical", "centro_horizontal", "centro_frontal"]
    }

    triadas_detectadas = []

    for nombre, trio in triadas_estructura.items():
        if all(n in G.nodes for n in trio):
            conexiones = sum(1 for i in range(3) for j in range(i+1, 3) if G.has_edge(trio[i], trio[j]) or G.has_edge(trio[j], trio[i]))
            if conexiones >= 2:  # al menos dos conexiones entre los tres
                triadas_detectadas.append(trio)

    if len(triadas_detectadas) == 6:
        print("✅ Todas las triadas estructurales del hipercubo han sido detectadas.")
        if "observador" not in G:
            G.add_node("observador", tipo="abstracto", nivel_conceptual=4)
        for nodo in {"centro_vertical", "centro_horizontal", "centro_frontal"}:
            if G.has_node(nodo):
                G.add_edge("observador", nodo, weight=2.5)
        for extremo in ["arriba", "abajo", "izquierda", "derecha", "delante", "detrás"]:
            if G.has_node(extremo):
                G.add_edge("observador", extremo, weight=1.8)

        # Guardar registro
        estructura = {
            "hipercubo_detectado": True,
            "nodo_central": "observador",
            "triadas": triadas_detectadas
        }
        with open("estructura_hipercubo.json", "w", encoding="utf-8") as f:
            json.dump(estructura, f, ensure_ascii=False, indent=2)
        print("🧠 Hipercubo registrado en 'estructura_hipercubo.json'")

        # Visualización
        subG = G.subgraph(set(sum(triadas_detectadas, [])) | {"observador"}).copy()
        net = Network(height="800px", width="100%", directed=True)

        for nodo in subG.nodes():
            color = color_node(nodo, G)
            net.add_node(nodo, label=nodo, color=color, title=nodo)

        for u, v in subG.edges():
            color = color_edge(u, v, G)
            net.add_edge(u, v, color=color, width=subG.edges[u, v].get("weight", 1.5))

        net.write_html("hipercubo_conceptual.html")
        print("🌐 Visualización generada en 'hipercubo_conceptual.html'")
        return True
    else:
        print("⚠️ No se detectó el hipercubo completo. Triadas encontradas:", len(triadas_detectadas))
        return False

def es_relacion_fuerte(G, u, v, umbral_peso=2.0):
#    """
#    Consideramos 'relación fuerte' si:
#    - existe arista u->v
#    - y su peso es alto o el tipo es 'relacion_fuerte'
#    """
    if not G.has_edge(u, v):
        return False
    datos = G[u][v]
    if datos.get("tipo") == "relacion_fuerte":
        return True
    peso = datos.get('weight', 0.0)
    return peso >= umbral_peso
    
def es_relacion_fuerteANTIGUO(G, u, v, umbral_peso=2.0):
    """
    Consideramos 'relación fuerte' si:
    - existe arista u->v
    - y su peso es alto o el tipo es 'relacion_fuerte'
    """
    if not G.has_edge(u, v):
        return False
    datos = G[u][v]
    if datos.get("tipo") == "relacion_fuerte":
        return True
    peso = datos.get("weight", 0.0)
    return peso >= umbral_peso


def detectar_sistemas_posicionales(G, min_dim=3):
    """
    Detecta patrones del tipo:

        centro -- A
              \\-- B
              \\-- C

    donde A,B,C,... son vecinos del 'centro' y están fuertemente conectados entre sí.
    Ejemplo típico: 'posición' con {latitud, longitud, altitud}.

    Devuelve una lista de estructuras:
      {
        "centro": nodo_central,
        "coordenadas": [a, b, c, ...]
      }
    """
    sistemas = []

    for centro, datos in G.nodes(data=True):
        if centro in ("IA_m", "Subconsciente", "Subconsciente_semantico"):
            continue

        vecinos = set(G.predecessors(entro)) | set(G.successors(centro))
        vecinos = [v for v in vecinos if v in G]  # por si acaso
        if len(vecinos) < min_dim:
            continue

        # Para cada combinación de min_dim vecinos, comprobamos si están bastante conectados entre sí
        for triple in combinations(vecinos, min_dim):
            coords = list(triple)
            pares_fuertes = 0
            total_pares = 0
            from itertools import combinations as _comb
            for x, y in _comb(coords, 2):
                total_pares += 1
                if es_relacion_fuerte(G, x, y) or es_relacion_fuerte(G, y, x):
                    pares_fuertes += 1

            # Exigimos al menos 2 conexiones fuertes dentro de la tríada
            if pares_fuertes < 2:
                continue

            sistema = {
                "centro": centro,
                "coordenadas": coords
            }
            if sistema not in sistemas:
                sistemas.append(sistema)

    return sistemas
    
def detectar_sistemas_posicionalesANTIGUO(G, min_dim=3):
#    """
#    Detecta patrones del tipo:

#      centro -- A
#             \- B
#             \- C

#    donde A,B,C están fuertemente conectados entre sí.
#    Ejemplo típico: posición y {latitud, longitud, altitud}.

#    Devuelve una lista de estructuras:
#      {
#        "centro": nodo_central,
#        "coordenadas": [a, b, c, ...]
#      }
#    """
    sistemas = []

    for centro, datos in G.nodes(data=True):
        # Podrías refinar filtrando por tipo, p.ej. solo conceptos o 'posicion'
        if centro in ("IA_m",):
            continue

        vecinos = set(G.successors(centro)) | set(G.predecessors(centro))
        if len(vecinos) < min_dim:
            continue

        vecinos = list(vecinos)

        # Buscamos tríadas altamente cohesionadas entre sí
        for triple in combinations(vecinos, min_dim):
            a, b, c = triple

            # Comprobamos que cada par del triple está fuertemente conectado
            pares_fuertes = 0
            for x, y in combinations(triple, 2):
                if es_relacion_fuerte(G, x, y) or es_relacion_fuerte(G, y, x):
                    pares_fuertes += 1

            # En un triángulo completo hay 3 pares; pedimos al menos 2
            if pares_fuertes < 2:
                continue

            # Si hemos llegado aquí, tenemos un sistema candidato
            sistema = {
                "centro": centro,
                "coordenadas": list(triple)
            }
            if sistema not in sistemas:
                sistemas.append(sistema)

    return sistemas


def agrupar_sistemas_posicionales(sistemas):
    """
    Agrupa sistemas posicionales que tienen el mismo centro y el mismo conjunto de coordenadas,
    ignorando el orden.
    """
    grupos = {}
    for s in sistemas:
        centro = s["centro"]
        coords = tuple(sorted(s["coordenadas"]))
        clave = (centro, coords)
        grupos[clave] = s  # si se repite, lo pisamos, nos da igual

    return list(grupos.values())
    
def similitud_lexica(a, b, min_long=3):
    """
    Similitud léxica basada en sufijos/prefijos comunes de longitud mínima.
    No es perfecto, pero sirve para ver cosas como 'latitud/longitud/altitud'
    o 'tensión negativa / flujo negativo / desplazamiento negativo'.
    """
    a = a.lower()
    b = b.lower()
    max_comun = 0

    # sufijos
    for i in range(len(a)):
        suf_a = a[i:]
        if len(suf_a) < min_long:
            continue
        if suf_a in b:
            max_comun = max(max_comun, len(suf_a))

    # prefijos
    for i in range(len(a), 0, -1):
        pref_a = a[:i]
        if len(pref_a) < min_long:
            continue
        if pref_a in b:
            max_comun = max(max_comun, len(pref_a))

    return max_comun

def coherencia_lexica_triple(coords, min_long=3):
    a, b, c = coords
    s_ab = similitud_lexica(a, b, min_long)
    s_ac = similitud_lexica(a, c, min_long)
    s_bc = similitud_lexica(b, c, min_long)
    return (s_ab + s_ac + s_bc) / 3.0

def guardar_estructuras_posicionales(estructuras, archivo="json/estructuras_posicionales.json"):
    """
    Guarda la lista de estructuras posicionales detectadas en un JSON.
    Cada elemento es un dict con 'centro' y 'coordenadas'.
    Si ya existe el archivo, acumula y evita duplicados.
    """
    existentes = []

    if os.path.exists(archivo):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                existentes = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ {archivo} corrupto o vacío, se sobrescribirá.")

    # Unificar por (centro, tuple(sorted(coordenadas)))
    def clave(e):
        return (e.get("centro"), tuple(sorted(e.get("coordenadas", []))))

    mapa = {clave(e): e for e in existentes}
    for e in estructuras:
        k = clave(e)
        if k not in mapa:
            mapa[k] = e

    final = list(mapa.values())

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"📁 Estructuras posicionales actualizadas y guardadas en {archivo} ({len(final)} únicas)")


def filtrar_estructuras_dimensionales(G, sistemas, min_sim_lex=3.0):
    """
    De la lista de sistemas posicionales detectados, selecciona aquellos
    que tienen coherencia léxica alta entre coordenadas y un centro geométricamente razonable.
    """
    candidatos = []
    sistemas_unicos = agrupar_sistemas_posicionales(sistemas)

    for s in sistemas_unicos:
        centro = s["centro"]
        coords = s["coordenadas"]
        if len(coords) != 3:
            continue

        datos_centro = G.nodes[centro]
        tipo_centro = datos_centro.get("tipo", "concepto")
        nivel_centro = datos_centro.get("nivel_conceptual", 0)

        # Solo consideramos ciertos tipos y niveles
        if nivel_centro > 3:
            continue

        # Coherencia léxica
        score_lex = coherencia_lexica_triple(coords)
        if score_lex < min_sim_lex:
            continue

        candidatos.append({
            "centro": centro,
            "coordenadas": coords,
            "tipo_centro": tipo_centro,
            "nivel_centro": nivel_centro,
            "coherencia_lexica": score_lex,
        })

    return candidatos

def detectar_triadas_lineales(G):
    """
    Detecta estructuras del tipo:
      extremo1 ↔ extremo2  (dualidad, aristas bidireccionales tipo 'dualidad')
      equilibrio           (equilibrio entre ambos)
      sintesis             (nodo que conecta con los tres como 'sintesis')

    Devuelve una lista de dicts:
      {
        "extremos": (a, b),
        "equilibrio": e,
        "sintesis": s
      }
    """
    estructuras = []

    # Candidatos a síntesis: nodos marcados como síntesis/nodo_superior
    candidatos_sintesis = [
        n for n, d in G.nodes(data=True)
        if d.get("tipo") in ("sintesis", "nodo_superior") or d.get("es_sintesis")
    ]

    for s in candidatos_sintesis:
        vecinos = set(G.predecessors(s)) | set(G.successors(s))
        if len(vecinos) < 3:
            continue

        for a, b, e in combinations(vecinos, 3):
            # 1) extremos a↔b deben estar enlazados como dualidad en ambos sentidos
            if not (G.has_edge(a, b) and G.has_edge(b, a)):
                continue
            ed_ab = G[a][b]
            ed_ba = G[b][a]
            if not (ed_ab.get("tipo") == "dualidad" and ed_ba.get("tipo") == "dualidad"):
                continue

            # 2) equilibrio e debe estar conectado a ambos extremos como 'equilibrio'
            if not (G.has_edge(e, a) and G.has_edge(e, b)):
                continue
            ed_ea = G[e][a]
            ed_eb = G[e][b]
            if not (ed_ea.get("tipo") == "equilibrio" and ed_eb.get("tipo") == "equilibrio"):
                continue

            # 3) síntesis s debe conectar con a,b,e como 'sintesis'
            ok_sintesis = True
            for x in (a, b, e):
                if not (G.has_edge(s, x) and G[s][x].get("tipo") == "sintesis"):
                    ok_sintesis = False
                    break

            if not ok_sintesis:
                continue

            estructura = {
                "extremos": tuple(sorted([a, b])),
                "equilibrio": e,
                "sintesis": s,
            }
            if estructura not in estructuras:
                estructuras.append(estructura)

    return estructuras


def detectar_triadas_linealesANTIGUO(G):
    """
    Detecta estructuras del tipo:
      extremo1 ↔ extremo2  (dualidad)
      equilibrio           (equilibrio entre ambos)
      sintesis             (se conecta a los 3 como síntesis)

    Devuelve una lista de dicts:
      {
        "extremos": (a, b),
        "equilibrio": e,
        "sintesis": s
      }
    """
    estructuras = []

    # Candidatos a síntesis: nodos marcados como síntesis/nodo_superior
    candidatos_sintesis = [
        n for n, d in G.nodes(data=True)
        if d.get("tipo") in ("sintesis", "nodo_superior") or d.get("es_sintesis")
    ]

    for s in candidatos_sintesis:
        # Vecinos del nodo síntesis (entrantes + salientes)
        vecinos = set(G.predecessors(s)) | set(G.successors(s))
        if len(vecinos) < 3:
            continue

        # Probamos todas las combinaciones de 3 vecinos
        for a, b, e in combinations(vecinos, 3):
            # 1) ¿a y b forman dualidad bidireccional?
            if not (G.has_edge(a, b) and G.has_edge(b, a)):
                continue
            ed_ab = G[a][b]
            ed_ba = G[b][a]
            if not (ed_ab.get("tipo") == "dualidad" and ed_ba.get("tipo") == "dualidad"):
                continue

            # 2) ¿e está conectado a a y b como equilibrio?
            if not (G.has_edge(e, a) and G.has_edge(e, b)):
                continue
            ed_ea = G[e][a]
            ed_eb = G[e][b]
            if not (ed_ea.get("tipo") == "equilibrio" and ed_eb.get("tipo") == "equilibrio"):
                continue

            # 3) ¿s está conectado a a, b y e como síntesis?
            ok_sintesis = True
            for x in (a, b, e):
                if not (G.has_edge(s, x) and G[s][x].get("tipo") == "sintesis"):
                    ok_sintesis = False
                    break
            if not ok_sintesis:
                continue

            estructura = {
                "extremos": tuple(sorted([a, b])),
                "equilibrio": e,
                "sintesis": s,
            }
            # Evitar duplicados
            if estructura not in estructuras:
                estructuras.append(estructura)

    return estructuras



#    return triadas_creadas
def detectar_triadas_extremas(G, crear_nodo_sintesis=True, ruta_json="triadas_horizontales.json", ruta_html="triadas_extremas.html"):
    triadas_creadas = []

    triadas = [
        ("arriba", "izquierda", "delante", "triada_extrema_positiva"),
        ("abajo", "derecha", "detrás", "triada_extrema_negativa"),
    ]

    for a, b, c, nombre in triadas:
        if all(n in G.nodes() for n in [a, b, c]):
            for u, v in [(a, b), (b, c), (c, a)]:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=1.8)
                if not G.has_edge(v, u):
                    G.add_edge(v, u, weight=1.8)

            if crear_nodo_sintesis:
                # Aquí llamamos a la función crear_nodo_sintesis_triada
                nombre_sintesis = crear_nodo_sintesis_triada(a, b, c, G, modelo_embeddings)
                if nombre_sintesis:
                    triadas_creadas.append([a, b, c, nombre_sintesis])
                else:
                    triadas_creadas.append([a, b, c])  # Si no se pudo crear el nodo de síntesis
            else:
                triadas_creadas.append([a, b, c])

    if triadas_creadas:
        print(f"✅ Triadas extremas detectadas: {len(triadas_creadas)}")

        # 📝 Guardar triadas en JSON sin duplicados
        if os.path.exists(ruta_json):
            try:
                with open(ruta_json, "r", encoding="utf-8") as f:
                    existentes = json.load(f)
            except:
                existentes = []
        else:
            existentes = []

        todas = existentes + triadas_creadas
        todas_unicas = list({tuple(sorted(t[:3])) for t in todas})
        todas_unicas = [list(t) for t in todas_unicas]

        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(todas_unicas, f, ensure_ascii=False, indent=2)
        print(f"📁 Triadas horizontales actualizadas en {ruta_json}")

        # 🌐 Visualizar
        visualizar_triadas_horizontales(G, todas_unicas, nombre=ruta_html)

    else:
        print("⚠️ No se detectaron triadas extremas (faltan nodos).")

    return triadas_creadas

def detectar_triadas_horizontales(G):
    horizontales = []
    candidatos = [n for n in G.nodes() if G.nodes[n].get("tipo") == "dualidad"]

    for i in range(len(candidatos)):
        for j in range(i + 1, len(candidatos)):
            a, b = candidatos[i], candidatos[j]
            # Ver si comparten conexiones similares o están relacionados semánticamente
            if G.has_edge(a, b) or G.has_edge(b, a):
                # Buscar si tienen un nodo común superior (ej. "observador", o embedding común)
                comunes = set(G.successors(a)) & set(G.successors(b))
                for c in comunes:
                    if G.nodes[c].get("tipo") in ("abstracto", "emergente") or "observador" in c.lower():
                        horizontales.append((a, b, c))
                        print(f"🧩 Tríada horizontal detectada: {a} – {b} → {c}")
    return horizontales

def detectar_triangulos_equilibrio(G):
    triangulos = []
    for nodo in G.nodes():
        if G.nodes[nodo].get("tipo") != "equilibrio":
            continue  # solo consideramos nodos equilibrio como centro

        vecinos = list(G.neighbors(nodo))
        for i in range(len(vecinos)):
            for j in range(i + 1, len(vecinos)):
                a, b = vecinos[i], vecinos[j]
                if (
                    G.has_edge(a, b) and G.has_edge(b, a)
                    and G.nodes[a].get("tipo") == "dualidad"
                    # G.nodes[a].get("tipo") in ("dualidad", "concepto")
                    and G.nodes[b].get("tipo") in ("dualidad", "concepto")
                    and G.nodes[a].get("tipo") in ("dualidad", "concepto")
                    and G.nodes[b].get("tipo") in ("dualidad", "concepto")
                ):
                    triangulos.append((a, b, nodo))
    return triangulos
    
def guardar_triangulos(triangulos, archivo="json/triangulos_fractales.json"):
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(triangulos, f, ensure_ascii=False, indent=4)

def detectar_estructura_emergente_adaptativaANTIGUA(G, min_triangulos=3, umbral_refuerzo=0.75, umbral_debilitamiento=0.4):
    """
    Detecta tríadas de dualidades con un nodo común de equilibrio.
    Solo crea nodos emergentes si se detecta un concepto superior con suficiente similitud.
    """
    triangulos = detectar_triangulos_equilibrio(G)
    mapa_equilibrios = defaultdict(list)

    for a, b, equilibrio in triangulos:
        mapa_equilibrios[equilibrio].append((a, b))

    conceptos_emergentes = []
    # 🩺 Revisar emergentes antiguos sin concepto_superior para completarlos
    for nodo in list(G.nodes()):
        if G.nodes[nodo].get("tipo") == "emergente" and not G.nodes[nodo].get("concepto_superior"):
            embedding_emergente = obtener_embedding(nodo, modelo_embeddings)
            candidatos = list(G.nodes())
            embeddings_red = obtener_embeddings_lista(candidatos, modelo_embeddings)
    
            sims = util.pytorch_cos_sim(embedding_emergente, embeddings_red)[0]
            max_sim, idx = torch.max(sims, 0)
    
            if max_sim.item() >= 0.75:
                nodo_superior = candidatos[idx.item()]
    
                tipo = G.nodes[nodo].get("tipo", "")
                if tipo == "emergente":
                    return False  # no expandir emergentes directamente
    
                nuevo_nombre = nodo_superior
                if nuevo_nombre not in G:
                    nx.relabel_nodes(G, {nodo: nuevo_nombre}, copy=False)
                    G.nodes[nuevo_nombre]["concepto_superior"] = nodo_superior
                    G.add_edge(nodo_superior, nuevo_nombre, weight=2.2)
                    print(f"🔁 Nodo emergente renombrado: {nodo} → {nuevo_nombre}")
                    
    for equilibrio, dualidades in mapa_equilibrios.items():
        # Si el nodo no tiene nivel, se asume 0
        nivel_equilibrio = G.nodes[equilibrio].get("nivel_emergencia", 0)
        # Si el equilibrio ya es demasiado profundo, no seguimos
        if nivel_equilibrio >= NIVEL_MAX_EMERGENCIA:
            print(f"⛔ Nodo '{equilibrio}' ya tiene nivel {nivel_equilibrio}, se omite para evitar recursión.")
            continue
        # ❗️Permitir solo hasta un nivel máximo
        # 🔎 Evita recursión solo si el equilibrio ya es emergente y no tiene nivel
        nivel_equilibrio = G.nodes[equilibrio].get("nivel_conceptual", 0)
        if G.nodes[equilibrio].get("tipo") == "emergente" and G.nodes[equilibrio].get("nivel_conceptual", 0) < 3:
            print(f"⛔ Nodo '{equilibrio}' es emergente sin nivel suficiente, se descarta.")
            continue
        if len(dualidades) >= min_triangulos:
            polos = set()
            for a, b in dualidades:
                polos.update([a, b])

            # Obtener embeddings desde caché
            embeddings_polos = obtener_embeddings_lista(list(polos), modelo_embeddings)
            embedding_promedio = embeddings_polos.mean(dim=0)

            # Buscar el nodo más cercano como concepto superior
            candidatos = list(G.nodes())
            embeddings_red = obtener_embeddings_lista(candidatos, modelo_embeddings)

            sims = util.pytorch_cos_sim(embedding_promedio, embeddings_red)[0]
            max_sim, idx = torch.max(sims, 0)

            if max_sim.item() >= 0.75:
                nodo_superior = candidatos[idx.item()]
                # ❌ Evitar conceptos por encima del nivel 4
                if G.nodes[nodo_superior].get("nivel_conceptual", 0) >= 4:
                    print(f"⛔ Nivel máximo alcanzado con '{nodo_superior}' (nivel 4). No se generará emergente superior.")
                    continue

                if G.nodes[nodo].get("tipo") == "emergente":
                    continue  # no expandir emergentes directamente

                nombre_emergente = f"{nodo_superior}_sintesis_{len(conceptos_emergentes)}"
                if nuevo_nodo in G:
                    print(f"⚠️ Nodo '{nuevo_nodo}' ya existe, se omite para evitar sobrescribir.")
                    continue
                elif nombre_emergente not in G:

                    if not isinstance(dualidades, list) or not all(isinstance(d, tuple) and len(d) == 2 for d in dualidades):
                        print(f"⚠️ Formato incorrecto en dualidades para '{equilibrio}', se omite este emergente.")
                        continue
                    G.add_node(nombre_emergente, tipo="emergente", dualidades=dualidades)
                    nivel_base = G.nodes[nodo_superior].get("nivel_conceptual")
                    G.nodes[nombre_emergente]["nivel_conceptual"] = nivel_base + 1 if nivel_base is not None else 3
                    #G.nodes[nombre_emergente]["nivel_conceptual"] = G.nodes[nodo_superior].get("nivel_conceptual", 0) + 1
                    G.nodes[nombre_emergente]["nivel_emergencia"] = nivel_equilibrio + 1
                    G.add_edge(nombre_emergente, equilibrio, weight=2.5)
                    G.add_edge(nodo_superior, nombre_emergente, weight=2.2)
                    G.nodes[nombre_emergente]["concepto_superior"] = nodo_superior
                    print(f"🌱 Nodo emergente creado: {nombre_emergente} (nivel {G.nodes[nombre_emergente]['nivel_conceptual']})")
                    for a, b in dualidades:
                        emb_a = obtener_embedding(a, modelo_embeddings)
                        emb_b = obtener_embedding(b, modelo_embeddings)

                        similitud = util.pytorch_cos_sim(emb_a, emb_b).item()

                        if similitud >= umbral_refuerzo:
                            G.add_edge(a, b, weight=2.0)
                            G.add_edge(b, a, weight=2.0)
                        elif similitud <= umbral_debilitamiento:
                            if G.has_edge(a, b): G.remove_edge(a, b)
                            if G.has_edge(b, a): G.remove_edge(b, a)

                        G.add_edge(nombre_emergente, a, weight=1.8)
                        G.add_edge(nombre_emergente, b, weight=1.8)
                        G.add_edge(equilibrio, a, weight=1.2)
                        G.add_edge(equilibrio, b, weight=1.2)

                    conceptos_emergentes.append((nombre_emergente, equilibrio, dualidades))
            else:
            
                print(f"⚠️ No se creó emergente para '{equilibrio}' (similitud máxima: {max_sim.item():.2f})")
                
    return conceptos_emergentes

def detectar_estructura_emergente_adaptativa(G, min_triangulos=3, umbral_refuerzo=0.75, umbral_debilitamiento=0.4):
    """
    Detecta estructuras emergentes a partir de tríadas (triángulos) que comparten
    un mismo nodo de equilibrio.

    Lógica simplificada y robusta:
    - Busca triángulos equilibrio–a–b (detectados por detectar_triangulos_equilibrio).
    - Agrupa por nodo de equilibrio.
    - Si un equilibrio tiene al menos `min_triangulos` triángulos, calcula el
      embedding promedio de todos sus polos {a, b}.
    - Busca en la red un candidato de concepto superior semánticamente cercano.
    - Si la similitud ≥ umbral_refuerzo, crea un nodo emergente:
        <equilibrio>_sintesis_<k>
      conectado al equilibrio y al concepto superior, y enlazado suavemente
      con los polos.
    - Respeta un máximo de profundidad NIVEL_MAX_EMERGENCIA.
    """
    triangulos = detectar_triangulos_equilibrio(G)
    mapa_equilibrios = defaultdict(list)

    # Agrupar triángulos por nodo de equilibrio
    for a, b, equilibrio in triangulos:
        mapa_equilibrios[equilibrio].append((a, b))

    conceptos_emergentes = []

    for equilibrio, dualidades in mapa_equilibrios.items():
        # Profundidad actual de emergencia
        nivel_emergencia = G.nodes[equilibrio].get("nivel_emergencia", 0)
        if nivel_emergencia >= NIVEL_MAX_EMERGENCIA:
            print(f"⛔ Nodo '{equilibrio}' ya tiene nivel_emergencia {nivel_emergencia}, se omite para evitar recursión.")
            continue

        # Si el equilibrio ya es emergente de nivel muy bajo, lo ignoramos
        if G.nodes[equilibrio].get("tipo") == "emergente" and G.nodes[equilibrio].get("nivel_conceptual", 0) < 3:
            print(f"⛔ Nodo '{equilibrio}' es emergente sin nivel suficiente, se descarta.")
            continue

        if len(dualidades) < min_triangulos:
            continue

        # Conjunto de polos a partir de todas las dualidades del equilibrio
        polos = set()
        for a, b in dualidades:
            polos.update([a, b])

        polos = list(polos)
        if not polos:
            continue

        # Embedding promedio de los polos
        embeddings_polos = obtener_embeddings_lista(polos, modelo_embeddings)
        if embeddings_polos is None or len(embeddings_polos) == 0:
            continue

        embedding_promedio = embeddings_polos.mean(dim=0, keepdim=True)

        # Candidatos a concepto superior: todos los nodos que no son el equilibrio
        # ni los polos; preferimos no usar nodos emergentes como concepto superior.
        candidatos = [
            n for n in G.nodes()
            if n != equilibrio
            and n not in polos
            and G.nodes[n].get("tipo") != "emergente"
        ]
        if not candidatos:
            continue

        embeddings_candidatos = obtener_embeddings_lista(candidatos, modelo_embeddings)
        if embeddings_candidatos is None or len(embeddings_candidatos) == 0:
            continue

        # Similitud coseno entre el embedding promedio y los candidatos
        sims = util.pytorch_cos_sim(embedding_promedio, embeddings_candidatos)[0]
        max_sim, idx = torch.max(sims, 0)
        max_sim_val = max_sim.item()

        if max_sim_val < umbral_refuerzo:
            print(f"⚠️ No se creó emergente para '{equilibrio}' (similitud máxima: {max_sim_val:.2f})")
            continue

        nodo_superior = candidatos[idx.item()]

        # Evitar conceptos superiores por encima del nivel 4 (capa cuantitativa máxima)
        if G.nodes[nodo_superior].get("nivel_conceptual", 0) >= 4:
            print(f"⛔ Nivel máximo alcanzado con '{nodo_superior}' (nivel 4). No se generará emergente superior.")
            continue

        # Crear nombre emergente coherente
        nombre_emergente = f"{equilibrio}_sintesis_{len(conceptos_emergentes)}"
        if nombre_emergente in G:
            print(f"⚠️ Nodo emergente '{nombre_emergente}' ya existe, se omite para evitar duplicados.")
            continue

        # Validar formato de dualidades
        if not isinstance(dualidades, list) or not all(isinstance(d, tuple) and len(d) == 2 for d in dualidades):
            print(f"⚠️ Formato incorrecto en dualidades para '{equilibrio}', se omite este emergente.")
            continue

        # Crear nodo emergente
        G.add_node(nombre_emergente, tipo="emergente", dualidades=dualidades)

        nivel_base = G.nodes[nodo_superior].get("nivel_conceptual")
        G.nodes[nombre_emergente]["nivel_conceptual"] = (nivel_base + 1) if nivel_base is not None else 3
        G.nodes[nombre_emergente]["nivel_emergencia"] = nivel_emergencia + 1
        G.nodes[nombre_emergente]["concepto_superior"] = nodo_superior

        # Conexiones estructurales principales
        G.add_edge(nombre_emergente, equilibrio, weight=2.5)
        G.add_edge(nodo_superior, nombre_emergente, weight=2.2)

        # Ajustar dualidades entre polos según su similitud y conectar con el emergente
        for a, b in dualidades:
            emb_a = obtener_embedding(a, modelo_embeddings)
            emb_b = obtener_embedding(b, modelo_embeddings)
            if emb_a is None or emb_b is None:
                continue

            similitud_ab = util.pytorch_cos_sim(emb_a, emb_b).item()

            if similitud_ab >= umbral_refuerzo:
                G.add_edge(a, b, weight=2.0)
                G.add_edge(b, a, weight=2.0)
            elif similitud_ab <= umbral_debilitamiento:
                if G.has_edge(a, b):
                    G.remove_edge(a, b)
                if G.has_edge(b, a):
                    G.remove_edge(b, a)

            # El emergente se conecta suavemente con los polos
            G.add_edge(nombre_emergente, a, weight=1.8)
            G.add_edge(nombre_emergente, b, weight=1.8)
            # Y el equilibrio mantiene un lazo más suave aún
            G.add_edge(equilibrio, a, weight=1.2)
            G.add_edge(equilibrio, b, weight=1.2)

        conceptos_emergentes.append((nombre_emergente, equilibrio, dualidades))
        print(f"🌱 Emergente creado: {nombre_emergente} desde {equilibrio} con {len(dualidades)} dualidades (sim={max_sim_val:.2f})")

    return conceptos_emergentes


###########################################################################################################################   ELIMINAR FUNCION
# ACTUALMENTE NO SE USA
def visualizar_meta_triangulo(G, emergente_data):
    """
    Visualiza un meta-triángulo emergente a partir de 3 dualidades conectadas por un nodo de equilibrio.
    emergente_data debe ser una tupla: (nombre_emergente, equilibrio, dualidades)
    """
    nombre, equilibrio, dualidades = emergente_data
    subG = nx.DiGraph()
    subG.add_node(nombre)
    subG.add_node(equilibrio)
    subG.add_edge(nombre, equilibrio)

    for a, b in dualidades:
        subG.add_node(a)
        subG.add_node(b)
        subG.add_edge(a, b)
        subG.add_edge(b, a)
        subG.add_edge(equilibrio, a)
        subG.add_edge(equilibrio, b)
        subG.add_edge(nombre, a)
        subG.add_edge(nombre, b)

    concepto_sup = G.nodes[nombre].get("concepto_superior")
    if concepto_sup:
        subG.add_node(concepto_sup)
        subG.add_edge(concepto_sup, nombre)

    net = Network(height="500px", width="100%", directed=True)
    for nodo in subG.nodes():
        color = ("gold" if nodo == nombre else
                 "green" if nodo == equilibrio else
                 "orange" if nodo == concepto_sup else
                 "red")
        net.add_node(nodo, label=nodo, color=color)

    for u, v in subG.edges():
        #net.add_edge(u, v, color="gray")
        color = color_edge(u, v, G)
        net.add_edge(u, v, color=color)

    filename = f"meta_triangulo_{nombre}.html".replace(" ", "_")
    net.write_html(filename)
    
    print(f"🔺 Meta-triángulo visualizado en '{filename}'")

# ACTUALMENTE NO SE USA
def visualizar_todas_triples(G):
    """
    Genera un único archivo HTML con todos los nodos emergentes tipo triple dualidad.
    """
    emergentes = [n for n, attr in G.nodes(data=True) if attr.get("tipo") == "emergente"]
    subG = G.subgraph(emergentes).copy()

    net = Network(height="700px", width="100%", directed=True)
    for nodo in subG.nodes():
        net.add_node(nodo, label=nodo, color="gold")

    for u, v in subG.edges():
        net.add_edge(u, v)

    net.write_html("triple_dualidades_completas.html")
    print("📁 Visualización global guardada como 'triple_dualidades_completas.html'")

def detectar_conceptos_emergentes(G, min_triangulos=3, umbral_similitud=0.70):
    """
    Detecta nodos que actúan como equilibrio en al menos 'min_triangulos' dualidades diferentes
    y, en lugar de crear '<equilibrio>_sintesis_X', busca un CONCEPTO REAL del grafo que actúe
    como emergente semánticamente coherente (usando embeddings).

    - Usa: detectar_triangulos_equilibrio(G)
    - Marca como emergente un nodo existente (tipo 'concepto'/'sintesis'/'abstracto', etc.)
    """

    triangulos = detectar_triangulos_equilibrio(G)
    mapa_equilibrios = defaultdict(list)

    # 1) Agrupar dualidades por equilibrio
    for a, b, equilibrio in triangulos:
        mapa_equilibrios[equilibrio].append((a, b))

    conceptos_emergentes = []

    for equilibrio, dualidades in mapa_equilibrios.items():
        # Necesitamos que ese equilibrio esté en suficientes triángulos
        if len(dualidades) < min_triangulos:
            continue

        # --- Construimos el conjunto de nodos de referencia (equilibrio + extremos) ---
        nodos_referencia = {equilibrio}
        for a, b in dualidades:
            nodos_referencia.add(a)
            nodos_referencia.add(b)

        vectores = []
        for nodo in nodos_referencia:
            try:
                v = obtener_embedding(nodo, cache=embeddings_cache)
                vectores.append(v)
            except Exception as e:
                print(f"⚠️ No se pudo obtener embedding para '{nodo}': {e}")

        if len(vectores) < 2:
            # No tenemos suficiente información para un promedio coherente
            continue

        # Embedding "centro" de esa constelación de dualidades alrededor del equilibrio
        centro = torch.stack(vectores, dim=0).mean(dim=0, keepdim=True)  # shape (1, d)

        # --- Elegimos candidato emergente entre todos los nodos del grafo ---
        candidatos = []
        cand_vecs = []

        for nodo in G.nodes():
            if nodo in nodos_referencia:
                continue
            if nodo in ("IA_m", "Subconsciente", "Subconsciente_semantico"):
                continue

            tipo_nodo = G.nodes[nodo].get("tipo", "concepto")
            # No queremos que el emergente sea otra dualidad o equilibrio "básico"
            if tipo_nodo in ("dualidad", "equilibrio"):
                continue

            try:
                v = obtener_embedding(nodo, cache=embeddings_cache)
            except Exception:
                continue

            candidatos.append(nodo)
            cand_vecs.append(v)

        if not candidatos:
            continue

        cand_tensor = torch.stack(cand_vecs, dim=0)  # (N, d)
        sims = util.pytorch_cos_sim(centro, cand_tensor)[0]  # (N,)

        # Mejor candidato
        best_idx = int(torch.argmax(sims).item())
        best_node = candidatos[best_idx]
        best_sim = float(sims[best_idx])

        if best_sim < umbral_similitud:
            print(
                f"⚠️ No se encontró emergente semántico fuerte para equilibrio '{equilibrio}' "
                f"(sim_max={best_sim:.2f})"
            )
            continue

        # --- Marcamos ese nodo como EMERGENTE sobre ese equilibrio ---
        datos = G.nodes[best_node]
        nivel_actual = datos.get("nivel_conceptual", 0)
        if nivel_actual < 3:
            datos["nivel_conceptual"] = 3
        datos["tipo"] = "emergente"
        datos["concepto_superior"] = equilibrio

        # Acumulamos las dualidades asociadas a este emergente
        dualidades_existentes = datos.get("dualidades", [])
        existentes_set = {
            tuple(d) for d in dualidades_existentes if isinstance(d, (list, tuple)) and len(d) == 2
        }

        for a, b in dualidades:
            par1 = (a, b)
            par2 = (b, a)
            if par1 not in existentes_set and par2 not in existentes_set:
                dualidades_existentes.append([a, b])
                existentes_set.add(par1)

        datos["dualidades"] = dualidades_existentes

        # --- Conectamos el emergente al equilibrio y a los extremos ---
        if not G.has_edge(best_node, equilibrio):
            G.add_edge(best_node, equilibrio, weight=2.5, tipo="emergente")

        for a, b in dualidades:
            for extremo in (a, b):
                if not G.has_edge(best_node, extremo):
                    G.add_edge(best_node, extremo, weight=1.8, tipo="emergente")

        conceptos_emergentes.append((best_node, equilibrio, dualidades))
        print(
            f"🌱 Emergente semántico: '{best_node}' asociado al equilibrio '{equilibrio}' "
            f"({len(dualidades)} dualidades, sim={best_sim:.2f})"
        )

    return conceptos_emergentes

def detectar_conceptos_emergentesANTIGUO(G, min_triangulos=3):
    """
    Detecta nodos que actúan como equilibrio en al menos tres dualidades diferentes
    y genera un nodo emergente (superior) como síntesis conceptual, usando atributos en lugar de nombres.
    """
    triangulos = detectar_triangulos_equilibrio(G)
    mapa_equilibrios = defaultdict(list)

    for a, b, equilibrio in triangulos:
        mapa_equilibrios[equilibrio].append((a, b))

    conceptos_emergentes = []

    for equilibrio, dualidades in mapa_equilibrios.items():
        if len(dualidades) >= min_triangulos:
            # El nuevo nodo será una síntesis superior al equilibrio
            nivel_eq = G.nodes[equilibrio].get("nivel_conceptual", 2)  # equilibrio suele estar en 2
            nivel_emergente = nivel_eq + 1

            # Creamos un identificador único basado en el equilibrio
            nombre_emergente = f"{equilibrio}_sintesis_{nivel_emergente}"

            if nombre_emergente not in G:
                G.add_node(nombre_emergente,
                           tipo="emergente",
                           nivel_conceptual=nivel_emergente,
                           concepto_superior=equilibrio,
                           dualidades=dualidades)
                G.add_edge(nombre_emergente, equilibrio, weight=2.5)

                for a, b in dualidades:
                    G.add_edge(nombre_emergente, a, weight=1.8)
                    G.add_edge(nombre_emergente, b, weight=1.8)

                conceptos_emergentes.append((nombre_emergente, equilibrio, dualidades))
                print(f"🌱 Emergente creado: {nombre_emergente} desde {equilibrio} con {len(dualidades)} dualidades")

    return conceptos_emergentes

def visualizar_meta_triangulo_global(G):
    emergentes = [n for n, attr in G.nodes(data=True) if attr.get("tipo") == "emergente"]
    nodos_vis = set()
    edges_vis = []

    for nodo in emergentes:
        nodos_vis.add(nodo)
        dualidades = G.nodes[nodo].get("dualidades", [])
        equilibrio = list(G.successors(nodo))[0] if list(G.successors(nodo)) else None
        if equilibrio:
            nodos_vis.add(equilibrio)
            edges_vis.append((nodo, equilibrio))
        for a, b in dualidades:
            nodos_vis.update([a, b])
            edges_vis.extend([(nodo, a), (nodo, b), (equilibrio, a), (equilibrio, b), (a, b), (b, a)])
        if G.nodes[nodo].get("concepto_superior"):
            sup = G.nodes[nodo]["concepto_superior"]
            nodos_vis.add(sup)
            edges_vis.append((sup, nodo))

    subG = G.subgraph(nodos_vis).copy()
    net = Network(height="800px", width="100%", directed=True)

    for n in subG.nodes():
        tipo = G.nodes[n].get("tipo")
        color = ("gold" if tipo == "emergente" else
                 "green" if tipo == "equilibrio" else
                 "orange" if tipo == "concepto_superior" else
                 "red")
        net.add_node(n, label=n, color=color)

    for u, v in edges_vis:
        if subG.has_edge(u, v):
            #net.add_edge(u, v)
            color = color_edge(u, v, G)
            net.add_edge(u, v, color=color)

    net.write_html("estructura_triple_dualidad_global.html")
    print("🌐 Visualización global de todas las estructuras emergentes guardada como 'estructura_triple_dualidad_global.html'")
    
def crear_sintesis(G, triada, modelo, sintesis_nombre=None):
    if not sintesis_nombre:
        embeddings = modelo.encode(triada)
        embedding_promedio = np.mean(embeddings, axis=0)
        
        todos_nodos = list(G.nodes)
        embeddings_todos = modelo.encode(todos_nodos)
        similitudes = cosine_similarity([embedding_promedio], embeddings_todos)[0]
        sintesis_nombre = todos_nodos[np.argmax(similitudes)]


def crear_nodo_sintesis_triada(a, b, c, G, modelo_embeddings, umbral_similitud=0.7):
    """
    Crea un nodo de síntesis para una triada (a, b, c) buscando un nombre semántico
    basado en el nodo más cercano en la red, usando embeddings con caché.
    """
    if not all(n in G for n in [a, b, c]):
        print(f"❌ Uno de los nodos ({a}, {b}, {c}) no está en la red.")
        return None

    # Calcular embedding promedio
    emb_a = obtener_embedding(a, modelo_embeddings)
    emb_b = obtener_embedding(b, modelo_embeddings)
    emb_c = obtener_embedding(c, modelo_embeddings)
    emb_prom = (emb_a + emb_b + emb_c) / 3

    # Buscar nodo más similar
    candidatos = list(G.nodes())
    embeddings_candidatos = obtener_embeddings_lista(candidatos, modelo_embeddings)

    similitudes = util.pytorch_cos_sim(emb_prom, embeddings_candidatos)[0]
    max_sim, idx = similitudes.max(0)
    nodo_similar = candidatos[idx.item()]
    similitud = max_sim.item()

    if similitud >= umbral_similitud:
        nombre_sintesis = f"{nodo_similar}_triada"
    else:
        nombre_sintesis = f"triada_sintesis_{a}_{b}_{c}"

    if nombre_sintesis not in G:
        G.add_node(nombre_sintesis, tipo="sintesis", nivel_conceptual=3)
        G.add_edge(nombre_sintesis, a, weight=2.0)
        G.add_edge(nombre_sintesis, b, weight=2.0)
        G.add_edge(nombre_sintesis, c, weight=2.0)
        print(f"🌟 Nodo de síntesis creado: {nombre_sintesis} (similitud: {similitud:.2f})")
    else:
        print(f"⚠️ El nodo '{nombre_sintesis}' ya existía.")

    return nombre_sintesis

def visualizar_triadas_horizontales(G, triadas, nombre="triadas_horizontales.html"):
    """
    Genera una visualización en HTML de las triadas horizontales detectadas.
    Usa los colores definidos en color_node y color_edge.
    """
    if not triadas:
        print("⚠️ No se encontraron triadas horizontales para visualizar.")
        return

    subG = nx.DiGraph()

    for a, b, c in triadas:
        for nodo in [a, b, c]:
            if nodo in G:
                subG.add_node(nodo, **G.nodes[nodo])
        subG.add_edge(a, b)
        subG.add_edge(b, c)
        subG.add_edge(c, a)

    net = Network(height="800px", width="100%", directed=True)
    posiciones = nx.spring_layout(subG)

    for nodo in subG.nodes():
        color = color_node(nodo, G)
        net.add_node(nodo, label=nodo, color=color, title=nodo)

    for u, v in subG.edges():
        color = color_edge(u, v, G)
        net.add_edge(u, v, color=color, width=1.5)

    net.write_html(nombre)
    print(f"✅ Visualización de triadas guardada en {nombre}")

# ACTUALMENTE NO SE USA
def visualizar_triangulo(G, trio):
    subG = G.subgraph(trio).copy()
    net = Network(height="300px", width="100%", directed=True)

    for nodo in subG.nodes():
        color = "orange" if nodo == trio[2] else "lightblue"
        net.add_node(nodo, label=nodo, color=color)
    
    for u, v in subG.edges():
        net.add_edge(u, v)

    nombre = f"triangulo_{trio[0]}_{trio[1]}_{trio[2]}.html".replace(" ", "_")
    net.write_html(nombre)
    print(f"✅ Visualización guardada como {nombre}")
    
def detectar_micro_tetraedros(G):
    tetraedros = []

    for nodo in G.nodes():
        if G.nodes[nodo].get("tipo") != "equilibrio":
            continue

        vecinos = list(G.neighbors(nodo))
        for i in range(len(vecinos)):
            for j in range(i + 1, len(vecinos)):
                a, b = vecinos[i], vecinos[j]

                if G.has_edge(a, b) and G.has_edge(b, a):
                    # Triángulo A-B-equilibrio
                    posibles_superiores = detectar_nodo_superior(a, b, G, top_n=1)

                    if posibles_superiores:
                        superior = posibles_superiores[0]
                        if superior in {a, b, nodo}:
                            continue

                        emb_a = obtener_embedding(a, modelo_embeddings)
                        emb_b = obtener_embedding(b, modelo_embeddings)
                        emb_eq = obtener_embedding(nodo, modelo_embeddings)
                        emb_sup = obtener_embedding(superior, modelo_embeddings)

                        emb_prom = (emb_a + emb_b) / 2
                        sim_dualidad = util.pytorch_cos_sim(emb_a, emb_b).item()
                        sim_equilibrio = util.pytorch_cos_sim(emb_prom, emb_eq).item()
                        sim_superior = util.pytorch_cos_sim(emb_prom, emb_sup).item()

                        tetraedros.append({
                            "dualidad_1": a,
                            "dualidad_2": b,
                            "equilibrio": nodo,
                            "concepto_superior": superior,
                            "similitud_dualidad": round(sim_dualidad, 3),
                            "similitud_equilibrio": round(sim_equilibrio, 3),
                            "similitud_superior": round(sim_superior, 3)
                        })

    return tetraedros

def guardar_micro_tetraedros(tetraedros, archivo="json/micro_tetraedros.json"):
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(tetraedros, f, ensure_ascii=False, indent=4)

def guardar_triadas(triadas, ruta="triadas_horizontales.json"):
    triadas_existentes = []

    # Si el archivo existe, cargar triadas previas
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            try:
                triadas_existentes = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Archivo existente corrupto o vacío, se sobrescribirá.")

    # Unir y eliminar duplicados (conversión a tuplas para poder comparar)
    todas = triadas_existentes + triadas
    todas_unicas = list({tuple(sorted(t)) for t in todas})  # ignora orden

    # Volver a listas para guardar en JSON
    todas_unicas = [list(t) for t in todas_unicas]

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(todas_unicas, f, ensure_ascii=False, indent=2)

    print(f"📁 Triadas horizontales actualizadas y guardadas en {ruta} ({len(todas_unicas)} únicas)")

def insertar_cuadrado_matematico_y_detectar(G, output_path="json/estructura_prismas.json"):
    """
    Inserta manualmente un cuadrado conceptual matemático en la red G con nodos surgidos de triadas
    y luego ejecuta la detección de prismas conceptuales.
    """
    operaciones = ["suma", "resta", "división", "multiplicación"]
    for op in operaciones:
        if not G.has_node(op):
            G.add_node(op, tipo="operación", origen="triada", nivel_conceptual=4)

    # Ejecutar detección como JSON
    return detectar_cuadrados_conceptuales_json(G, output_path)

def detectar_cuadrados_conceptuales_generalizados(
    G,
    modelo,
    umbral_similitud=0.75,
    output_path="json/prismas_hibridos.json"
):
    """
    Detecta estructuras cuadradas emergentes (nivel 4) con alta cohesión semántica.
    Si no son parte de patrones predefinidos, infiere su nodo emergente por embeddings.
    """
    patrones_validos = [
        {"resta", "suma", "división", "multiplicación"},
        {"seno", "coseno", "tangente", "cotangente"},
        {"∪", "∩", "⊆", "⊇"}
    ]
    prismas_detectados = []
    nodos = list(G.nodes())
    candidatos = [
        n for n in nodos
        if G.nodes[n].get("origen") == "triada" and G.nodes[n].get("nivel_conceptual") == 4
    ]
    for grupo in itertools.combinations(candidatos, 4):
        simbolos = set(grupo)
        # Si es un patrón explícito
        if simbolos in patrones_validos:
            if simbolos == {"resta", "suma", "división", "multiplicación"}:
                emergente = "raíz cuadrada"
            elif simbolos == {"seno", "coseno", "tangente", "cotangente"}:
                emergente = "identidad trigonométrica"
            elif simbolos == {"∪", "∩", "⊆", "⊇"}:
                emergente = "lógica de conjuntos"
            else:
                emergente = "emergencia desconocida"
        else:
            # Evaluar similitud entre los 4 por embeddings
            embs = [obtener_embedding(n, modelo, cache=embeddings_cache) for n in grupo]
            if len(embs) < 2:
                continue  # ⚠️ Prevención de error por embeddings vacíos o fallidos
            embs = torch.stack(embs)  # 🔧 Solución clave
            sim_matrix = util.pytorch_cos_sim(embs, embs)
            if not all(sim_matrix[i][j] > umbral_similitud for i in range(4) for j in range(4) if i != j):
                continue

            # Calcular nodo emergente por centro semántico
            emb_prom = embs.mean(dim=0)
            todos = list(G.nodes())
            emb_todos = [obtener_embedding(t, modelo, cache=embeddings_cache) for t in todos]
            emb_todos = torch.stack(emb_todos)  # 🔧 Segunda solución clave
            sim_centro = util.pytorch_cos_sim(emb_prom, emb_todos)[0]
            idx_max = sim_centro.argmax().item()
            emergente = todos[idx_max]

        prisma = {
            "estructura": "prisma_conceptual",
            "nodos_base": list(simbolos),
            "centro": "promedio",
            "emergente": emergente,
            "origen": "triadas"
        }
        prismas_detectados.append(prisma)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prismas_detectados, f, indent=2, ensure_ascii=False)
    return output_path, prismas_detectados

# ACTUALMENTE NO SE USA    
def visualizar_prismas_conceptuales(prismas, output_html="subgrafos/visualizaciones_prismas/prismas_detectados.html"):
    """
    Genera una visualización HTML interactiva para los prismas conceptuales detectados.
    Cada prisma se representa como una estructura cuadrada con un nodo central y un nodo emergente.
    """
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for idx, prisma in enumerate(prismas):
        base = prisma["nodos_base"]
        centro = f"{prisma['centro']}_{idx}"
        emergente = f"{prisma['emergente']}_{idx}"
        # Añadir nodos base
        for nodo in base:
            net.add_node(f"{nodo}_{idx}", label=nodo, color="#FFD700", size=20)
        # Añadir nodo centro
        net.add_node(centro, label=prisma["centro"], color="#90EE90", size=18)
        # Añadir nodo emergente
        net.add_node(emergente, label=prisma["emergente"], color="#87CEFA", size=22)
        # Conectar base al centro
        for nodo in base:
            net.add_edge(f"{nodo}_{idx}", centro, color="#AAAAAA")
        # Conectar centro al emergente
        net.add_edge(centro, emergente, color="#FF69B4", width=2)
    net.show(output_html)
    return output_html
    
def visualizar_prisma_individual(prisma, idx, carpeta="subgrafos/visualizaciones_prismas"):
    """
    Genera una visualización HTML individual para un prisma conceptual.
    """
    os.makedirs(carpeta, exist_ok=True)
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    base = prisma["nodos_base"]
    centro = f"{prisma['centro']}_{idx}"
    emergente = f"{prisma['emergente']}_{idx}"

    # Añadir nodos base
    for nodo in base:
        net.add_node(f"{nodo}_{idx}", label=nodo, color="#FFD700", size=20)

    # Nodo central
    net.add_node(centro, label=prisma["centro"], color="#90EE90", size=18)

    # Nodo emergente
    net.add_node(emergente, label=prisma["emergente"], color="#87CEFA", size=22)

    # Enlaces base → centro
    for nodo in base:
        net.add_edge(f"{nodo}_{idx}", centro, color="#AAAAAA")

    # Centro → emergente
    net.add_edge(centro, emergente, color="#FF69B4", width=2)

    output_path = os.path.join(carpeta, f"prisma_{idx}.html")
    net.write_html(output_path)
    print(f"✅ Prisma {idx} visualizado en: {output_path}")


def generar_visualizaciones_prismas_individuales(path_json="json/prismas_hibridos.json"):
    """
    Carga un archivo JSON con prismas detectados y genera visualizaciones individuales.
    """
    with open(path_json, "r", encoding="utf-8") as f:
        prismas = json.load(f)

    for idx, prisma in enumerate(prismas):
        visualizar_prisma_individual(prisma, idx)

    print(f"🎉 Visualizaciones generadas para {len(prismas)} prismas.")

def detectar_cuadrados_conceptuales_json(G, output_path="json/estructura_prismas.json"):
    """
    Detecta estructuras cuadradas simbólicas formadas por nodos emergentes de triadas
    y guarda su estructura como JSON sin modificar el grafo.
    """
    patrones_validos = [
        {"resta", "suma", "división", "multiplicación"},
        {"seno", "coseno", "tangente", "cotangente"},
        {"∪", "∩", "⊆", "⊇"}
    ]

    prismas_detectados = []
    nodos = list(G.nodes())
    candidatos = [n for n in nodos if G.nodes[n].get("origen") == "triada"]

    for i in range(len(candidatos)):
        for j in range(i+1, len(candidatos)):
            for k in range(j+1, len(candidatos)):
                for l in range(k+1, len(candidatos)):
                    grupo = [candidatos[i], candidatos[j], candidatos[k], candidatos[l]]
                    simbolos = set(grupo)

                    if simbolos in patrones_validos:
                        if simbolos == {"resta", "suma", "división", "multiplicación"}:
                            emergente = "raíz cuadrada"
                        elif simbolos == {"seno", "coseno", "tangente", "cotangente"}:
                            emergente = "identidad trigonométrica"
                        elif simbolos == {"∪", "∩", "⊆", "⊇"}:
                            emergente = "lógica de conjuntos"
                        else:
                            emergente = "emergencia desconocida"

                        prisma = {
                            "estructura": "prisma_conceptual",
                            "nodos_base": list(simbolos),
                            "centro": "promedio",
                            "emergente": emergente,
                            "origen": "triadas"
                        }
                        prismas_detectados.append(prisma)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prismas_detectados, f, indent=2, ensure_ascii=False)

    return output_path, prismas_detectados

def generar_indice_visual_prismas(carpeta="subgrafos/visualizaciones_prismas", salida="subgrafos/visualizaciones_prismas/index.html"):
    """
    Genera un índice HTML con enlaces a todos los prismas visualizados en la carpeta dada.
    """
    archivos = sorted([
        f for f in os.listdir(carpeta)
        if f.startswith("prisma_") and f.endswith(".html")
    ])

    html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>🔷 Índice de Prismas Detectados</title>
  <style>
    body {{ font-family: sans-serif; padding: 2rem; background: #f9f9f9; }}
    h1 {{ color: #444; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ margin-bottom: 0.5rem; }}
    a {{ text-decoration: none; color: #0077cc; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>🔷 Prismas Conceptuales Detectados</h1>
  <ul>
"""

    for f in archivos:
        html += f'    <li><a href="{f}" target="_blank">{f}</a></li>\n'

    html += """  </ul>
</body>
</html>
"""

    # Guardar el índice en el mismo directorio
    with open(salida, "w", encoding="utf-8") as out:
        out.write(html)

    print(f"📚 Índice generado en: {salida}")

def decidir_modo_exploracion(G):
    """
    IA_m elige dinámicamente el modo de atención según el estado de la red fractal.
    """
    total_nodos = len(G.nodes())
    dualidades = [n for n in G.nodes() if G.nodes[n].get("tipo") == "dualidad"]
    emergentes = [n for n in G.nodes() if G.nodes[n].get("tipo") == "emergente"]
    poco_conectados = [n for n in G.nodes() if G.degree(n) <= 1 and G.nodes[n].get("nivel_conceptual", 0) <= 1]
    nodos_ia_m = list(G.successors("IA_m")) if "IA_m" in G else []


    
    # Si hay muchos nodos flotantes, entrar en subconsciente
    nodos_flotantes = [n for n in G.nodes() if G.degree(n) == 0]
    if len(nodos_flotantes) > 80:
        print("🌌 Hay muchos nodos flotando sin integrar... modo subconsciente recomendado.")
        return "d"

    nodos_ia_m = list(G.successors("IA_m")) if "IA_m" in G else []
    if len(nodos_ia_m) > 30:
        print("⚖️ IA_m nota que su atención está saturada... replegando enfoque.")
        return "profundo"

    # Si hay muchos nodos poco conectados, IA_m profundiza
    elif len(poco_conectados) > total_nodos * 0.25:
        return "profundo" #ponia profundo OJOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

    # Si hay suficiente base dual pero pocos emergentes, IA_m busca síntesis
    elif len(dualidades) > 10 and len(emergentes) < 7:
        return "emergente"
    else:
        # Por defecto, se expande lateralmente
        return "horizontal"

        
def detectar_nodos_flotantes(G, umbral=1):
    return [n for n in G.nodes if G.degree(n) <= umbral]

def sugerir_nodos_interiores(G, top_n=5):
    """Sugiere nodos interesantes para explorar en profundidad."""
    registro = cargar_registro()
    conteo_expansiones = Counter([r["nodo"] for r in registro])
    
    candidatos = [
        nodo for nodo in G.nodes()
        if G.degree(nodo) >= 3 and nodo.lower() != "ia_m"
    ]
    
    candidatos_ordenados = sorted(
        candidatos,
        key=lambda n: (G.degree(n) / (1 + conteo_expansiones.get(n, 0))),
        reverse=True
    )

    sugerencias = candidatos_ordenados[:top_n]
    print("📍 Nodos recomendados para exploración interior:")
    for nodo in sugerencias:
        print(f"   • {nodo} (conexiones: {G.degree(nodo)}, expansiones: {conteo_expansiones.get(nodo, 0)})")
    
    return sugerencias
    
def explorar_subnodo(G, nodo_base, top_n=5):
    """Explora los vecinos directos de un nodo como si IA_m entrara en él"""
    nodo_base = nodo_base.lower()
    if nodo_base not in G:
        print(f"❌ El nodo '{nodo_base}' no está en la red.")
        return

    vecinos = list(G.neighbors(nodo_base))
    if not vecinos:
        print(f"⚠️ El nodo '{nodo_base}' no tiene subconceptos directos.")
        return

    print(f"🔽 IA_m entra dentro de '{nodo_base}' y observa sus subconceptos...")
    nodos_a_expandir = vecinos[:top_n]

    for nodo in nodos_a_expandir:
        print(f"🔍 Explorando subconcepto: {nodo}")
        G.add_edge("IA_m", nodo, weight=0.1)
        añadir_a_visualizacion(nodo_base, [nodo], G)

        nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
        if nuevos_conceptos:
            G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
            registrar_expansion(nodo, nuevos_conceptos, "GPT-4")
            añadir_a_visualizacion(nodo, nuevos_conceptos, G)
            
    guardar_visualizacion_dinamica()
    #generar_subgrafos_principales(G)
    guardar_red(G)
    print(f"✅ Exploración dentro de '{nodo_base}' completada.")
    if "IA_m" in G:
        G.remove_node("IA_m")
        if usarFTP:
            subir_htmls_recientes_por_ftp(max_horas=2)

def handler_sigint(sig, frame):
    print("\n🛑 Ctrl+C detectado. Guardando estado...")
    guardar_red(GLOBAL_G)
    print("✅ Red guardada. Saliendo con elegancia...")
    expansion_activa = False
    retirar_atencion(G)
    guardar_visualizacion_dinamica()
    guardar_red(G)
    visualizar_red(G)
    sys.exit(0)

# 🔹 EXPANSIÓN AUTOMÁTICA FINAL Y FUNCIONAL
def expansion_automatica(G, diccionario, expansion_activa, modo_expansion, usar_wikipedia, usar_gpt):
    """ Expande la red automáticamente según el modo de atención de IA_m """
    signal.signal(signal.SIGINT, handler_sigint)
    respuesta_recordada = None

    while expansion_activa:
        reiniciar_visualizacion_proceso()  # 🔁 Limpiar el IA_m_proceso de la iteración anterior
        print("\n🔄 Iniciando expansión automática...")
        modo_IA = decidir_modo_exploracion(G)
        visualizar_red(G)
        if usarFTP:
            subir_htmls_recientes_por_ftp(max_horas=2)
        # 🔹 MODO SUBCONSCIENTE: nodos flotantes o desconectados
        if modo_expansion == "d" or (modo_expansion == "i" and modo_IA == "d"):
            print("🧠 IA_m se va a dormir...")
            print("🧠🌀 Conectando con el subconsciente para integrar nodos flotantes...")
            nodos_flotantes = [n for n in G.nodes() if G.degree(n) <= 1 and n.lower() != "ia_m"]
            nodos_a_expandir = nodos_flotantes[:5] if nodos_flotantes else priorizar_expansion(G)[:5]

        # 🔹 MODO ACTIVO: IA_m decide el foco de atención
        else:
            print("🧠 IA_m enfocando atención...")
            if modo_expansion == "i":
                modo_IA = decidir_modo_exploracion(G)
                print(f"🧭 Modo IA_m detectado: {modo_IA}")

                if modo_IA == "d":
                    print("😴 Muchos nodos flotantes. Cambiando a modo subconsciente...")
                    modo_expansion = "d"
                    continue

                elif modo_IA == "profundo":
                    print("🌿 IA_m entra en modo profundo: exploración interior.")
                    sugerencias = sugerir_nodos_interiores(G)
                    if sugerencias:
                        explorar_subnodo(G, sugerencias[0])
                        if "IA_m" in G:
                            G.remove_node("IA_m")  
                        #subir_htmls_recientes_por_ftp(max_horas=2)
                        continue
                    else:
                        print("⚠️ No hay nodos adecuados para exploración profunda.")
                        nodos_a_expandir = []

                elif modo_IA == "emergente":
                    print("🌟 IA_m busca síntesis estructurales emergentes...")
                    candidatos = [n for n in G.nodes() if G.nodes[n].get("tipo") in ("equilibrio", "dualidad")]
                    nodos_a_expandir = sorted(candidatos, key=lambda n: G.degree(n))[:5]

                else:
                    print("🔗 Expansión lateral por atención consciente...")
                    nodos_a_expandir = (
                        calcular_atencion_consciente_nodo_central(G, NODO_CENTRAL, 5)
                        if NODO_CENTRAL else calcular_atencion_consciente(G, 5)
                    )

            elif modo_expansion == "p":
                print("🌌 IA_m busca nodos con baja actividad...")
                nodos_a_expandir = priorizar_expansion(G)[:5]
            else:
                nodos_a_expandir = []
                

        # 🔸 Verificación de nodos válidos
        if not nodos_a_expandir:
            print("⚠️ No hay nodos elegibles para expansión automática.")
            expansion_activa = False
            retirar_atencion(G)
            return

        # 🔸 Conectar IA_m a nodos seleccionados
        if "IA_m" not in G:
            G.add_node("IA_m")
            
        nodos_a_expandir = nodos_a_expandir[:5]
        for nodo in nodos_a_expandir:
            if not expansion_activa:
                print("⏹️ Expansión automática detenida.")
                expansion_activa = False
                retirar_atencion(G)
                return

            print(f"🔍 Expandiendo nodo enfocado: {nodo}")
            G.add_edge("IA_m", nodo, weight=0.1)
            añadir_a_visualizacion(nodo, [], G)
            nuevos_conceptos = []

            if usar_gpt == "s":
                print(f"📡 Consultando ChatGPT sobre: {nodo}")
                nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
                añadir_a_visualizacion(nodo, nuevos_conceptos, G)

            if nuevos_conceptos:
                #G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
                G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos, nodo_origen=nodo)
                registrar_expansion(nodo, nuevos_conceptos, "GPT-4")
                guardar_diccionario(diccionario)

            expandir_concepto_embedding(nodo, G, diccionario)
            registrar_expansion(nodo, [], "Embeddings")
            guardar_diccionario(diccionario)

            # 🔸 Guardar visualización dinámica + atención  LO HE INCLUIDO DENTRO DEL FOR
            guardar_visualizacion_dinamica()
            visualizar_red(G)
            if usarFTP:
                subir_htmls_recientes_por_ftp(max_horas=2)
        evaluar_expansion(G)
        guardar_red(G)

        # 🔸 Desconectar IA_m tras cada ciclo
        if "IA_m" in G:
            G.remove_node("IA_m")
        

        # 🔸 Procesamiento estructural
        print(f"🔍 Buscando dualidades...")
        pipeline_dualidades_auto(G, modelo_embeddings) 
#        buscar_dualidades_faltantes(G)
        reorganizar_red(G)
        guardar_red(G)
        evaluar_expansion(G)
        guardar_historial(cargar_historial())

        triangulos = detectar_triangulos_equilibrio(G)
        emergentes = detectar_conceptos_emergentes(G)
        insertar_cuadrado_matematico_y_detectar(G)
        ruta, prismas = detectar_cuadrados_conceptuales_generalizados(G, modelo_embeddings)
        if prismas:
            print(f"🧠 Prismas detectados: {len(prismas)}")
            generar_visualizaciones_prismas_individuales(ruta)
            generar_indice_visual_prismas()
        guardar_triangulos(triangulos)
        guardar_micro_tetraedros(detectar_micro_tetraedros(G))
        detectar_estructura_emergente_adaptativa(G)
        visualizar_meta_triangulo_global(G)

        horizontales = detectar_triadas_horizontales(G)
        guardar_triadas(horizontales)
        visualizar_triadas_horizontales(G, horizontales)

        detectar_hipercubo_conceptual(G)
        rastrear_evolucion_conceptual(G)
        evaluar_progreso_fractal(G)

        print("✅ Expansión automática completada.")
        retirar_atencion(G)
        generar_reportes()
        visualizar_red(G)
        generar_subgrafos_principales(G)
        guardar_visualizacion_dinamica()
        if usarFTP:
            subir_htmls_recientes_por_ftp(max_horas=2)

def reparar_dualidades(G):
    reparados = 0
    for nodo in G.nodes():
        if G.nodes[nodo].get("tipo") == "emergente":
            dualidades = G.nodes[nodo].get("dualidades")
            if isinstance(dualidades, list) and all(isinstance(d, list) and len(d) == 2 for d in dualidades):
                G.nodes[nodo]["dualidades"] = [tuple(d) for d in dualidades]
                reparados += 1
    print(f"🛠 Dualidades reparadas en {reparados} nodos.")
    return G

def corregir_niveles_por_tipo(G):
    corregidos = 0
    for nodo in G.nodes():
        tipo = G.nodes[nodo].get("tipo")
        if tipo:
            nivel_actual = G.nodes[nodo].get("nivel_conceptual")
            nivel_esperado = {
                "concepto": 0,
                "dualidad": 1,
                "equilibrio": 2,
                "emergente": 3,
                "abstracto": 4,
            }.get(tipo)
            if nivel_esperado is not None and nivel_actual != nivel_esperado:
                G.nodes[nodo]["nivel_conceptual"] = nivel_esperado
                corregidos += 1
    print(f"🔧 Niveles corregidos por tipo en {corregidos} nodos.")
    return G

def ejecutar_auditoria_y_ofrecer_reparacion(G):
    errores = auditar_red_semantica(G)

    # Solo consideramos errores si no son triangulos_invalidos
    hay_errores = any(
        clave != "triangulos_invalidos" and len(lista) > 0
        for clave, lista in errores.items()
    )

    if hay_errores:
        print("⚠️ Se han detectado inconsistencias en la red.")
        respuesta = input("¿Quieres intentar repararlas automáticamente? (s/n): ").strip().lower()

        if respuesta == "s":
            G = reparar_dualidades(G)
            G = corregir_niveles_por_tipo(G)
            G = reparar_emergentes_sin_equilibrio(G)
            print("🔁 Ejecutando nueva auditoría tras la reparación...")
            errores = auditar_red_semantica(G)
        else:
            print("❗ Revisión manual recomendada antes de continuar.")
    else:
        print("✅ Red limpia. No se detectaron errores estructurales.")

    return G

def auditar_red_semantica(G):
    # 🔍 Antes de auditar, limpiar dualidades débiles
    try:
        G = evaluar_dualidades_por_estructura(G)
    except Exception as e:
        print(f"⚠️ Error al evaluar dualidades por estructura durante la auditoría: {e}")

    errores = {
        "emergentes_sin_dualidades": [],
        "dualidades_mal_formadas": [],
        "nodos_sin_nivel": [],
        "inconsistencias_tipo_nivel": [],
        "emergentes_sin_equilibrio": [],
        "triangulos_invalidos": [],
    }

    for nodo, datos in G.nodes(data=True):
        tipo = datos.get("tipo")
        nivel = datos.get("nivel_conceptual")

        # 🧩 Nivel ausente
        if nivel is None:
            errores["nodos_sin_nivel"].append(nodo)

        # 🧩 Inconsistencias tipo-nivel
        if tipo == "dualidad" and nivel != 1:
            errores["inconsistencias_tipo_nivel"].append((nodo, tipo, nivel))
        if tipo == "equilibrio" and nivel != 2:
            errores["inconsistencias_tipo_nivel"].append((nodo, tipo, nivel))
        if tipo == "emergente" and nivel is not None and nivel < 3:
            errores["inconsistencias_tipo_nivel"].append((nodo, tipo, nivel))

        # 🧩 Dualidades mal asignadas
        if tipo == "emergente":
            dualidades = datos.get("dualidades")
            if not dualidades:
                errores["emergentes_sin_dualidades"].append(nodo)
            elif not all(isinstance(d, tuple) and len(d) == 2 for d in dualidades):
                errores["dualidades_mal_formadas"].append((nodo, dualidades))

            # 🧩 Emergente sin conexión a equilibrio
            tiene_equilibrio = any(
                G.has_edge(nodo, succ) and G.nodes[succ].get("tipo") == "equilibrio"
                for succ in G.successors(nodo)
            )
            if not tiene_equilibrio:
                errores["emergentes_sin_equilibrio"].append(nodo)

    # 🧩 Detectar triángulos no válidos
    for n in G.nodes():
        vecinos = list(G.neighbors(n))
        for a, b in combinations(vecinos, 2):
            if G.has_edge(a, b) or G.has_edge(b, a):
                if not (
                    G.nodes[n].get("tipo") == "equilibrio"
                    and G.nodes[a].get("tipo") == "dualidad"
                    and G.nodes[b].get("tipo") == "dualidad"
                ):
                    errores["triangulos_invalidos"].append((n, a, b))

    # 🧾 Reporte
    print("\n📋 RESULTADO DE AUDITORÍA SEMÁNTICA")
    for clave, lista in errores.items():
        print(f"🔹 {clave}: {len(lista)}")
        if lista:
            for item in lista[:5]:
                print(f"   - {item}")
            if len(lista) > 5:
                print(f"   ... y {len(lista) - 5} más.")
    print("✅ Auditoría finalizada.\n")

    # 💊 Reparación especial de nodo si existe
    if "equilibrio_términos" in G.nodes():
        G.nodes["equilibrio_términos"]["tipo"] = "equilibrio"
        G.nodes["equilibrio_términos"]["nivel_conceptual"] = 2
        if "dualidades" in G.nodes["equilibrio_términos"]:
            del G.nodes["equilibrio_términos"]["dualidades"]
        print("♻️ Nodo 'equilibrio_términos' corregido como equilibrio (nivel 2).")

    # 💡 Depuración segura (si queda alguna referencia antigua)
    if "términos_emergente" in G.nodes():
        print(G.nodes["términos_emergente"])
        print(list(G.successors("términos_emergente")))
        print(list(G.predecessors("términos_emergente")))

    return errores

def rastrear_evolucion_conceptual(G):
    """
    Analiza la red para detectar:
    - extremos y centros del hipercubo
    - triadas estructurales del hipercubo
    - triadas lineales (tipo tiempo)
    - sistemas posicionales (coordenadas + centro)
    y registra la evolución en un JSON.
    """
    resultado = {
        "timestamp": datetime.now().isoformat(),
        "extremos_detectados": [],
        "centros_detectados": [],
        "triadas_hipercubo": [],
        "observador_presente": False,
        "hipercubo_completo": False,
        "triadas_lineales": [],
        "sistemas_posicionales": []
    }

    # --- Nodos clave del hipercubo actual ---
    extremos = {"arriba", "abajo", "izquierda", "derecha", "delante", "detrás"}
    centros = {"centro_vertical", "centro_horizontal", "centro_frontal"}

    triadas_estructurales = [
        {"arriba", "abajo", "centro_vertical"},
        {"izquierda", "derecha", "centro_horizontal"},
        {"delante", "detrás", "centro_frontal"},
        {"arriba", "izquierda", "delante"},
        {"abajo", "derecha", "detrás"},
        {"centro_vertical", "centro_horizontal", "centro_frontal"}
    ]

    # Detección de nodos presentes
    for nodo in G.nodes():
        if nodo in extremos:
            resultado["extremos_detectados"].append(nodo)
        if nodo in centros:
            resultado["centros_detectados"].append(nodo)
        if str(nodo).lower() == "observador":
            resultado["observador_presente"] = True

    # Detección de triadas estructurales del hipercubo
    for triada in triadas_estructurales:
        if all(n in G.nodes for n in triada):
            conexiones = sum(
                1
                for a in triada
                for b in triada
                if a != b and G.has_edge(a, b)
            )
            if conexiones >= 4:  # al menos 2 de las 3 aristas presentes
                resultado["triadas_hipercubo"].append(sorted(triada))

    # Verificación del hipercubo completo
    if (
        len(resultado["extremos_detectados"]) == 6 and
        len(resultado["centros_detectados"]) == 3 and
        resultado["observador_presente"] and
        len(resultado["triadas_hipercubo"]) >= 6
    ):
        resultado["hipercubo_completo"] = True

    # --- NUEVO: detectar triadas lineales (tipo tiempo) ---
    try:
        triadas_lin = detectar_triadas_lineales(G)
        resultado["triadas_lineales"] = triadas_lin
        if triadas_lin:
            print(f"⏱️ Triadas lineales detectadas: {len(triadas_lin)}")
    except Exception as e:
        print(f"⚠️ Error al detectar triadas lineales: {e}")

    # --- NUEVO: detectar sistemas posicionales (coordenadas + centro) ---
    try:
        sistemas_pos = detectar_sistemas_posicionales(G, min_dim=3)
        resultado["sistemas_posicionales"] = sistemas_pos
        if sistemas_pos:
            print(f"📐 Sistemas posicionales detectados: {len(sistemas_pos)}")
            # Guardamos también en su propio JSON acumulativo
            guardar_estructuras_posicionales(sistemas_pos)
    except Exception as e:
        print(f"⚠️ Error al detectar sistemas posicionales: {e}")

    # Guardar historial de evolución conceptual
    historial_path = "evolucion_conceptual.json"
    if os.path.exists(historial_path):
        with open(historial_path, "r", encoding="utf-8") as f:
            try:
                historial = json.load(f)
            except json.JSONDecodeError:
                historial = []
    else:
        historial = []

    historial.append(resultado)
    with open(historial_path, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    print(f"🧭 Evolución conceptual actualizada en {historial_path}")


def rastrear_evolucion_conceptualANTIGUO(G):
    resultado = {
        "timestamp": datetime.now().isoformat(),
        "extremos_detectados": [],
        "centros_detectados": [],
        "triadas_detectadas": [],
        "observador_presente": False,
        "hipercubo_completo": False
    }

    # Nodos clave del hipercubo
    extremos = {"arriba", "abajo", "izquierda", "derecha", "delante", "detrás"}
    centros = {"centro_vertical", "centro_horizontal", "centro_frontal"}
    triadas_estructurales = [
        {"arriba", "abajo", "centro_vertical"},
        {"izquierda", "derecha", "centro_horizontal"},
        {"delante", "detrás", "centro_frontal"},
        {"arriba", "izquierda", "delante"},
        {"abajo", "derecha", "detrás"},
        {"centro_vertical", "centro_horizontal", "centro_frontal"}
    ]

    # Detección de nodos presentes
    for nodo in G.nodes():
        if nodo in extremos:
            resultado["extremos_detectados"].append(nodo)
        if nodo in centros:
            resultado["centros_detectados"].append(nodo)
        if nodo.lower() == "observador":
            resultado["observador_presente"] = True

    # Detección de triadas estructurales conectadas
    for triada in triadas_estructurales:
        if all(n in G.nodes for n in triada):
            conexiones = sum(1 for a in triada for b in triada if a != b and G.has_edge(a, b))
            if conexiones >= 4:  # al menos 2 de las 3 conexiones
                resultado["triadas_detectadas"].append(sorted(triada))

    # Verificación del hipercubo completo
    if (
        len(resultado["extremos_detectados"]) == 6 and
        len(resultado["centros_detectados"]) == 3 and
        resultado["observador_presente"] and
        len(resultado["triadas_detectadas"]) >= 6
    ):
        resultado["hipercubo_completo"] = True

    # Guardar historial JSON
    historial_path = "evolucion_conceptual.json"
    if os.path.exists(historial_path):
        with open(historial_path, "r", encoding="utf-8") as f:
            historial = json.load(f)
    else:
        historial = []

    historial.append(resultado)
    with open(historial_path, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    # Generar HTML resumen
    resumen = f"""
    <html>
    <head><meta charset="utf-8"><title>Evolución Conceptual IA_m</title></head>
    <body style="font-family:sans-serif">
    <h1>Evolución Conceptual del Hipercubo</h1>
    <p><strong>Última evaluación:</strong> {resultado['timestamp']}</p>
    <ul>
      <li>✅ Extremos detectados: {len(resultado['extremos_detectados'])} / 6</li>
      <li>✅ Centros detectados: {len(resultado['centros_detectados'])} / 3</li>
      <li>🔺 Triadas estructurales detectadas: {len(resultado['triadas_detectadas'])} / 6</li>
      <li>🌀 Observador presente: {'Sí' if resultado['observador_presente'] else 'No'}</li>
      <li><strong>{'✅ Hipercubo completo detectado.' if resultado['hipercubo_completo'] else '❌ Hipercubo incompleto.'}</strong></li>
    </ul>
    </body>
    </html>
    """
    with open("evolucion_conceptual.html", "w", encoding="utf-8") as f:
        f.write(resumen)

    print("📊 Evaluación conceptual guardada en 'evolucion_conceptual.json' y 'evolucion_conceptual.html'")
    return resultado

def limpiar_emergentes_sintesis(G):
    """
    Elimina nodos antiguos del estilo '<equilibrio>_sintesis_3'
    y redirige sus conexiones al concepto_superior si existe.
    Así limpiamos nombres feos sin perder toda la estructura.
    """
    eliminados = []

    for nodo, datos in list(G.nodes(data=True)):
        if datos.get("tipo") == "emergente" and "_sintesis_" in nodo:
            sup = datos.get("concepto_superior")

            # 1) Si hay concepto_superior en el grafo, re-enrutamos aristas
            if sup and sup in G:
                # salientes: emergente -> X  pasa a  sup -> X
                for _, dest, attrs in list(G.out_edges(nodo, data=True)):
                    if dest == sup:
                        continue
                    if not G.has_edge(sup, dest):
                        G.add_edge(sup, dest, **attrs)

                # entrantes: X -> emergente  pasa a  X -> sup
                for src, _, attrs in list(G.in_edges(nodo, data=True)):
                    if src == sup:
                        continue
                    if not G.has_edge(src, sup):
                        G.add_edge(src, sup, **attrs)

            # 2) Eliminar el nodo emergente feo
            G.remove_node(nodo)
            eliminados.append(nodo)

    print(f"♻️ Eliminados emergentes sintéticos: {eliminados}")
    return G


def reparar_emergentes_sin_equilibrio(G):
    reparados = 0

    for nodo in list(G.nodes()):  # ✅ solución al error
        if G.nodes[nodo].get("tipo") != "emergente":
            continue

        # Verificar si tiene conexión saliente hacia un equilibrio
        tiene_equilibrio = any(
            G.has_edge(nodo, succ) and G.nodes[succ].get("tipo") == "equilibrio"
            for succ in G.successors(nodo)
        )

        if tiene_equilibrio:
            continue

        dualidades = G.nodes[nodo].get("dualidades", [])
        if not isinstance(dualidades, list):
            continue

        # Reparar dualidades si están en formato incorrecto
        if all(isinstance(d, list) and len(d) == 2 for d in dualidades):
            dualidades = [tuple(d) for d in dualidades]
            G.nodes[nodo]["dualidades"] = dualidades

        # Crear un nodo de equilibrio artificial
        nombre_eq = f"equilibrio_{nodo}"
        if nombre_eq not in G:
            G.add_node(nombre_eq, tipo="equilibrio", nivel_conceptual=2, origen="auto-generado")

        G.add_edge(nodo, nombre_eq, weight=2.5)

        for a, b in dualidades:
            G.add_edge(nombre_eq, a, weight=1.2)
            G.add_edge(nombre_eq, b, weight=1.2)

        reparados += 1
        print(f"🔁 Reparado '{nodo}' → añadido '{nombre_eq}' como nuevo equilibrio.")

    print(f"✅ Emergentes sin equilibrio reparados: {reparados}")
    return G
def limpiar_nombres_con_sufijo_emergente(G):
    cambios = 0
    for nodo in list(G.nodes()):
        if nodo.endswith("_emergente"):
            base = nodo
            while base.endswith("_emergente"):
                base = base[:-10]
            if base in G.nodes():
                print(f"⚠️ Nodo base '{base}' ya existe, no se renombra '{nodo}'")
            else:
                nx.relabel_nodes(G, {nodo: base}, copy=False)
                G.nodes[base]["tipo"] = "emergente"
                cambios += 1
                print(f"♻️ Nodo renombrado: {nodo} → {base}")
    print(f"✅ Nodos renombrados: {cambios}")
    for nodo in list(G.nodes()):
        if nodo.endswith("_emergente") and nodo[:-10] in G.nodes():
            G.remove_node(nodo)
            print(f"🗑 Nodo redundante eliminado: {nodo}")

    return G

def nodos_sospechosos(G):
    sospechosos = set()
    focos = ["fondo", "detrás", "consciencia"]

    for foco in focos:
        if foco in G:
            vecinos_salientes = set(G.successors(foco)) if G.is_directed() else set(G.neighbors(foco))
            vecinos_entrantes = set(G.predecessors(foco)) if G.is_directed() else set()

            for vecino in vecinos_salientes.union(vecinos_entrantes):
                if not isinstance(vecino, str) or vecino.strip() == "" or len(vecino.strip()) <= 2:
                    sospechosos.add(vecino)
    
    return sospechosos



if __name__ == "__main__":
    print("🔄 Buscando json/grafo.json")
    # Cargar el grafo desde "json/grafo_consciencia.json"
    G_archivo = cargar_grafo()

    # Cargar la red desde "json/red_fractal.json"
    G_red, diccionario = cargar_red()

    # Fusionar ambos grafos (si es necesario)
    #G = nx.compose(G_archivo, G_red)  # Une los dos grafos sin perder nodos ni conexiones
    G = fusionar_grafos(G_archivo, G_red)
    G = asignar_niveles_por_defecto(G)
    print("🔄 Reorganizando la red para conectar nodos en espera...")
    G = reorganizar_red(G)
    #G.remove_node("IA_m")
    guardar_red(G)
    GLOBAL_G = G
    
#    nodos_vacios = [n for n in G.nodes if not n.strip()]
#    for n in nodos_vacios:
#        nx.relabel_nodes(G, {n: "vacío"}, copy=False)
#    if "yo" in G and "vacío" in G:
#        G.add_edge("vacío", "yo", weight=0.8)
#        G.add_edge("yo", "vacío", weight=0.2)   
#    añadir_a_visualizacion("vacío", ["yo"], G)
#    guardar_red(G)

    #guardar_visualizacion_dinamica()

    for nodo in G.nodes():
        if not nodo or not isinstance(nodo, str) or nodo.strip() == "":
            print(f"⚠️ Nodo inválido encontrado: {repr(nodo)}")
    print("🕵️‍♂️ Nodos sospechosos:", nodos_sospechosos(G))    
    
    print(f"📊 Nodos cargados: {len(G.nodes())}")
    print(f"🔗 Conexiones cargadas: {len(G.edges())}")
    print("🌌 Nodos sueltos en subconsciente:", [nodo for nodo in G.nodes() if G.degree(nodo) == 0])

    if NODO_CENTRAL:
        # Contar cuántos nodos están directamente conectados al nodo central
        conexiones_centrales = Counter([nodo for nodo in G.neighbors(NODO_CENTRAL)])

        # Ajustamos el umbral a un número más bajo
        umbral_conexiones = 2  # Ajusta este número según lo que desees mostrar

        print("🔍 Nodos con muchas conexiones al nodo central:")
        for nodo, conexiones in conexiones_centrales.items():
            if conexiones > umbral_conexiones:  # Ajusta el nivel de agrupación que desees
                print(f"{nodo}: {conexiones} conexiones")
        # Ahora, visualizamos las conexiones de los nodos principales (si los hay)
        print(f"🔗 Conexiones del nodo central '{NODO_CENTRAL}':")
        for vecino in G.neighbors(NODO_CENTRAL):
            print(f"{vecino}: {G.degree(vecino)} conexiones")
    try:
        while True:
            entrada = input("\n🤔 Opciones: [salir] [consultar] [sistema] [añadir] [expandir] [historial] [ver red] [ver hipercubo] [auditar] [auto]: ").strip().lower()

            if entrada == "salir":
                print("👋 Saliendo... Guardando cambios en la red.")
                expansion_activa = False
                retirar_atencion(G)
                guardar_red(G)
                visualizar_red(G)
                break  # 🔹 Evita usar sys.exit(0), permitiendo un cierre más natural
                
            elif entrada == "sistema":
                concepto = input("🔎 Nodo base del sistema dual: ").strip().lower()
                visualizar_sistema_dual(concepto, G)

            elif entrada == "ver red":
                visualizar_red(G)
                #generar_subgrafos_principales(G, top_n=100)
                generar_subgrafos_principales(G)

                print("🌟 Conceptos en la red:", {len(G.nodes())})

                # Guardar gráfico en archivo en lugar de mostrarlo en pantalla
                plt.figure(figsize=(10, 5))
                plt.hist([G.degree(nodo) for nodo in G.nodes()], bins=10, color="blue", alpha=0.7)
                plt.xlabel("Número de conexiones por nodo")
                plt.ylabel("Cantidad de nodos")
                plt.title("Distribución de Conexiones en la Red Fractal")
                plt.grid()
                plt.savefig("grafico_red.png")
                plt.close()  # 🔧 Importante para liberar memoria
                print("🎨 Gráfico guardado como 'grafico_red.png'. Ábrelo manualmente.")

            elif entrada == "añadir":
                print("➞ Introduce conceptos:")
                print("➞ INFO: Si el concepto tiene espacios usa \"_\" para separar palabras")

                print("   - Conceptos sueltos:     concepto1 concepto2 ...")
                print("   - Dualidad:              A/B")
                print("   - Triada:                A/B-C  (ej: tesis/antítesis-síntesis)")
                dato = input("➞ ").strip()

                # ----- 1️⃣ Detección de triadas A/B-C -----
                if "/" in dato and "-" in dato:
                    try:
                        parte_dual, parte_equilibrio = dato.split("-", 1)
                        a, b = parte_dual.split("/", 1)
                        a = a.strip().lower().replace("_", " ")
                        b = b.strip().lower().replace("_", " ")
                        c = parte_equilibrio.strip().lower().replace("_", " ")
            
                        print(f"🔺 Añadiendo TRÍADA: {a} ↔ {b} → {c}")

                        # 1) Crear dualidad A/B
                        agregar_dualidad(G, a, b)

                        # 2) Crear nodo de equilibrio C
                        if c not in G:
                            G.add_node(c, tipo="equilibrio", nivel_conceptual=2, es_sintesis=True)

                        # 3) Conectar equilibrio con ambos extremos
                        for extremo in (a, b):
                            if not G.has_edge(c, extremo):
                                G.add_edge(c, extremo, tipo="equilibrio", color="goldenrod", weight=1.8)
                            if not G.has_edge(extremo, c):
                                G.add_edge(extremo, c, tipo="equilibrio", color="goldenrod", weight=1.2)

                        # 4) Guardar en diccionario
                        diccionario.setdefault(a, []).append(b)
                        diccionario.setdefault(b, []).append(a)
                        diccionario.setdefault(c, []).append(a)
                        diccionario[c].append(b)

                        guardar_diccionario(diccionario)
                        guardar_red(G)
                        print("✅ Triada añadida correctamente.\n")
                        continue

                    except Exception as e:
                        print(f"❌ Error interpretando triada: {e}")
                        continue

                # ----- 2️⃣ Detección de dualidad A/B -----
                if "/" in dato:
                    a, b = [x.strip().lower().replace("_", " ") for x in dato.split("/", 1)]
                    print(f"↔️ Añadiendo dualidad manual: {a} ↔ {b}")
                    agregar_dualidad(G, a, b)
                    guardar_red(G)
                    continue

                # ----- 3️⃣ Si no es triada ni dualidad, es lista de conceptos -----
                conceptos = dato.split()
                G = agregar_nuevo_nodo(G, diccionario, conceptos)
                guardar_red(G)


            elif entrada == "expandir":
                concepto = input("➞ Concepto a expandir: ").strip()
                print("📚 Expandiendo desde GPT...")
                #print("📚 Expandiendo desde Wikipedia y GPT...")
                resultado_wikipedia = ""#consultar_wikipedia(concepto, G, diccionario)
                nuevos_conceptos = consultar_chatgpt(concepto, diccionario)
                if nuevos_conceptos:
                    #G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
                    G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos, nodo_origen=concepto)
                
                # 🔹 Se restauró la verificación antes de expandir con embeddings
                if concepto in G.nodes():
                    print("🔄 Expandiendo con embeddings...")
                    expandir_concepto_embedding(concepto, G, diccionario)
                else:
                    print("⚠️ El concepto no está en la red. No se expandirá con embeddings.")

                guardar_red(G)

            elif entrada == "historial":
                ver_registro()
                generar_reportes()
            elif entrada == "ver hipercubo":
                visualizar_hipercubo_conceptual_3D(G)
            elif entrada == "auto":
                usar_wikipedia = False #input("🌍 ¿Deseas extraer información de Wikipedia? (s/n): ").strip().lower()
                usar_gpt = input("🧠 ¿Deseas usar ChatGPT para la expansión? (s/n): ").strip().lower()
                modo_expansion = input("¿Deseas despertar a IA_m (i), profundizar en tema interesante (p), llevarla a dormir (d)?: ").strip().lower()

                print("⚡ Iniciando IA_m! (Presiona Ctrl+C para detener)")
                
                expansion_activa = True

                try:
                    while expansion_activa:
                        expansion_automatica(G, diccionario, expansion_activa, modo_expansion, usar_wikipedia, usar_gpt)
                        expansion_dinamica(G, diccionario)
            
                        for _ in range(10):  # Intervalo corto para poder cortar con Ctrl+C
                            if not expansion_activa:
                                break
                            time.sleep(1)
                except KeyboardInterrupt:
                    print("\n⏹ Expansión detenida por el usuario.")
                    expansion_activa = False
                    retirar_atencion(G)
                    guardar_visualizacion_dinamica()
                    guardar_red(G)
                    
            elif entrada == "consultar":
                tema = input("🔎 Tema a consultar (subgrafo): ").strip().lower()
                generar_subgrafo_html(G, tema, modelo_embeddings, carpeta_subgrafos)
                
            elif entrada == "auditar":
                G = limpiar_nombres_con_sufijo_emergente(G)
                G = ejecutar_auditoria_y_ofrecer_reparacion(G)
                G = reparar_emergentes_sin_equilibrio(G)
                G = limpiar_emergentes_sintesis(G)
                guardar_red(G)
            elif entrada == "dualidad":
                pipeline_dualidades_auto(G, modelo_embeddings) 
            elif entrada == "triadas":
                print("\n🔍 Detectando triadas lineales (tipo pasado–presente–futuro–tiempo)...")
                triadas_lineales = detectar_triadas_lineales(G)
                if triadas_lineales:
                    for t in triadas_lineales:
                        print("📐 Triada lineal:", t)
                else:
                    print("⚠️ No se detectaron triadas lineales.")

                print("\n🔍 Detectando sistemas posicionales brutos...")
                sistemas_pos = detectar_sistemas_posicionales(G)
                for s in sistemas_pos:
                    print("📐 Sistema posicional:", s)

                print("\n🔍 Filtrando estructuras dimensionales coherentes...")
                estructuras_dim = filtrar_estructuras_dimensionales(G, sistemas_pos)
                for e in estructuras_dim:
                    print("📐 Estructura dimensional candidata:", e)

                print("\n✅ Fin del análisis de triadas / sistemas dimensionales.")
            else:
                print(f"⚠️ Entrada errónea '{entrada}'.")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupción detectada. Guardando y cerrando de manera segura...")
        expansion_activa = False
        retirar_atencion(G)
        guardar_visualizacion_dinamica()
        guardar_red(G)
        visualizar_red(G)

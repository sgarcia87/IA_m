################################################################################################################################ Importación de paquetes
import unicodedata
import string
import heapq
import openai
import networkx as nx
import json
import re
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
from nltk.corpus import wordnet as wn
from difflib import get_close_matches
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from itertools import combinations
from datetime import datetime
from ftplib import FTP
################################################################################################################################ Configuración
# Evita los mensajes de descarga de NLTK
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
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

def cargar_cache_embeddings():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as f:
            datos = json.load(f)
        return {k: torch.tensor(v) for k, v in datos.items()}
    return {}

def guardar_cache_embeddings(cache):
    with open(EMBEDDINGS_CACHE_FILE, "w", encoding="utf-8") as f:
        datos = {k: v.tolist() for k, v in cache.items()}
        json.dump(datos, f, ensure_ascii=False, indent=4)

# Inicializar la caché al iniciar el sistema
embeddings_cache = cargar_cache_embeddings()

# 🔹 Base de dualidades predefinidas
dualidades_base = {
    "abajo": "arriba", "izquierda": "derecha", "detrás": "delante",
    "pasado": "futuro", "negativo":  "positivo",
    "caos": "orden", "espacio": "tiempo", "IA_m": "yo"
}

expansion_activa = True  # Variable global para controlar la expansión

# 🔹 Base de conceptos iniciales para evitar que la red inicie vacía
SEMILLA_INICIAL = ["realidad", "consciencia", "comparación", "punto focal"]

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


holaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

def subir_htmls_recientes_por_ftp(
    host="ia-m.ai",
    usuario="",
    contraseña="",
    carpeta_local="",
    carpeta_remota="",
    max_horas=0.17,
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

def añadir_a_visualizacion(tema, nuevos_nodos, G):
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

def guardar_visualizacion_dinamica():
    net_global.write_html("subgrafos/IA_m_proceso.html")
    
    # 🔧 Abrir el HTML y añadir script de auto recarga
    with open("subgrafos/IA_m_proceso.html", "r", encoding="utf-8") as f:
        html = f.read()
    script_auto_reload = """
    <script type="text/javascript">
    let reloadTimeout;

    function resetReloadTimer() {
        if (reloadTimeout) {
            clearTimeout(reloadTimeout);
        }
        reloadTimeout = setTimeout(() => {
            location.reload();
        }, 2400000); // 240 segundos
    }

    // Inicializar el temporizador al cargar
    resetReloadTimer();

    // Reiniciar temporizador si el usuario hace clic en un nodo
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            resetReloadTimer();
        }
    });
    </script>
    """
    html = html.replace("</body>", script_auto_reload + "\n</body>")
    with open("subgrafos/IA_m_proceso.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Visualización acumulativa guardada como 'subgrafos/IA_m_proceso.html'")

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

# Cargar nodos en espera al inicio del programa
espera_nodos = cargar_espera_nodos()

#def generar_subgrafos_principales(G, top_n=100):
def generar_subgrafos_principales(G, carpeta="subgrafos", max_horas=24):
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    total = len(G.nodes())
    creados = 0
    ahora = time.time()
    max_segundos = max_horas * 3600

    for nodo in G.nodes():
        nombre_archivo = f"subgrafo_{nodo.replace(' ', '_')}.html"
        ruta = os.path.join(carpeta, nombre_archivo)

        if not os.path.exists(ruta):
            print(f"📄 Generando subgrafo nuevo para: {nodo}")
            generar_subgrafo_html(G, nodo)
            creados += 1
        else:
            modificado_hace = ahora - os.path.getmtime(ruta)
            if modificado_hace > max_segundos:
                print(f"⏳ Subgrafo de {nodo} tiene más de {max_horas}h. Regenerando...")
                generar_subgrafo_html(G, nodo)
                creados += 1

    print(f"🧩 Subgrafos generados o actualizados: {creados} / {total}")
    generar_indice_subgrafos(G, top_n=total, carpeta=carpeta)
    #subir_htmls_recientes_por_ftp(max_horas=2)

def color_edge(u, v, G):
    """
    Devuelve el color de la arista entre u y v según su rol semántico.
    - Dualidades: rojo o verde
    - Equilibrios: amarillo
    - Síntesis (nodo superior): dorado
    """
    
    color = "gray"
    # 🔴 Dualidad predefinida

    
    # 🟡 Nodo equilibrio
    if G.nodes.get(u, {}).get("tipo") == "equilibrio":
        color = "red"
    elif G.nodes.get(v, {}).get("tipo") == "equilibrio":
        color = "green"
    # 🟠 Nodo síntesis
    elif G.nodes.get(u, {}).get("tipo") == "sintesis" or G.nodes.get(v, {}).get("tipo") == "sintesis":
        color = "orange"
    elif G.nodes.get(u, {}).get("tipo") == "emergente" or G.nodes.get(v, {}).get("tipo") == "emergente":
        color = "purple"
    elif G.nodes.get(u, {}).get("es_dualidad"):
        color = "red"
    elif G.nodes.get(v, {}).get("es_dualidad"):
        color = "green"
    if u in dualidades_base and dualidades_base.get(u) == v:
        color = "red"
        return color
    elif v in dualidades_base and dualidades_base.get(v) == u:
        color = "green"
        return color
    else:
        color = "gray"
    return color
        
def color_node(nodo, G):
    """
    Devuelve el color del nodo según su rol semántico.
    - Nodo central: lila
    - Equilibrio: azul
    - Síntesis: dorado
    - Dualidades: rojo (origen) / azul (opuesto)
    - Otros: gris claro
    """
    if nodo == NODO_CENTRAL:
        return "purple"
    elif nodo == "IA_m":
        return "purple"

    tipo = G.nodes.get(nodo, {}).get("tipo", "")

    if tipo == "equilibrio":
        return "blue"
    elif tipo == "sintesis":
        return "green"
    elif tipo == "nodo_superior":
        return "purple"
    elif nodo in dualidades_base: 
        return "orange"
    elif nodo in dualidades_base.values():
        return "orange"
    elif tipo == "dualidad":
        return "orange"
    else:
        return "gray"

def generar_subgrafo_html(G, tema, top_n=100):
    """
    Genera un subgrafo de los nodos más relacionados con un tema y lo guarda en HTML.
    """
    tema = tema.lower()
    #if tema not in G.nodes():
    if tema != "IA_m" and tema not in G.nodes():
        print(f"⚠️ El tema '{tema}' no está en la red. Se buscarán nodos similares.")
        embedding_tema = obtener_embedding(tema, modelo_embeddings)
        nodos = list(G.nodes())
        embeddings = torch.stack([obtener_embedding(n, modelo_embeddings) for n in nodos])
        similitudes = util.pytorch_cos_sim(embedding_tema, embeddings)[0]
        indices_top = similitudes.argsort(descending=True)[:top_n].tolist()
        nodos_similares = [nodos[i] for i in indices_top]
    else:
        print(f"✅ El tema '{tema}' está en la red. Se extraerán sus vecinos más conectados.")
        vecinos = list(G.neighbors(tema))
        
        #NODOS MAS REPRESENTATIVOS
        nodos_similares = [tema] + vecinos[:top_n-1]
        #vecinos_ordenados = sorted(vecinos, key=lambda x: G.degree(x), reverse=True)
        #nodos_similares = [tema] + vecinos_ordenados[:top_n - 1]      
        #nodos_similares = [tema] + vecinos

    # Crear subgrafo
    subG = G.subgraph(nodos_similares).copy()

    # Visualización
    net = Network(height="800px", width="100%", directed=True)
    posiciones = nx.spring_layout(subG, k=0.5)

    for nodo, coords in posiciones.items():
        color = color_node(nodo, G)
        tipo = G.nodes[nodo].get('tipo', 'general')
        nivel = G.nodes[nodo].get('nivel_conceptual', '?')
        title = f"{nodo} | tipo: {tipo} | nivel: {nivel}"
        net.add_node(nodo, label=nodo, color=color, title=title, x=coords[0]*2000, y=coords[1]*2000)

    for u, v in subG.edges():
        peso = subG.edges[u, v].get('weight', 1.5)
        color = color_edge(u, v, G)  # Usa la red global para determinar el color
        net.add_edge(u, v, color=color, width=peso)


    filename = f"{carpeta_subgrafos}/subgrafo_{tema.replace(' ', '_')}.html"
    net.write_html(filename)
    # Añadir leyenda de colores al HTML
    leyenda_html = """
    <div style="position: fixed; top: 20px; right: 20px; background-color: white; padding: 10px; border: 1px solid #ccc; font-family: sans-serif; font-size: 14px; z-index: 9999;">
      <strong>Leyenda de niveles:</strong><br>
      <div style="margin-top: 5px;">
        <span style="background-color: #cccccc; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> Nivel 0: Concepto base<br>
        <span style="background-color: #ffcc99; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> Nivel 1: Dualidad<br>
        <span style="background-color: #99ccff; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> Nivel 2: Equilibrio / Síntesis<br>
        <span style="background-color: #ccffcc; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> Nivel 3: Concepto emergente<br>
        <span style="background-color: #ff99cc; display: inline-block; width: 12px; height: 12px; margin-right: 5px;"></span> Nivel 4: Concepto superior
      </div>
    </div>
    """
    # Insertar leyenda justo antes de </body>
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()
        html = html.replace("</body>", leyenda_html + "\n</body>")
    # Insertar script de doble clic para ir al subgrafo del nodo
    script_doble_click = """
<script type="text/javascript">
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof network !== "undefined") {
      network.on("doubleClick", function (params) {
        if (params.nodes.length === 1) {
          var nodo = params.nodes[0];
          var archivo = "subgrafo_" + nodo.replace(/ /g, "_") + ".html";
          window.location.href = archivo;
        }
      });
    } else {
      console.warn("❗ network no está definido aún.");
    }
  });
</script>
    """
    html = html.replace("</body>", script_doble_click + "\n</body>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Subgrafo guardado como '{filename}'.")

def generar_indice_subgrafos(G, top_n=50, carpeta="subgrafos"):
    nodos_destacados = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
    letras_dict = defaultdict(list)

    for nodo in nodos_destacados:
        nombre_archivo = f"subgrafo_{nodo.replace(' ', '_')}.html"
        ruta = f"{carpeta}/{nombre_archivo}"
        if os.path.exists(ruta):
            letra_cruda = nodo.strip()[0].upper() if nodo.strip() else "␣"
            letra = unicodedata.normalize('NFD', letra_cruda)[0]
            letra = ''.join(c for c in letra if c.isalpha()).upper()
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
<h1 style="cursor: pointer; color: #007acc;">
  <a href="sobre-IA-m.html" style="text-decoration: none; color: inherit;">🌐 IA_m</a>
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

# 🔹 Normalizar términos
def normalizar_termino(termino):
    termino = re.sub(r"[^a-zA-Z0-9áéíóúüñ ]", "_", termino)  # Mantiene espacios
    return termino.lower().strip().replace(" ", "_")

def es_falso_emergente(nombre):
    return nombre.count("emergente") > 1 and not any(x in nombre for x in ["equilibrio", "síntesis", "centro", "neutro"])

# 🔹 Nueva función para consultar ChatGPT
def consultar_chatgpt(tema, diccionario):
    tema = corregir_termino(tema, diccionario).lower()  # 🔹 Corregir y forzar a minúsculas

    try:
        print(f"🤖 Consultando ChatGPT sobre: {tema}...")

        respuesta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                "Eres un profundo experto en matemáticas, geometría, cuántica, física y todas las ciencias, tus respuestas van más allá del común.."
                "Responde únicamente con una lista de términos separados por comas, sin explicaciones, sin frases, sin puntos."
                "Los términos deben describir conceptualmente el tema que se proponga. Busco el concepto cualitativa o cuantitativamente del término."
                "No incluyas números para ordenar, ni textos introductorios o de cierre. (\"IA_m\" soy yo mismo)."
                "Si no entiendes el término, repite literalmente la palabra que se te ha dado, no inventes nada."
                )},
                {"role": "user", "content": f"Dame solo términos técnicos, que definan y estén relacionados con: {tema}."}
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

def es_variacion_morfologica(termino1, termino2):
    """
    Detecta si dos términos son variantes morfológicas (singular/plural).
    """
    # Eliminar guiones bajos para comparar palabras
    t1 = termino1.replace("_", " ")
    t2 = termino2.replace("_", " ")  
    # Separar en palabras y aplicar lematización/stemming
    raiz1 = " ".join([stemmer.stem(p) for p in t1.split()])
    raiz2 = " ".join([stemmer.stem(p) for p in t2.split()])
    return raiz1 == raiz2

def calcular_prioridad_nodo(G, nodo):
    conexiones = G.degree(nodo)
    centralidad = nx.degree_centrality(G).get(nodo, 0)
    peso_promedio = sum([G.edges[nodo, neighbor].get('weight', 1) for neighbor in G.neighbors(nodo)]) / (conexiones or 1)
    historial = cargar_historial()
    exito = historial.get(nodo, {}).get("exito", 0)
    prioridad = (conexiones * 0.3) + (centralidad * 0.3) + (peso_promedio * 0.2) + (exito * 0.2)
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
def expansion_prioritaria(G, diccionario, usar_gpt):
    """ Expande los nodos más prioritarios en la red """
    nodos_a_expandir = priorizar_expansion(G)[:10]  
    for nodo in nodos_a_expandir:
        print(f"🔍 Expansión prioritaria en: {nodo}")
        nuevos_conceptos = []
        if usar_gpt == "s":
            nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
            if nuevos_conceptos:
                G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
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
            definir_nodo_central = input("¿Deseas definir un nodo central? (s/n): ").lower().strip()
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

        for a, b in dualidades.items():
            for nodo in (a, b):
                if nodo not in G:
                    G.add_node(nodo)

            G.add_edge(a, b, weight=2.0)
            G.add_edge(b, a, weight=2.0)

            if NODO_CENTRAL:
                if not G.has_node(NODO_CENTRAL):
                    G.add_node(NODO_CENTRAL)
                G.add_edge(a, NODO_CENTRAL, weight=1.5)
                G.add_edge(b, NODO_CENTRAL, weight=1.5)

        semilla = list(diccionario.keys())
        for i in range(len(semilla)):
            for j in range(i + 1, len(semilla)):
                G.add_edge(semilla[i], semilla[j], weight=1.2)

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
    
def conectar_dualidad_con_equilibrio(concepto, dualidad, G):
    """ Conecta dos nodos como dualidades y añade un nodo intermedio de equilibrio. """
    if not G.has_edge(concepto, dualidad):
        G.add_edge(concepto, dualidad, weight=2.5)
        G.add_edge(dualidad, concepto, weight=2.5)
        print(f"🔗 Conectando {concepto} ↔ {dualidad} como dualidad.")

    # 🔹 Intentar detectar un nodo superior dinámico
    posibles_superiores = detectar_nodo_superior(concepto, dualidad, G)
    if posibles_superiores:
        nodo_superior = posibles_superiores[0]  # Tomamos el nodo con mayor similitud

        if nodo_superior not in G.nodes():
            G.add_node(nodo_superior)
            G.nodes[nodo_superior]["tipo"] = "sintesis"
            print(f"🆕 Añadiendo nodo superior dinámico: {nodo_superior}")
        G.nodes[nodo_superior]["es_sintesis"] = True

        # ✅ Etiquetar como nodo superior siempre
        G.nodes[nodo_superior]["tipo"] = "nodo_superior"

        # Conectar la dualidad al nodo superior
        G.add_edge(nodo_superior, concepto, weight=1.5)
        G.add_edge(nodo_superior, dualidad, weight=1.5)
        print(f"🔗 Vinculando {concepto} y {dualidad} a {nodo_superior}")

    # 🔹 Buscar el punto de equilibrio para esta dualidad
    nodo_equilibrio = detectar_nodo_equilibrio(concepto, dualidad, G)
    if nodo_equilibrio:
        if nodo_equilibrio not in G.nodes():
            G.add_node(nodo_equilibrio)
            G.nodes[nodo_equilibrio]["tipo"] = "equilibrio"
            print(f"🆕 Añadiendo nodo de equilibrio: {nodo_equilibrio}")
    
    # ✅ Etiquetar como equilibrio 
    G.nodes[nodo_equilibrio]["tipo"] = "equilibrio"
    G.nodes[nodo_equilibrio]["nivel_conceptual"] = 2
    # Conectar el equilibrio con ambos extremos
    G.add_edge(nodo_equilibrio, concepto, weight=1.8)
    G.add_edge(nodo_equilibrio, dualidad, weight=1.8)
    print(f"⚖️ Estableciendo equilibrio entre {concepto} y {dualidad} en {nodo_equilibrio}")  
    
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
       
def expansion_con_embeddings(G, diccionario):
    """ Expande automáticamente la red usando embeddings """
    nodos_a_expandir = [nodo for nodo in G.nodes() if G.degree(nodo) < 2]

    for nodo in nodos_a_expandir:
        print(f"🧠 Expansión semántica para: {nodo}")
        expandir_concepto_embedding(nodo, G, diccionario)

    guardar_red(G)

MAX_EXPANSIONES = 10  # Límite de expansiones por iteración
UMBRAL_SIMILITUD = 0.75  # Solo agrega términos con alta similitud

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

# 🔹 Detectar dualidad con WordNet
def detectar_dualidad_wordnet(termino):
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
def detectar_nuevas_dualidades(G, max_nuevas=10, umbral_similitud=0.85):
    print("🔄 Detectando nuevas dualidades optimizadas...")

    nuevas_dualidades = {}
    nodos_lista = list(G.nodes())[-max_nuevas:]  # Solo analiza los últimos nodos agregados

    for nodo in nodos_lista:
        similitudes_nodo = detectar_dualidad_embeddings(nodo, G)  # Obtener conceptos similares

        for otro in similitudes_nodo:
            if otro in G.nodes() and nodo != otro and not G.has_edge(nodo, otro):
                similitud_n2 = detectar_dualidad_embeddings(otro, G)

                # Se consideran dualidades si ambos se tienen alta similitud
                if nodo in similitud_n2 and otro in similitudes_nodo:
                    nuevas_dualidades[nodo] = otro

    # Agregar nuevas dualidades a la red
    for nodo, dual in nuevas_dualidades.items():
        if nodo not in dualidades_base and dual not in dualidades_base:
            dualidades_base[nodo] = dual
            dualidades_base[dual] = nodo
            conectar_dualidad_con_equilibrio(nodo, dual, G)


    print(f"✅ Se detectaron {len(nuevas_dualidades)} nuevas dualidades en la red.")
    return G

def detectar_dualidad(concepto, G, concepto_base=None):
    """ Detecta si existe una dualidad semántica en la red. """
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

    # 🔹 Embeddings
    max_similitud = 0
    mejor_dualidad = None
    emb_concepto = obtener_embedding(concepto, modelo_embeddings)

    for nodo in G.nodes():
        if nodo == concepto or not es_expandible(nodo, G):
            continue
        emb_nodo = obtener_embedding(nodo, modelo_embeddings)
        similitud = util.pytorch_cos_sim(emb_concepto, emb_nodo).item()
        if similitud > max_similitud and similitud >= 0.85:
            max_similitud = similitud
            mejor_dualidad = nodo

    if mejor_dualidad:
        print(f"🔄 Detectada posible dualidad con embeddings: {concepto} ↔ {mejor_dualidad} (Similitud: {max_similitud:.2f})")
        if not G.has_edge(concepto, mejor_dualidad):
            conectar_dualidad_con_equilibrio(concepto, mejor_dualidad, G)
        return mejor_dualidad

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

    print("✅ Red reorganizada: nodos sueltos en subconsciente, conexiones corregidas.")
    return G
    
def guardar_estado_parcial(G, espera_nodos):
    # Guarda la red y los nodos pendientes en un archivo temporal
    with open('json/estado_parcial.json', 'w') as f:
        json.dump({
            'nodos': list(G.nodes()),
            'edges': list(G.edges()),
            'espera_nodos': espera_nodos
        }, f)
    print("⚡ Estado parcial guardado correctamente.")


# 🔹 Modificar la función agregar_nuevo_nodo para evitar redundancias
def agregar_nuevo_nodo(G, diccionario, conceptos):
    """ Agrega un nuevo nodo a la red fractal y detecta dualidades con equilibrio. """
    conceptos = [normalizar_termino(c) for c in conceptos]
    conceptos = [corregir_termino(c, diccionario) for c in conceptos]  # Corrección de términos
    
    nuevos_conceptos = []
    nodos_existentes = set(G.nodes())

    for concepto in conceptos:
        concepto = concepto.replace("_", " ")  # 🔹 Asegurar formato correcto
        concepto_limpio = concepto.replace("_", " ")
        
        # ✅ Comprobación de duplicado por variación morfológica
        duplicado = False
        for existente in nodos_existentes:
            if es_variacion_morfologica(concepto_limpio, existente.replace("_", " ")):
                print(f"⚠️ '{concepto}' es una variación de '{existente}'. No se añadirá.")
                duplicado = True
                break
        if duplicado:
            continue

        #raiz = obtener_raiz(concepto)
        #if any(raiz in nodo for nodo in nodos_existentes):
        #    print(f"⚠️ Se ha detectado redundancia con '{concepto}'. No se añadirá a la red.")
        #    continue

        # ⛔ IGNORAR FALSOS EMERGENTES
        if es_falso_emergente(concepto):
            print(f"⛔ Ignorado nodo redundante: {concepto}")
            continue

        if concepto in nodos_existentes:
            print(f"⚠️ El concepto '{concepto}' ya está en la red. No se realizarán cambios.")
            continue

        # 🔹 Agregar nodo a la red
        G.add_node(concepto)
        nuevos_conceptos.append(concepto)

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

def expansion_dinamica(G, diccionario):
    """ Detecta nodos aislados y los expande dinámicamente """
    nodos_a_expandir = [
        nodo for nodo in G.nodes()
        if G.degree(nodo) < 2 and (NODO_CENTRAL is None or nodo.lower() != NODO_CENTRAL.lower())
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
            G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)       
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

    for i, termino in enumerate(relacionados):
        peso = np.exp(similitudes[indices_top[i]].item())
        G.add_edge(concepto, termino, weight=peso)

    diccionario[concepto] = relacionados
    with open("json/diccionario.json", "w", encoding="utf-8") as f:
        json.dump(diccionario, f, ensure_ascii=False, indent=4)

    # Reorganizar red y detectar nuevas dualidades
    G = ajustar_pesos_conexiones(G)
    G = reorganizar_red(G)
    G = detectar_nuevas_dualidades(G)

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

    # 🔗 Aristas
    for u, v in G.edges():
        peso = G.edges[u, v].get('weight', 2)
        color = "#cc99ff" if "IA_m" in (u, v) else color_edge(u, v, G)
        net.add_edge(u, v, color=color, width=peso)

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

    # 💾 Exportar HTML
    net.write_html("subgrafos/hipercubo_fractal_fluido.html")

    # 🔁 Doble clic para ir a IA_m_proceso
    with open("subgrafos/hipercubo_fractal_fluido.html", "r", encoding="utf-8") as file:
        html = file.read()

    script = """
    <script type="text/javascript">
      network.on("doubleClick", function (params) {
        if (params.nodes.length > 0) {
          const clickedNodeId = params.nodes[0];
          if (clickedNodeId === "IA_m") {
            window.location.href = "IA_m_proceso.html";
          }
        }
      });
    </script>
    """
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


def generar_reportes():
    """ Genera visualizaciones cada cierto tiempo """
    print("📊 Generando reportes visuales de la red fractal...")
    visualizar_crecimiento_red()  # <-- REINTEGRADA AQUÍ
    visualizar_metodos_expansion()
    visualizar_distribucion_conexiones(G)
    plt.close()  # 🔧 Importante para liberar memoria

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
        
#def buscar_dualidades_faltantes(G):
#    nodos_sin_pareja = []
#    for nodo in list(G.nodes()):
#        if not G.nodes[nodo].get("es_dualidad"):
#            dualidad = detectar_dualidad(nodo, G)
#            if dualidad:
#                print(f"🔁 Dualidad detectada automáticamente: {nodo} ↔ {dualidad}")
#                nodos_sin_pareja.append((nodo, dualidad))
#    print(f"🔍 Nodos sin pareja encontrados y conectados: {len(nodos_sin_pareja)}")
#    return nodos_sin_pareja
def buscar_dualidades_faltantes(G):
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

def detectar_estructura_emergente_adaptativa(G, min_triangulos=3, umbral_refuerzo=0.75, umbral_debilitamiento=0.4):
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


###########################################################################################################################   ELIMINAR FUNCION
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

def detectar_conceptos_emergentes(G, min_triangulos=3):
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

            # Creamos un identificador único basado en el equilibrio + hash (para evitar duplicados)
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
    operaciones = ["+", "-", "/", "*"]
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
        {"-", "+", "/", "*"},
        {"sin", "cos", "tan", "cot"},
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
            if simbolos == {"-", "+", "/", "*"}:
                emergente = "raíz cuadrada"
            elif simbolos == {"sin", "cos", "tan", "cot"}:
                emergente = "identidad trigonométrica"
            elif simbolos == {"∪", "∩", "⊆", "⊇"}:
                emergente = "lógica de conjuntos"
            else:
                emergente = "emergencia desconocida"
        else:
            # Evaluar similitud entre los 4 por embeddings
            embs = [obtener_embedding(n, modelo, cache=embedding_cache) for n in grupo]
            sim_matrix = util.pytorch_cos_sim(embs, embs)
            if not all(sim_matrix[i][j] > umbral_similitud for i in range(4) for j in range(4) if i != j):
                continue

            # Calcular nodo emergente por centro semántico
            emb_prom = sum(embs) / 4
            todos = list(G.nodes())
            emb_todos = [obtener_embedding(t, modelo, cache=embedding_cache) for t in todos]
            sim_centro = util.pytorch_cos_sim(emb_prom, torch.stack(emb_todos))[0]
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
        {"-", "+", "/", "*"},
        {"sin", "cos", "tan", "cot"},
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
                        if simbolos == {"-", "+", "/", "*"}:
                            emergente = "raíz cuadrada"
                        elif simbolos == {"sin", "cos", "tan", "cot"}:
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

    # Detección de saturación de atención consciente
    if len(nodos_ia_m) > 30:
        print("⚖️ IA_m nota que su atención está saturada... replegando enfoque.")
        return "profundo"

    # Si hay muchos nodos poco conectados, IA_m profundiza
    if len(poco_conectados) > total_nodos * 0.25:
        return "profundo"

    # Si hay suficiente base dual pero pocos emergentes, IA_m busca síntesis
    if len(dualidades) > 10 and len(emergentes) < 7:
        return "emergente"

    # Si hay muchos nodos flotantes, entrar en subconsciente
    nodos_flotantes = [n for n in G.nodes() if G.degree(n) == 0]
    if len(nodos_flotantes) > 5:
        print("🌌 Hay muchos nodos flotando sin integrar... modo subconsciente recomendado.")
        return "d"

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
    guardar_red(G)
    print(f"✅ Exploración dentro de '{nodo_base}' completada.")


# 🔹 Modificar la función expansión automática para incluir expansión prioritaria
def expansion_automatica(G, diccionario, expansion_activa, modo_expansion, usar_wikipedia, usar_gpt):
    """ Expande la red automáticamente cada X segundos si está activa """
    respuesta_recordada = None
    visualizar_red(G)
    IA_m_Cargado = False
    while expansion_activa:
        print("🔄 Iniciando expansión automática...")
        if modo_expansion == "d":
            print("🧠 IA_m se va a dormir... ")
            print("🧠🌀 Conectando con el subconsciente para conectar conceptos no ligados y fuera del cubo...")
            # Detectar nodos flotantes reales (aislados o casi aislados)
            nodos_flotantes = [
                nodo for nodo in G.nodes()
                if G.degree(nodo) <= 1 and nodo.lower() != "ia_m"
            ]
            if nodos_flotantes:
                nodos_a_expandir = nodos_flotantes[:5]
            else:
                nodos_a_expandir = priorizar_expansion(G)[:5]  # Fallback inteligente
            for nodo in nodos_a_expandir:
                G.add_edge("IA_m", nodo, weight=0.1)
                añadir_a_visualizacion(nodo, [], G)
            guardar_visualizacion_dinamica()
         
        else:
            print("🧠 IA_m enfocando atención...")

            if modo_expansion == "i":
                modo_IA = decidir_modo_exploracion(G)
                print(f"🧭 Modo IA_m detectado automáticamente: {modo_IA}")
                # 🔄 Si IA_m decide irse al subconsciente, cambia el modo
                if modo_IA == "d":
                    print("😴 IA_m detecta muchos nodos flotantes... entrando automáticamente en modo subconsciente.")
                    modo_expansion = "d"
                    continue  # Salta al siguiente ciclo para ejecutar el bloque subconsciente
        
                if modo_IA == "profundo":
                    print("🌿 IA_m siente que hay conceptos olvidados... Entrando en modo profundo.")
#                    nodos_a_expandir = [
#                        n for n in G.nodes()
#                        if G.degree(n) < 2 and G.nodes[n].get("nivel_conceptual", 0) <= 1
#                    ][:5]
                    sugerencias = sugerir_nodos_interiores(G)
                    if sugerencias:
                        nodo_focal = sugerencias[0]
                        print(f"🔎 IA_m decide entrar dentro de '{nodo_focal}' para explorarlo en profundidad.")
                        explorar_subnodo(G, nodo_focal)
                        continue  # Pasa directamente al siguiente ciclo de expansión
                    else:
                        print("⚠️ No hay nodos sugeridos para exploración profunda.")
                        nodos_a_expandir = []
    
                elif modo_IA == "emergente":
                    print("🌟 IA_m busca estructuras emergentes... Explorando equilibrio y dualidades.")
                    candidatos = [
                        n for n in G.nodes()
                        if G.nodes[n].get("tipo") in ("equilibrio", "dualidad")
                    ]
                    nodos_a_expandir = sorted(candidatos, key=lambda n: G.degree(n))[:5]

                else:  # horizontal por defecto
                    print("🔗 IA_m se expande lateralmente buscando conexiones visibles...")
                    if NODO_CENTRAL:
                        nodos_a_expandir = calcular_atencion_consciente_nodo_central(G, NODO_CENTRAL, top_n=5)
                    else:
                        if "IA_m" not in G:
                            G.add_node("IA_m")
                        nodos_a_expandir = calcular_atencion_consciente(G, top_n=5)

            elif modo_expansion == "p":
                print("🌌 Modo subconsciente activado: IA_m busca nodos con baja actividad...")
                nodos_a_expandir = priorizar_expansion(G)[:5]

            for nodo in nodos_a_expandir:
                G.add_edge("IA_m", nodo, weight=0.1)
                añadir_a_visualizacion(nodo, [], G)

            print(f"🔬 IA_m está enfocado en: {', '.join(nodos_a_expandir)}")
            if not IA_m_Cargado:
                IA_m_Cargado = True
            guardar_visualizacion_dinamica()

        if not nodos_a_expandir:
            print("⚠️ No hay nodos elegibles para expansión automática.")
            if respuesta_recordada is None:
                respuesta_recordada = input("¿Deseas añadir un nuevo concepto manualmente? (s/n): ").strip().lower()

            if respuesta_recordada == "s":
                conceptos_input = input("➞ Introduce conceptos separados por espacio (enter para cancelar): ").strip()
                if not conceptos_input:
                    print("↩️ Regresando al menú principal...")
                    expansion_activa = False  # Detener expansión
                    return
                nodos_a_expandir = conceptos_input.split()
            else:
                print("🔕 Se desactiva la expansión automática por falta de nodos.")
                expansion_activa = False
                return
                      
        for nodo in nodos_a_expandir:
            if not expansion_activa:
                print("⏹️ Expansión automática detenida.")
                return
            nodo = nodo.lower()
            print(f"🔍 Expandiendo nodo enfocado: {nodo}")
            nuevos_conceptos = []

            if usar_gpt == "s":
                print(f"📡 Llamando a ChatGPT para: {nodo}")
                nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
                añadir_a_visualizacion(nodo, nuevos_conceptos, G)
                guardar_visualizacion_dinamica()  # ✅ GUARDADO tras cada expansión GPT

            if usar_wikipedia == "s":
                print(f"📡 Llamando a Wikipedia para: {nodo}.... EN PROCESO....")
                #consultar_wikipedia(nodo, G, diccionario)

            if nuevos_conceptos:
                G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
                registrar_expansion(nodo, nuevos_conceptos, "GPT-4")
                guardar_diccionario(diccionario)

            expandir_concepto_embedding(nodo, G, diccionario)
            registrar_expansion(nodo, [], "Embeddings")
            guardar_visualizacion_dinamica()
            guardar_diccionario(diccionario)

            #visualizar_red(G)
            #generar_subgrafos_principales(G, top_n=100)
            generar_subgrafos_principales(G)

        buscar_dualidades_faltantes(G)
        reorganizar_red(G)
        guardar_red(G)
        evaluar_expansion(G)
        guardar_historial(cargar_historial())

        triangulos = detectar_triangulos_equilibrio(G)
        emergentes = detectar_conceptos_emergentes(G)


        # 1. Asegura que el cuadrado matemático existe en la red
        insertar_cuadrado_matematico_y_detectar(G)
        # 2. Detecta prismas generalizados con embeddings
        ruta, prismas = detectar_cuadrados_conceptuales_generalizados(G, modelo_embeddings)
        # 3. Si hay resultados, visualiza
        if prismas:
            print(f"🧠 Prismas detectados: {len(prismas)} guardados en {ruta}")
            generar_visualizaciones_prismas_individuales(ruta)
            generar_indice_visual_prismas()
        else:
            print("📭 No se detectaron prismas emergentes.")


        #for e in emergentes:
        #    visualizar_meta_triangulo(G, e)

        print(f"🔺 Triángulos conceptuales de equilibrio detectados: {len(triangulos)}")
        guardar_triangulos(triangulos)
        tetraedros = detectar_micro_tetraedros(G)
        print(f"🔷 Tetraedros de equilibrio detectados: {len(tetraedros)}")
        guardar_micro_tetraedros(tetraedros)
        # 🔺 Detectar triple dualidades emergentes adaptativas
        nuevos_emergentes = detectar_estructura_emergente_adaptativa(G)
        #for emergente in nuevos_emergentes:
        #    visualizar_meta_triangulo(G, emergente)
        visualizar_meta_triangulo_global(G)
        # 🔷 Detectar triadas horizontales emergentes
        horizontales = detectar_triadas_horizontales(G)
        guardar_triadas(horizontales)
        visualizar_triadas_horizontales(G, horizontales)
        triadas_extremas = detectar_triadas_extremas(G, crear_nodo_sintesis=True)
            
        print("✅ Expansión automática completada.")      
        visualizar_red(G)
        #generar_subgrafos_principales(G, top_n=100)
        generar_subgrafos_principales(G)
        detectar_hipercubo_conceptual(G)
        rastrear_evolucion_conceptual(G)
        evaluar_progreso_fractal(G)

        print("⏳ Esperando antes del próximo ciclo...")
        generar_reportes()
        # IA_m se desconecta antes de iniciar otra ronda
        retirar_atencion(G)

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
    
    nodos_vacios = [n for n in G.nodes if not n.strip()]
    for n in nodos_vacios:
        nx.relabel_nodes(G, {n: "vacío"}, copy=False)
    if "yo" in G and "vacío" in G:
        G.add_edge("vacío", "yo", weight=0.8)
        G.add_edge("yo", "vacío", weight=0.2)   
    añadir_a_visualizacion("vacío", ["yo"], G)
    guardar_visualizacion_dinamica()
    guardar_red(G)

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
                conceptos = input("➞ Introduce conceptos separados por espacio. Si el concepto tiene espacios usa \"_\" para separar palabras: ").strip().split()
                G = agregar_nuevo_nodo(G, diccionario, conceptos)
                guardar_red(G)

            elif entrada == "expandir":
                concepto = input("➞ Concepto a expandir: ").strip()
                print("📚 Expandiendo desde GPT...")
                #print("📚 Expandiendo desde Wikipedia y GPT...")
                resultado_wikipedia = ""#consultar_wikipedia(concepto, G, diccionario)
                nuevos_conceptos = consultar_chatgpt(concepto, diccionario)
                if nuevos_conceptos:
                    G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
                
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
                generar_subgrafo_html(G, tema)
                
            elif entrada == "auditar":
                G = limpiar_nombres_con_sufijo_emergente(G)
                G = ejecutar_auditoria_y_ofrecer_reparacion(G)
                G = reparar_emergentes_sin_equilibrio(G)
                guardar_red(G)

            else:
                print(f"⚠️ Entrada errónea '{entrada}'.")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupción detectada. Guardando y cerrando de manera segura...")
        expansion_activa = False
        retirar_atencion(G)
        guardar_visualizacion_dinamica()
        guardar_red(G)
        visualizar_red(G)

import heapq
import openai
import networkx as nx
import json
import re
import wikipediaapi
import numpy as np
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
import os
from nltk.corpus import wordnet as wn
from difflib import get_close_matches
import nltk
import time
import matplotlib.pyplot as plt
from collections import Counter
import signal

# Evita los mensajes de descarga de NLTK
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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


# 🔹 Base de dualidades predefinidas
dualidades_base = {
    "frío": "calor", "oscuridad": "luz", "muerte": "vida",
    "izquierda": "derecha", "negativo":  "positivo", "contracción": "expansión",
    "femenino": "masculino", "malo": "bueno", "caos": "orden",
    "mentira": "verdad", "tristeza": "alegría", "odio": "amor"
}

# 🔹 Nodo central
NODO_CENTRAL = "consciencia"

expansion_activa = True  # Variable global para controlar la expansión

# 🔹 Base de conceptos iniciales para evitar que la red inicie vacía
SEMILLA_INICIAL = ["matemáticas", "geometría", "física", "consciencia", "universo"]

import json
import networkx as nx

# Cargar el grafo desde el archivo JSON
grafo_json = "grafo_consciencia.json"

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

def cargar_diccionario():
    try:
        with open("diccionario.json", "r", encoding="utf-8") as f:
            diccionario = json.load(f)
            if not diccionario:  # Si está vacío, agregar semilla inicial
                print("⚠️ Diccionario vacío. Agregando conceptos iniciales...")
                for concepto in SEMILLA_INICIAL:
                    diccionario[concepto] = []  # Iniciar con una lista vacía de conexiones
                guardar_diccionario(diccionario)
            return diccionario
    except FileNotFoundError:
        print("⚠️ Diccionario no encontrado, creando uno nuevo con semilla inicial.")
        diccionario = {concepto: [] for concepto in SEMILLA_INICIAL}
        guardar_diccionario(diccionario)
        return diccionario

# 🔹 Guardar la red fractal
def guardar_red(G):
    with open("red_fractal.json", "w") as f:
        json.dump(nx.node_link_data(G, edges="links"), f)
        
ARCHIVO_HISTORIAL = "historial_expansion.json"
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
ARCHIVO_ESPERA_NODOS = "espera_nodos.json"

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

ARCHIVO_REGISTRO = "registro_expansion.json"

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
    plt.show()


def visualizar_distribucion_conexiones(G):
    """ Grafica la distribución de conexiones en la red """
    grados = [G.degree(nodo) for nodo in G.nodes()]

    plt.figure(figsize=(8, 5))
    plt.hist(grados, bins=range(1, max(grados) + 1), color="purple", alpha=0.7, edgecolor="black")
    plt.xlabel("Número de conexiones por nodo")
    plt.ylabel("Cantidad de nodos")
    plt.title("Distribución de Conexiones en la Red Fractal")
    plt.grid()
    plt.show()

# Cargar nodos en espera al inicio del programa
espera_nodos = cargar_espera_nodos()

# 🔹 Normalizar términos
def normalizar_termino(termino):
    termino = re.sub(r"[^a-zA-Z0-9áéíóúüñ ]", "_", termino)  # Mantiene espacios
    return termino.lower().strip().replace(" ", "_")


# 🔹 Nueva función para consultar ChatGPT
def consultar_chatgpt(tema, diccionario):
    tema = corregir_termino(tema, diccionario).lower()  # 🔹 Corregir y forzar a minúsculas

    try:
        print(f"🤖 Consultando ChatGPT sobre: {tema}...")

        respuesta = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
#                    "Eres un experto en matemáticas, geometría, estructuras fractales, biología, física y todas las ciencias. "
#                    "Tu objetivo es generar términos técnicos. "
                        "Proporciona solo una lista de términos separados por comas definiendo el concepto propuesto. TUS RESULTADOS TIENEN QUE SER ENTENDIBLES PARA UN PRIMARIA NIÑO DE PRIMARIA."
                        "Responde exclusivamente con una lista separada por comas, sin explicaciones, sin números (al menos para contarlos, pero sí si forma parte del concepto) sin frases ni puntos porfavor..."
                )},
                {"role": "user", "content": f"Dame solo términos técnicos relacionados con {tema}."}
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

def calcular_prioridad_nodo(G, nodo):
    """ Calcula la prioridad de un nodo basándose en su importancia en la red """
    conexiones = G.degree(nodo)
    centralidad = nx.degree_centrality(G).get(nodo, 0)  # Centralidad del nodo
    peso_promedio = sum([G.edges[nodo, neighbor].get('weight', 1) for neighbor in G.neighbors(nodo)]) / (conexiones or 1)

    # Prioridad basada en conexiones, peso de enlaces y centralidad
    prioridad = (conexiones * 0.4) + (centralidad * 0.4) + (peso_promedio * 0.2)
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

# 🔹 Agregar llamadas en puntos clave
def expansion_prioritaria(G, diccionario):
    """ Expande los nodos más prioritarios en la red """
    nodos_a_expandir = priorizar_expansion(G)[:10]  
    for nodo in nodos_a_expandir:
        print(f"🔍 Expansión prioritaria en: {nodo}")
        nuevos_conceptos = consultar_chatgpt(nodo, diccionario)
        if nuevos_conceptos:
            G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
            registrar_expansion(nodo, nuevos_conceptos, "GPT-4")  # 🔹 Se registra aquí
        expandir_concepto_embedding(nodo, G, diccionario)
        registrar_expansion(nodo, [], "Embeddings")  # 🔹 Se registra la expansión con embeddings
    guardar_red(G)

def cargar_red():
    diccionario = cargar_diccionario()
    try:
        with open("red_fractal.json", "r") as f:
            data = json.load(f)
            G = nx.node_link_graph(data, edges="links")

            # 🔹 Corregir nombres de nodos: convertir "_" en espacios
            G = nx.relabel_nodes(G, {nodo: nodo.replace("_", " ") for nodo in G.nodes()})
            
            print("✅ Red fractal cargada correctamente.")
            return G, diccionario
    except FileNotFoundError:
        print("🚀 Creando nueva red fractal con semilla inicial.")

        if not diccionario:
            diccionario = {concepto.replace("_", " "): [] for concepto in SEMILLA_INICIAL}
            guardar_diccionario(diccionario)

        G = nx.DiGraph()
        G.add_node(NODO_CENTRAL.replace("_", " "))  # Agregar nodo central corregido

        # 🔹 Conectar dualidades predefinidas con nombres corregidos
        for concepto, dual in dualidades_base.items():
            concepto = concepto.replace("_", " ")
            dual = dual.replace("_", " ")
            G.add_node(concepto)
            G.add_node(dual)
            G.add_edge(concepto, dual, weight=2.0)
            G.add_edge(dual, concepto, weight=2.0)
            G.add_edge(concepto, NODO_CENTRAL, weight=1.5)
        # 🔹 Conectar los conceptos de la semilla entre sí
        for i in range(len(SEMILLA_INICIAL)):
            for j in range(i + 1, len(SEMILLA_INICIAL)):
                concepto1 = SEMILLA_INICIAL[i].replace("_", " ")
                concepto2 = SEMILLA_INICIAL[j].replace("_", " ")
                G.add_edge(concepto1, concepto2, weight=1.2)
        guardar_red(G)
        return G, diccionario

# 🔹 Consultar Wikipedia y detectar términos opuestos
def consultar_wikipedia(concepto, G, diccionario):
    concepto = corregir_termino(concepto, diccionario)  # Corrección antes de la consulta
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
    # Extraer solo los primeros 10 enlaces relevantes
    enlaces_relacionados = [link.strip() for link in list(page.links)[:10] if es_nodo_relevante(link)]
    # Mostrar los enlaces encontrados
    if enlaces_relacionados:
        print("🔗 Wikipedia sugiere los siguientes conceptos:")
        for termino in enlaces_relacionados:
            print(f"- {termino}")
    for termino in enlaces_relacionados:
        if not es_nodo_relevante(termino):
            continue  # Filtra nodos irrelevantes
        # 🔹 Mantener la capitalización de los términos
        termino = termino.strip()
        if termino not in G.nodes():
            G.add_node(termino)
            diccionario.setdefault(concepto, []).append(termino)
            G.add_edge(concepto, termino, weight=1.2)
        # ✅ Se pasa el concepto original como 'concepto_base'
        dualidad_opuesta = detectar_dualidad(termino, G, concepto)
        if dualidad_opuesta and dualidad_opuesta in G.nodes():
            conectar_dualidad_con_equilibrio(termino, dualidad_opuesta, G)
    guardar_red(G)
    return f"📖 Wikipedia: {resumen}...\n🔗 Más info: {page.fullurl}"
    
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
            print(f"🆕 Añadiendo nodo superior dinámico: {nodo_superior}")

        # Conectar la dualidad al nodo superior
        G.add_edge(nodo_superior, concepto, weight=1.5)
        G.add_edge(nodo_superior, dualidad, weight=1.5)
        print(f"🔗 Vinculando {concepto} y {dualidad} a {nodo_superior}")

    # 🔹 Buscar el punto de equilibrio para esta dualidad
    nodo_equilibrio = detectar_nodo_equilibrio(concepto, dualidad)

    if nodo_equilibrio:
        if nodo_equilibrio not in G.nodes():
            G.add_node(nodo_equilibrio)
            print(f"🆕 Añadiendo nodo de equilibrio: {nodo_equilibrio}")

        # Conectar el equilibrio con ambos extremos
        G.add_edge(nodo_equilibrio, concepto, weight=1.8)
        G.add_edge(nodo_equilibrio, dualidad, weight=1.8)
        print(f"⚖️ Estableciendo equilibrio entre {concepto} y {dualidad} en {nodo_equilibrio}")

def detectar_nodo_equilibrio(concepto, dualidad):
    """ Determina un nodo de equilibrio basado en reglas predefinidas y embeddings. """
    nodos_equilibrio = {
        ("suma", "resta"): "promedio",
        ("multiplicación", "división"): "raíz cuadrada",
        ("positivo", "negativo"): "cero",
        ("orden", "caos"): "autoorganización",
        ("luz", "oscuridad"): "penumbra",
        ("bien", "mal"): "ética"
    }

    # Revisar si la dualidad está en la lista predefinida
    for (a, b), equilibrio in nodos_equilibrio.items():
        if (concepto == a and dualidad == b) or (concepto == b and dualidad == a):
            return equilibrio

    # Si no está en la lista, intentamos encontrar una relación semántica con embeddings
    return detectar_equilibrio_embeddings(concepto, dualidad)
    
def detectar_equilibrio_embeddings(concepto, dualidad):
    """ Usa embeddings para detectar un nodo de equilibrio no predefinido. """
    palabras_relacionadas = ["equilibrio", "media", "neutro", "balance", "punto medio"]
    
    embedding_concepto = modelo_embeddings.encode(concepto, convert_to_tensor=True)
    embedding_dualidad = modelo_embeddings.encode(dualidad, convert_to_tensor=True)
    embeddings_referencia = modelo_embeddings.encode(palabras_relacionadas, convert_to_tensor=True)

    # Calcular similitud con palabras de equilibrio
    similitudes_concepto = util.pytorch_cos_sim(embedding_concepto, embeddings_referencia)[0]
    similitudes_dualidad = util.pytorch_cos_sim(embedding_dualidad, embeddings_referencia)[0]

    # Promediar similitudes y seleccionar el término con mayor relación
    similitudes_totales = (similitudes_concepto + similitudes_dualidad) / 2
    indice_max = similitudes_totales.argmax().item()

    return palabras_relacionadas[indice_max]  # Devuelve el término de equilibrio más cercano

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


# 🔹 Guardar diccionario en un archivo JSON
def guardar_diccionario(diccionario):
    with open("diccionario.json", "w", encoding="utf-8") as f:
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
        conceptos_filtrados = [
            c for c in conceptos if max(
                util.pytorch_cos_sim(modelo.encode(c, convert_to_tensor=True), modelo.encode(nodo, convert_to_tensor=True))[0].tolist()
            ) > UMBRAL_SIMILITUD
        ]

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
    embedding_concepto = modelo.encode(nuevo_concepto, convert_to_tensor=True)
    embeddings_red = modelo.encode(palabras_red, convert_to_tensor=True)

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
#    print(f"✅ PAUSA 20s")
#    time.sleep(20)
    return G

def detectar_dualidad(concepto, G, concepto_base=None):
    """ Detecta si existe una dualidad semántica en la red. """
    # 🔹 Buscar dualidades predefinidas primero
    dualidades_predefinidas = {
        "luz": "oscuridad", "positivo": "negativo", "vida": "muerte", "izquierda": "derecha",
        "arriba": "abajo", "orden": "caos"
    }

    if concepto in dualidades_predefinidas:
        dualidad = dualidades_predefinidas[concepto]
        if not G.has_edge(concepto, dualidad):  # Solo conectar si no está ya conectado
            print(f"🔄 Conectando dualidad predefinida: {concepto} ↔ {dualidad}")
            conectar_dualidad_con_equilibrio(concepto, dualidad, G)
        return dualidad
    
    # 🔹 Buscar en dualidades de WordNet
    dualidades_wordnet = detectar_dualidad_wordnet(concepto)
    if dualidades_wordnet:
        for dualidad in dualidades_wordnet:
            if dualidad in G.nodes() and not G.has_edge(concepto, dualidad):  # Verificar si no existe conexión
                print(f"🌿 Dualidad detectada vía WordNet: {concepto} ↔ {dualidad}")
                conectar_dualidad_con_equilibrio(concepto, dualidad, G)
                return dualidad  # Retorna la primera dualidad válida encontrada
    
    # 🔹 Detectar dualidad usando embeddings
    max_similitud = 0
    mejor_dualidad = None
    emb_concepto = modelo_embeddings.encode(concepto, convert_to_tensor=True)
    for nodo in G.nodes():
        if nodo == concepto:
            continue
        emb_nodo = modelo_embeddings.encode(nodo, convert_to_tensor=True)
        similitud = util.pytorch_cos_sim(emb_concepto, emb_nodo).item()
        if similitud > max_similitud and similitud >= 0.85:
            max_similitud = similitud
            mejor_dualidad = nodo

    if mejor_dualidad:
        print(f"🔄 Detectada posible dualidad con embeddings: {concepto} ↔ {mejor_dualidad} (Similitud: {max_similitud:.2f})")
        if not G.has_edge(concepto, mejor_dualidad):  # Solo conectar si no existe la conexión
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
    """ Busca un nodo superior dinámico basado en embeddings. """
    palabras_red = list(G.nodes())
    embedding_concepto = modelo_embeddings.encode(concepto, convert_to_tensor=True)
    embedding_dualidad = modelo_embeddings.encode(dualidad, convert_to_tensor=True)
    embeddings_red = modelo_embeddings.encode(palabras_red, convert_to_tensor=True)

    similitudes_concepto = util.pytorch_cos_sim(embedding_concepto, embeddings_red)[0]
    similitudes_dualidad = util.pytorch_cos_sim(embedding_dualidad, embeddings_red)[0]

    # Promediamos las similitudes con ambos conceptos
    similitudes_totales = (similitudes_concepto + similitudes_dualidad) / 2
    indices_top = similitudes_totales.argsort(descending=True)[:top_n].tolist()

    posibles_superiores = [palabras_red[i] for i in indices_top if palabras_red[i] not in [concepto, dualidad]]

    return posibles_superiores
        

    
# 🔹 Ajustar pesos en conexiones de la red fractal
def ajustar_pesos_conexiones(G):
    """
    Modifica los pesos de las conexiones en función de su relación con dualidades
    y combinaciones de términos en la red. Se asegura de que los pesos no crezcan descontroladamente.
    """
    max_peso = 5.0  # 🔹 Límite superior para evitar líneas excesivamente gruesas
    min_peso = 0.5  # 🔹 Límite inferior para que las conexiones no desaparezcan

    for nodo1, nodo2 in G.edges():
        peso_actual = G.edges[nodo1, nodo2].get('weight', 1.0)

        # 🔹 Mantener pesos dentro de un rango controlado
        if nodo1 in dualidades_base and dualidades_base[nodo1] == nodo2:
            peso_nuevo = max(2.0, min(peso_actual * 1.1, max_peso))  # Incrementa pero no sobrepasa 5.0
            G.edges[nodo1, nodo2]['weight'] = peso_nuevo
            G.edges[nodo2, nodo1]['weight'] = peso_nuevo

        # 🔹 Reducir peso si es demasiado débil
        elif peso_actual < 1.2:
            peso_nuevo = max(min_peso, peso_actual * 0.8)  # Nunca por debajo de 0.5
            G.edges[nodo1, nodo2]['weight'] = peso_nuevo

        # 🔹 Aumentar peso si la conexión aparece muchas veces
        elif G.degree(nodo1) > 3 and G.degree(nodo2) > 3:
            peso_nuevo = min(peso_actual * 1.05, max_peso)  # Incremento menor para evitar sobrecarga
            G.edges[nodo1, nodo2]['weight'] = peso_nuevo

    print("✅ Pesos de conexiones ajustados y normalizados en la red.")
    return G

# 🔹 Reorganizar la red eliminando nodos sueltos y corrigiendo conexiones incorrectas
def reorganizar_red(G, max_espera=5):
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
        if G.degree(nodo) == 0 and nodo != NODO_CENTRAL:
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

    print("✅ Red reorganizada: nodos sueltos en espera, conexiones corregidas.")
    return G
    
def guardar_estado_parcial(G, espera_nodos):
    # Guarda la red y los nodos pendientes en un archivo temporal
    with open('estado_parcial.json', 'w') as f:
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
        raiz = obtener_raiz(concepto)

        if any(raiz in nodo for nodo in nodos_existentes):
            print(f"⚠️ Se ha detectado redundancia con '{concepto}'. No se añadirá a la red.")
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
    nodos_a_expandir = [nodo for nodo in G.nodes() if G.degree(nodo) < 2 and nodo.lower() != NODO_CENTRAL.lower()]

    for nodo in nodos_a_expandir:
        nodo = nodo.lower()  # 🔹 Forzar a minúsculas

        print(f"🚀 Expansión automática para: {nodo}")
        consultar_wikipedia(nodo, G, diccionario)
        nuevos_conceptos = consultar_chatgpt(nodo, diccionario)

        if nuevos_conceptos:
            G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
        
        expandir_concepto_embedding(nodo, G, diccionario)

    guardar_red(G)

# 🔹 Expandir la red fractal con embeddings
def expandir_concepto_embedding(concepto, G, diccionario, top_n=5):
    palabras_red = list(G.nodes())
    if concepto not in palabras_red:
        G.add_node(concepto)

    embedding_concepto = modelo.encode(concepto, convert_to_tensor=True)
    embeddings_red = modelo.encode(palabras_red, convert_to_tensor=True)

    similitudes = util.pytorch_cos_sim(embedding_concepto, embeddings_red)[0]
    indices_top = similitudes.argsort(descending=True)[:top_n].tolist()
    relacionados = [palabras_red[i] for i in indices_top if palabras_red[i] != concepto]

    for i, termino in enumerate(relacionados):
        # peso = np.tanh(similitudes[indices_top[i]].item())  # Usa tanh para normalizar
        peso = np.exp(similitudes[indices_top[i]].item())

        G.add_edge(concepto, termino, weight=peso)

    diccionario[concepto] = relacionados
    with open("diccionario.json", "w", encoding="utf-8") as f:
        json.dump(diccionario, f, ensure_ascii=False, indent=4)

    # Ajustar pesos de conexiones después de la expansión
    G = ajustar_pesos_conexiones(G)
    G = reorganizar_red(G)
    G = detectar_nuevas_dualidades(G)

    guardar_red(G)
    return relacionados

# 🔹 Visualizar la red fractal con mejor distribución
def visualizar_red(G):
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    print("📊 Cargando nodos para visualización...")
    
    # 🔹 Usar layout de resorte (spring_layout) para una distribución más natural
    posiciones = nx.spring_layout(G, k=0.5, iterations=50)  # k controla la dispersión

    for nodo, coords in posiciones.items():
        x, y = coords[0] * 2000, coords[1] * 2000  # Ajustar escala de distribución
        color = (
            "green" if nodo == NODO_CENTRAL else
            "orange" if nodo in dualidades_base else
            "blue"
        )
        net.add_node(nodo, label=nodo, color=color, size=20, x=x, y=y, physics=True)

    # 🔹 Configurar físicas en PyVis para evitar colapso de nodos
#    net.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=250, spring_strength=0.1)
    #net.force_atlas_2based(gravity=-100, central_gravity=0.005, spring_length=500, spring_strength=0.1)
    net.set_options("""
        var options = {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04
                }
            }
        }
    """)

    for edge in G.edges():
        net.add_edge(edge[0], edge[1], color="gray", width=G.edges[edge].get('weight', 1.5))

    print(f"✅ Nodos en la visualización: {len(G.nodes())}")
    print(f"✅ Conexiones en la visualización: {len(G.edges())}")


    net.write_html("hipercubo_fractal.html")
    print("✅ Hipercubo fractal guardado como 'hipercubo_fractal.html'.")

# 🔹 Modificar la función expansión automática para incluir expansión prioritaria
def expansion_automatica(G, diccionario, expansion_activa, intervalo=5):
    """ Expande la red automáticamente cada X segundos si está activa """
    while expansion_activa:
        print("🔄 Iniciando expansión automática con prioridad...")
        expansion_prioritaria(G, diccionario)  # Integrando la expansión prioritaria
        
        nodos_a_expandir = [nodo for nodo in G.nodes() if G.degree(nodo) < 3 and nodo.lower() != NODO_CENTRAL.lower()]

        for nodo in nodos_a_expandir:
            if not expansion_activa:
                print("⏹️ Expansión automática detenida.")
                return
                
            nodo = nodo.lower()

            print(f"🔍 Expandiendo nodo aislado: {nodo}")
            resultado_wikipedia = consultar_wikipedia(nodo, G, diccionario)
            nuevos_conceptos = consultar_chatgpt(nodo, diccionario)

            if nuevos_conceptos:
                G = agregar_nuevo_nodo(G, diccionario, nuevos_conceptos)
                registrar_expansion(nodo, nuevos_conceptos, "GPT-4")
                guardar_diccionario(diccionario)

            expandir_concepto_embedding(nodo, G, diccionario)
            registrar_expansion(nodo, [], "Embeddings")
            guardar_diccionario(diccionario)

        guardar_red(G)
        reorganizar_red(G)
        evaluar_expansion(G)
        guardar_historial(cargar_historial())

        print("✅ Expansión automática completada. Esperando nuevo ciclo...")
        visualizar_red(G)
        print("⏳ Esperando antes del próximo ciclo...")
        generar_reportes()
        time.sleep(intervalo)

if __name__ == "__main__":
    # Cargar el grafo desde "grafo_consciencia.json"
    G_archivo = cargar_grafo()

    # Cargar la red desde "red_fractal.json"
    G_red, diccionario = cargar_red()

    # Fusionar ambos grafos (si es necesario)
    G = nx.compose(G_archivo, G_red)  # Une los dos grafos sin perder nodos ni conexiones


    print("🔄 Reorganizando la red para conectar nodos en espera...")
    G = reorganizar_red(G)

    print(f"📊 Nodos cargados: {len(G.nodes())}")
    print(f"🔗 Conexiones cargadas: {len(G.edges())}")
    print("🧐 Nodos sueltos en espera:", [nodo for nodo in G.nodes() if G.degree(nodo) == 0])

    if NODO_CENTRAL and NODO_CENTRAL in G:
        # Contar cuántos nodos están directamente conectados al nodo central
        conexiones_centrales = Counter(G.neighbors(NODO_CENTRAL))
        umbral_conexiones = 2  # Ajusta este número según lo que desees mostrar

        print(f"🔍 Nodos con muchas conexiones directas a '{NODO_CENTRAL}':")
        for nodo in conexiones_centrales:
            if G.degree(nodo) > umbral_conexiones:
                print(f"{nodo}: {G.degree(nodo)} conexiones")
        print(f"🔗 Conexiones del nodo central '{NODO_CENTRAL}':")
        for vecino in G.neighbors(NODO_CENTRAL):
            print(f"{vecino}: {G.degree(vecino)} conexiones")


    try:
        while True:
            entrada = input("\n🤔 Opciones: [salir] [ver red] [añadir] [expandir] [historial] [auto]: ").strip().lower()

            if entrada == "salir":
                print("👋 Saliendo... Guardando cambios en la red.")
                guardar_red(G)
                break  # 🔹 Evita usar sys.exit(0), permitiendo un cierre más natural

            elif entrada == "ver red":
                visualizar_red(G)
                print("🌟 Conceptos en la red:", {len(G.nodes())})
#                for nodo in G.nodes():
#                    print(f"- {nodo}")  # 🔹 Se restauró la lista de conceptos en la red

                # Guardar gráfico en archivo en lugar de mostrarlo en pantalla
                plt.figure(figsize=(10, 5))
                plt.hist([G.degree(nodo) for nodo in G.nodes()], bins=10, color="blue", alpha=0.7)
                plt.xlabel("Número de conexiones por nodo")
                plt.ylabel("Cantidad de nodos")
                plt.title("Distribución de Conexiones en la Red Fractal")
                plt.grid()
                plt.savefig("grafico_red.png")
                print("🎨 Gráfico guardado como 'grafico_red.png'. Ábrelo manualmente.")

            elif entrada == "añadir":
                conceptos = input("➞ Introduce conceptos separados por espacio. Si el concepto tiene espacios usa \"_\" para separar palabras: ").strip().split()
                G = agregar_nuevo_nodo(G, diccionario, conceptos)
                guardar_red(G)

            elif entrada == "expandir":
                concepto = input("➞ Concepto a expandir: ").strip()
                print("📚 Expandiendo desde Wikipedia y GPT...")
                resultado_wikipedia = consultar_wikipedia(concepto, G, diccionario)
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
                
            elif entrada == "auto":
                print("⚡ Iniciando expansión automática y dinámica... (Presiona Ctrl+C para detener)")
                expansion_activa = True

                try:
                    while expansion_activa:
                        expansion_automatica(G, diccionario, expansion_activa)
                        expansion_dinamica(G, diccionario)  # 🔹 Se añade la expansión dinámica aquí                     

                        for _ in range(10):  # Espera en intervalos más cortos para permitir interrupción rápida
                            if not expansion_activa:
                                break
                            time.sleep(1)  # Espera en intervalos de 1 segundo para mejorar respuesta a Ctrl+C
                except KeyboardInterrupt:
                    print("\n⏹ Expansión detenida por el usuario.")
                    expansion_activa = False
                    guardar_red(G)

            else:
                print(f"⚠️ Entrada errónea '{entrada}'.")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupción detectada. Guardando y cerrando de manera segura...")
        expansion_activa = False
        guardar_red(G)


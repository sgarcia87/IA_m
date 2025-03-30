
import math

def generar_cuadrado_conceptual(a, b, c, d):
    cuadrado = {
        "superior_izquierdo": a,
        "superior_derecho": b,
        "inferior_izquierdo": c,
        "inferior_derecho": d
    }
    centro = calcular_centro([a, b, c, d])
    emergencia = buscar_emergente(cuadrado, centro)
    return {
        "estructura": cuadrado,
        "centro": centro,
        "emergente": emergencia
    }

def calcular_centro(valores):
    try:
        # Si los valores son numéricos, calcula promedio matemático
        valores_numericos = [float(v) for v in valores]
        return sum(valores_numericos) / len(valores_numericos)
    except:
        # Si son strings, devuelve el término común o "promedio"
        return "promedio"

def buscar_emergente(cuadrado, centro):
    # Simplificación inicial: si las operaciones son básicas, sugiere raíz cuadrada
    simbolos = set(cuadrado.values())
    if simbolos == {"-", "+", "/", "*"}:
        return "raíz cuadrada"
    elif simbolos == {"sin", "cos", "tan", "cot"}:
        return "identidad trigonométrica"
    elif simbolos == {"∪", "∩", "⊆", "⊇"}:
        return "lógica de conjuntos"
    else:
        return "emergencia desconocida"

if __name__ == "__main__":
    resultado = generar_cuadrado_conceptual("-", "+", "/", "*")
    for k, v in resultado.items():
        print(f"{k}: {v}")

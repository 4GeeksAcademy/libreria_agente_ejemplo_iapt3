"""Agente conversacional que gestiona la libreria llamando a la API FastAPI local."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MAX_ITERATIONS = 5

LOG_PATH = Path(__file__).resolve().parent.parent / "conversation_log.csv"
LOG_COLUMNS = ["timestamp", "role", "content"]

SYSTEM_PROMPT = (
    "Eres el asistente de una libreria. Dispones de herramientas para consultar y "
    "modificar el inventario a traves de una API. Usa las herramientas siempre que "
    "necesites datos reales del inventario y responde en espaniol de forma breve."
)


# --------------------------------------------------------------------------- #
# Clientes HTTP contra la API FastAPI
# --------------------------------------------------------------------------- #

def _request(method: str, path: str, **kwargs) -> dict | list:
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", timeout=10.0, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return {"error": exc.response.text, "status_code": exc.response.status_code}
    except httpx.HTTPError as exc:
        return {"error": f"No se pudo contactar la API: {exc}"}


def get_books() -> dict | list:
    return _request("GET", "/books")


def add_books(title: str, author: str, quantity: int = 0) -> dict | list:
    return _request(
        "POST", "/books", json={"title": title, "author": author, "quantity": quantity}
    )


def update_books(book_id: int, delta: int) -> dict | list:
    return _request("PATCH", f"/books/{book_id}", json={"delta": delta})


def get_stock_alert(threshold: int = 5) -> dict | list:
    return _request("GET", "/books/alerts", params={"threshold": threshold})


AVAILABLE_FUNCTIONS = {
    "get_books": get_books,
    "add_books": add_books,
    "update_books": update_books,
    "get_stock_alert": get_stock_alert,
}


# --------------------------------------------------------------------------- #
# Definicion de tools (JSON Schema compatible con Groq)
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_books",
            "description": "Devuelve el catalogo completo de libros con su id, titulo, autor y cantidad en stock.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_books",
            "description": "Agrega un libro nuevo al inventario. El id se genera automaticamente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Titulo del libro."},
                    "author": {"type": "string", "description": "Autor del libro."},
                    "quantity": {
                        "type": "integer",
                        "description": "Cantidad inicial en stock. Por defecto 0.",
                        "minimum": 0,
                    },
                },
                "required": ["title", "author"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_books",
            "description": (
                "Ajusta el stock de un libro existente. Usa delta positivo para reposiciones "
                "y delta negativo para ventas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "integer", "description": "Id del libro a actualizar."},
                    "delta": {
                        "type": "integer",
                        "description": "Unidades a sumar (positivo) o restar (negativo).",
                    },
                },
                "required": ["book_id", "delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_alert",
            "description": "Lista los libros cuyo stock es menor o igual al umbral indicado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "integer",
                        "description": "Umbral de stock bajo. Por defecto 5.",
                        "minimum": 0,
                    }
                },
                "required": [],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Registro de la conversacion
# --------------------------------------------------------------------------- #

def log_event(role: str, content: str) -> None:
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), role, content])


# --------------------------------------------------------------------------- #
# Bucle del agente: observar -> pensar -> actuar -> actualizar
# --------------------------------------------------------------------------- #

def run_tool(name: str, raw_arguments: str) -> str:
    function = AVAILABLE_FUNCTIONS.get(name)
    if function is None:
        return json.dumps({"error": f"Herramienta desconocida: {name}"})
    try:
        arguments = json.loads(raw_arguments or "{}")
        result = function(**arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        result = {"error": f"Argumentos invalidos para {name}: {exc}"}
    return json.dumps(result, ensure_ascii=False)


def think(client: Groq, messages: list[dict]):
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.6,
        max_tokens=500,
        top_p=0.95,
    )
    return completion.choices[0].message


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Falta GROQ_API_KEY. Copia .env.example a .env y completala.")

    client = Groq()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Agente de libreria listo. Escribe 'salir' para terminar.\n")
    while True:
        # 1. Observar
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        log_event("user", user_input)

        for _ in range(MAX_ITERATIONS):
            # 2. Pensar
            message = think(client, messages)
            tool_calls = message.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ]
                    or None,
                }
            )

            # 5. Repetir hasta obtener una respuesta final sin tools
            if not tool_calls:
                answer = message.content or "(sin respuesta)"
                print(f"Agente: {answer}\n")
                log_event("assistant", answer)
                break

            for call in tool_calls:
                # 3. Actuar
                log_event("tool_call", f"{call.function.name}({call.function.arguments})")
                result = run_tool(call.function.name, call.function.arguments)
                log_event("tool_result", result)
                # 4. Actualizar el historial con el resultado
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": result,
                    }
                )
        else:
            aviso = "Se alcanzo el limite de iteraciones sin una respuesta final."
            print(f"Agente: {aviso}\n")
            log_event("assistant", aviso)

    print("Hasta luego.")


if __name__ == "__main__":
    main()

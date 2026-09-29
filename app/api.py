from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.database import ensure_books_csv, load_books, save_books

load_dotenv()

app = FastAPI(title="Libreria Agente MVP")


class BookIn(BaseModel):
    title: str = Field(min_length=1)
    author: str = Field(min_length=1)
    quantity: int = Field(default=0, ge=0)


class Book(BookIn):
    id: int


class StockUpdate(BaseModel):
    delta: int = Field(description="Positivo = reposicion, negativo = venta")


@app.on_event("startup")
def startup() -> None:
    ensure_books_csv()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/books", response_model=list[Book])
def list_books() -> list[dict]:
    return load_books().to_dict(orient="records")


# Debe ir antes de /books/{book_id} para que la ruta dinamica no capture "alerts".
@app.get("/books/alerts", response_model=list[Book])
def low_stock_alerts(threshold: int = Query(default=5, ge=0)) -> list[dict]:
    df = load_books()
    return df[df["quantity"] <= threshold].to_dict(orient="records")


@app.get("/books/{book_id}", response_model=Book)
def get_book(book_id: int) -> dict:
    df = load_books()
    match = df[df["id"] == book_id]
    if match.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Libro no encontrado")
    return match.iloc[0].to_dict()


@app.post("/books", response_model=Book, status_code=status.HTTP_201_CREATED)
def create_book(book: BookIn) -> dict:
    df = load_books()
    new_id = 1 if df.empty else int(df["id"].max()) + 1
    record = {"id": new_id, **book.model_dump()}
    df.loc[len(df)] = record
    save_books(df)
    return record


@app.patch("/books/{book_id}", response_model=Book)
def update_stock(book_id: int, update: StockUpdate) -> dict:
    df = load_books()
    matches = df.index[df["id"] == book_id]
    if matches.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Libro no encontrado")

    idx = matches[0]
    new_quantity = int(df.at[idx, "quantity"]) + update.delta
    if new_quantity < 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stock insuficiente para aplicar el delta",
        )

    df.at[idx, "quantity"] = new_quantity
    save_books(df)
    return df.loc[idx].to_dict()

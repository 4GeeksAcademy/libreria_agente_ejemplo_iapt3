from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOOKS_CSV = DATA_DIR / "books.csv"

COLUMNS = ["id", "title", "author", "quantity"]

SAMPLE_BOOKS = [
    {"id": 1, "title": "Cien anios de soledad", "author": "Gabriel Garcia Marquez", "quantity": 3},
    {"id": 2, "title": "La casa de los espiritus", "author": "Isabel Allende", "quantity": 5},
]


def ensure_books_csv() -> Path:
    """Create data/books.csv with sample rows if it does not exist yet."""
    if not BOOKS_CSV.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(SAMPLE_BOOKS, columns=COLUMNS).to_csv(BOOKS_CSV, index=False)
    return BOOKS_CSV


def load_books() -> pd.DataFrame:
    ensure_books_csv()
    return pd.read_csv(BOOKS_CSV)


def save_books(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BOOKS_CSV, index=False)


if __name__ == "__main__":
    print(f"books.csv listo en: {ensure_books_csv()}")

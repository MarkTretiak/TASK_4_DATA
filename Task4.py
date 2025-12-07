import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import pandas as pd
import yaml

EUR_TO_USD = 1.2

def clean_numeric_col(series: pd.Series) -> pd.Series:
    """Make messy numeric column usable: strip junk, handle commas, etc."""
    s = series.astype(str).str.replace(r"[^0-9\.,\-]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = clean_numeric_col(df[c])
    return df

def normalize_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in orders.parquet")

    df = df.copy()

    ts = pd.to_datetime(df[col], errors="coerce", dayfirst=True, utc=True)

    ts = ts.dt.tz_convert(None)

    today = pd.Timestamp("today").normalize()

    mask_valid = ts.notna() & (ts.dt.normalize() < today)

    dropped = (~mask_valid).sum()
    if dropped:
        print(
            f"Dropping {dropped} orders with invalid / today / future timestamps "
            f"(min={ts[mask_valid].min()}, max={ts[mask_valid].max()})"
        )

    df = df.loc[mask_valid].copy()
    df[col] = ts[mask_valid]

    return df


def normalize_authors_field(value):
    if isinstance(value, list):
        parts = [str(a).strip() for a in value if str(a).strip()]
    elif isinstance(value, str):
        raw = value.replace("&", ",").replace(";", ",").split(",")
        parts = [p.strip() for p in raw if p.strip()]
    else:
        parts = []

    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    out.sort(key=str.lower)
    return out

def author_set_key(authors: List[str]) -> str:
    return "|".join(a.lower().strip() for a in authors)

def find_file(base: Path, main_name: str, dataset_suffix: str, extra_names=None) -> Path:
    candidates = [main_name]
    if dataset_suffix:
        stem, ext = main_name.split(".", 1)
        candidates.append(f"{stem}{dataset_suffix}.{ext}")
    if extra_names:
        candidates.extend(extra_names)

    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No file found for any of: {candidates} in {base}")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def normalize_user_columns(users: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    cols = set(users.columns)

    if "user_id" not in cols:
        for cand in ["user_id", "id", "userId"]:
            if cand in cols:
                col_map[cand] = "user_id"
                break

    if "name" not in cols:
        for cand in ["name", "full_name", "username", "user_name"]:
            if cand in cols:
                col_map[cand] = "name"
                break

    if "email" not in cols:
        for cand in ["email", "email_address", "mail", "e_mail", "Email"]:
            if cand in cols:
                col_map[cand] = "email"
                break

    if "phone" not in cols:
        for cand in ["phone", "phone_number", "tel", "telephone", "Phone"]:
            if cand in cols:
                col_map[cand] = "phone"
                break

    if "address" not in cols:
        for cand in ["address", "addr", "location", "address1", "Address"]:
            if cand in cols:
                col_map[cand] = "address"
                break

    if col_map:
        print("Renaming user columns:", col_map)
        users = users.rename(columns=col_map)

    return users


def reconcile_users(users: pd.DataFrame) -> Dict[int, Set[int]]:
    users = normalize_user_columns(users)

    needed = ["user_id", "name", "email", "phone", "address"]
    for c in needed:
        if c not in users.columns:
            raise ValueError(f"Expected column '{c}' in users.csv")

    users = users.reset_index(drop=True)
    n = len(users)
    uf = UnionFind(n)

    combos = [
        ("name", "email", "phone"),
        ("name", "email", "address"),
        ("name", "phone", "address"),
        ("email", "phone", "address"),
    ]

    for combo in combos:
        grouped = (
            users.dropna(subset=list(combo))
                 .groupby(list(combo))
                 .indices
        )
        for _, idxs in grouped.items():
            if len(idxs) > 1:
                base = idxs[0]
                for other in idxs[1:]:
                    uf.union(base, other)

    clusters: Dict[int, Set[int]] = {}
    for i, row in users.iterrows():
        r = uf.find(i)
        clusters.setdefault(r, set()).add(row["user_id"])
    return clusters


def load_books_yaml(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        rows = []
        for bid, info in data.items():
            row = dict(info)
            row["id"] = bid
            rows.append(row)
        return pd.DataFrame(rows)
    elif isinstance(data, list):
        return pd.json_normalize(data)
    else:
        raise ValueError("Unsupported books.yaml structure")

def normalize_book_columns(books: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    cols = set(books.columns)

    if "book_id" not in cols:
        for cand in ["book_id", "id", "bookId", "book", ":id"]:
            if cand in cols:
                col_map[cand] = "book_id"
                break

    if "authors" not in cols:
        for cand in ["authors", "author", "author_list", ":author"]:
            if cand in cols:
                col_map[cand] = "authors"
                break

    if col_map:
        print("Renaming book columns:", col_map)
        books = books.rename(columns=col_map)
    return books

@dataclass
class DatasetSummary:
    dataset_name: str
    top_5_days: List[dict]
    unique_users: int
    unique_author_sets: int
    most_popular_authors: List[str]
    best_buyer_ids: List[int]


def process_dataset(base_dir: Path, dataset_name: str) -> DatasetSummary:
    suffix = "".join(ch for ch in dataset_name if ch.isdigit())
    orders_path = find_file(base_dir, "orders.parquet", suffix)
    users_path = find_file(base_dir, "users.csv", suffix)
    books_path = find_file(base_dir, "books.yaml", suffix)

    print(f"{dataset_name}: using files:")
    print("  orders:", orders_path.name)
    print("  users :", users_path.name)
    print("  books :", books_path.name)

    orders = pd.read_parquet(orders_path, engine="pyarrow")
    users = pd.read_csv(users_path)
    books = load_books_yaml(books_path)
    orders = orders.drop_duplicates()
    users = users.drop_duplicates()
    books = books.drop_duplicates()

    print(f"{dataset_name}: orders rows before clean:", len(orders))
    orders = ensure_numeric(orders, ["quantity", "unit_price"])
    orders = normalize_timestamp(orders, "timestamp")
    print(f"{dataset_name}: orders rows after clean:", len(orders))

    base_paid = orders["quantity"] * orders["unit_price"]
    if "currency" in orders.columns:
        is_eur = orders["currency"].astype(str).str.upper().eq("EUR")
        orders["paid_price"] = base_paid.where(~is_eur, base_paid * EUR_TO_USD)
    else:
        orders["paid_price"] = base_paid

    orders["date"] = orders["timestamp"].dt.date
    orders["year"] = orders["timestamp"].dt.year
    orders["month"] = orders["timestamp"].dt.month
    orders["day"] = orders["timestamp"].dt.day

    daily_rev = (
        orders.dropna(subset=["date"])
              .groupby("date", as_index=False)["paid_price"]
              .sum()
              .rename(columns={"paid_price": "revenue"})
              .sort_values("date")
    )
    print(f"{dataset_name}: daily_rev rows:", len(daily_rev))

    top_5 = (
        daily_rev.sort_values("revenue", ascending=False)
                 .head(5)
                 .copy()
    )
    top_5["date_str"] = pd.to_datetime(top_5["date"]).dt.strftime("%Y-%m-%d")
    top_5_days = [
        {"date": row["date_str"], "revenue": float(row["revenue"])}
        for _, row in top_5.iterrows()
    ]

    user_clusters = reconcile_users(users)
    unique_users = len(user_clusters)
    user_id_to_cluster: Dict[int, int] = {}
    for root, ids in user_clusters.items():
        for uid in ids:
            user_id_to_cluster[uid] = root

    books = normalize_book_columns(books)
    if "book_id" not in books.columns:
        raise ValueError("books.yaml must contain 'book_id'")
    if "authors" not in books.columns:
        raise ValueError("books.yaml must contain 'authors'")

    books["authors_list"] = books["authors"].apply(normalize_authors_field)
    books["author_set_key"] = books["authors_list"].apply(author_set_key)
    books["author_set_key"] = books["author_set_key"].replace("", pd.NA)
    unique_author_sets = int(books["author_set_key"].dropna().nunique())

    if "book_id" not in orders.columns:
        for cand in ["book_id", "bookId", "book"]:
            if cand in orders.columns:
                orders = orders.rename(columns={cand: "book_id"})
                break
        if "book_id" not in orders.columns:
            raise ValueError("orders.parquet must contain 'book_id'")

    if "user_id" not in orders.columns:
        for cand in ["user_id", "userId", "uid"]:
            if cand in orders.columns:
                orders = orders.rename(columns={cand: "user_id"})
                break
        if "user_id" not in orders.columns:
            raise ValueError("orders.parquet must contain 'user_id'")

    if "user_id" not in users.columns:
        users = normalize_user_columns(users)
        if "user_id" not in users.columns:
            raise ValueError("users.csv must contain 'user_id'")

    ob = orders.merge(
        books[["book_id", "author_set_key", "authors_list"]],
        on="book_id",
        how="left",
    )

    qty_by_set = (
        ob.dropna(subset=["author_set_key"])
          .groupby("author_set_key", as_index=False)["quantity"]
          .sum()
          .rename(columns={"quantity": "sold_qty"})
    )

    if qty_by_set.empty:
        most_popular_authors: List[str] = []
    else:
        max_qty = qty_by_set["sold_qty"].max()
        top_sets = qty_by_set[qty_by_set["sold_qty"] == max_qty]
        best_key = top_sets.iloc[0]["author_set_key"]
        sample_row = books[books["author_set_key"] == best_key].iloc[0]
        most_popular_authors = sample_row["authors_list"]
    ou = orders.merge(users[["user_id"]], on="user_id", how="left")
    
    def cluster_for(uid):
        return user_id_to_cluster.get(uid, uid)
    ou["cluster_id"] = ou["user_id"].apply(cluster_for)

    spend_by_cluster = (
        ou.groupby("cluster_id", as_index=False)["paid_price"]
          .sum()
          .rename(columns={"paid_price": "total_spent"})
    )

    if spend_by_cluster.empty:
        best_buyer_ids: List[int] = []
    else:
        max_spent = spend_by_cluster["total_spent"].max()
        best_clusters = spend_by_cluster[
            spend_by_cluster["total_spent"] == max_spent
        ]["cluster_id"].tolist()
        best_buyer_ids_set: Set[int] = set()
        for root, ids in user_clusters.items():
            if root in best_clusters:
                best_buyer_ids_set.update(ids)
        best_buyer_ids = sorted(best_buyer_ids_set)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if not daily_rev.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(daily_rev["date"], daily_rev["revenue"])
        plt.xlabel("Date")
        plt.ylabel("Revenue (USD)")
        plt.title(f"Daily Revenue - {dataset_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_daily_revenue.png", dpi=120)
        plt.close()

    summary = DatasetSummary(
        dataset_name=dataset_name,
        top_5_days=top_5_days,
        unique_users=unique_users,
        unique_author_sets=unique_author_sets,
        most_popular_authors=most_popular_authors,
        best_buyer_ids=best_buyer_ids,
    )

    with open(output_dir / f"{dataset_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary


def main():
    base = Path(".")
    datasets = {
        "DATA1": base / "DATA1",
        "DATA2": base / "DATA2",
        "DATA3": base / "DATA3",
    }

    all_summaries: List[DatasetSummary] = []

    for name, path in datasets.items():
        if not path.exists():
            print(f"Skipping {name}: folder {path} not found")
            continue

        print(f"\nProcessing {name} from {path} ...")
        summary = process_dataset(path, name)
        all_summaries.append(summary)
        print(f"\n{name} summary:")
        print(json.dumps(asdict(summary), indent=2))

    output_dir = Path("output")
    combined = [asdict(s) for s in all_summaries]
    with open(output_dir / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

if __name__ == "__main__":
    main()
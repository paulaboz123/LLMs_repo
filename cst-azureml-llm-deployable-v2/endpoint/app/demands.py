from dataclasses import dataclass
import os
import pandas as pd

@dataclass(frozen=True)
class Demand:
    label: str
    item: str
    demand: str
    description: str

def _norm(s: str) -> str:
    return str(s).strip().lower()

def load_demands(path: str, sheet: str | None = None) -> list[Demand]:
    """Load demand catalog from CSV or XLSX.

    Required columns (case-insensitive):
      - item
      - demand (or 'global demand name')
      - demand description (or 'global demand description' / 'description' / 'clarification')
    """
    if not path:
        raise ValueError("DEMANDS_PATH is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Demand file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)

    colmap = {c.strip().lower(): c for c in df.columns}

    def get(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    item_col = get("item")
    dem_col = get("demand", "global demand name", "global_demand_name", "label")
    desc_col = get("demand description", "global demand description", "global_demand_description", "description", "clarification")

    if not item_col or not dem_col or not desc_col:
        raise ValueError("Demand file must contain: item, demand, demand description (case-insensitive).")

    out: list[Demand] = []
    for _, r in df.iterrows():
        item = _norm(r[item_col])
        dem = _norm(r[dem_col])
        desc = str(r[desc_col]).strip()
        if not item or item == "nan" or not dem or dem == "nan":
            continue
        label = f"{item}|{dem}"
        out.append(Demand(label=label, item=item, demand=dem, description=desc))

    if not out:
        raise ValueError("No demands loaded. Check your file content/columns.")
    return out

def demands_block(demands: list[Demand]) -> str:
    return "\n".join([f"- {d.label} | {d.item} | {d.demand} | {d.description}" for d in demands])

def demand_by_label(demands: list[Demand]) -> dict[str, Demand]:
    return {d.label: d for d in demands}

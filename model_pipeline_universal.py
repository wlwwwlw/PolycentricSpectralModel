import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import networkx as nx
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POPULATION_FILE = BASE_DIR / "population.xlsx"
RESULTS_FILE = BASE_DIR / "results.xlsx"
DISTANCES_FILE = BASE_DIR / "distances_matrix.xlsx"

OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
PER_CAPITA_BASE = 10_000
MAX_POPULATION = 50_000
REGION_NAME = "Кировская область"

# Для быстрого теста можно поставить 50–100.
# Для финального прогона лучше 300–1000.
N_TRIALS = 1000

# Сколько лучших попыток сохранять в Excel.
TOP_TRIALS_TO_EXPORT = 50


# ============================================================
# 1. ВЕСА ЦЕЛЕВОЙ ФУНКЦИИ
# ============================================================

OBJECTIVE_WEIGHTS = {
    "edge_coverage": 0.16,
    "share_close_pairs": 0.12,
    "avg_complementarity": 0.14,
    "avg_soft_reciprocity": 0.08,
    "balance_score": 0.14,
    "agglomeration_gain": 0.14,
    "avg_affinity": 0.06,
    "avg_edge_distance_inv": 0.07,
    "max_edge_distance_inv": 0.06,
    "max_pairwise_distance_inv": 0.04,
    "max_path_distance_inv": 0.04,
    "dominance_penalty_inv": 0.03,
    "size_penalty_inv": 0.02,
}

STRICT_FILTERS = {
    "disconnected_clusters_max": 0,
    "edge_coverage_min": 0.55,
    "share_close_pairs_min": 0.55,
    "agglomeration_gain_min": -0.03,
    "max_cluster_size_soft": 9,
}


# ============================================================
# 2. СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[«»\"']", "", text)
    text = text.strip()

    aliases = {
        "володарский муниципальный округ": "володарск",
        "городской округ семеновский": "семенов",
        "городской округ город кулебаки": "кулебаки",
        "лысковский муниципальный округ": "лысково",
        "городской округ город шахунья": "шахунья",
        "уренский муниципальный округ": "урень",
        "сергачский муниципальный округ": "сергач",
        "лукояновский муниципальный округ": "лукоянов",
        "починковский муниципальный округ": "починки",
        "шатковский муниципальный округ": "шатки",
        "ардатовский муниципальный округ": "ардатов",

        "городской округ навашинский": "навашино",
        "навашинский": "навашино",

        "городской округ город чкаловск": "чкаловск",
        "дальнеконстантиновский муниципальный округ": "дальнее константиново",
        "пильнинский муниципальный округ": "пильна",
        "краснобаковский муниципальный округ": "красные баки",
        "сосновский муниципальный округ": "сосновское",
        "ковернинский муниципальный округ": "ковернино",
        "городской округ город первомайск": "первомайск",
        "вачский муниципальный округ": "вача",

        "городской округ перевозский": "перевоз",
        "перевозский": "перевоз",

        "воскресенский муниципальный округ": "воскресенское",

        "городской округ сокольский": "сокольское",
        "сокольский": "сокольское",

        "тоншаевский муниципальный округ": "тоншаево",
        "бутурлинский муниципальный округ": "бутурлино",
        "княгининский муниципальный округ": "княгинино",
        "большемурашкинский муниципальный округ": "большое мурашкино",
        "варнавинский муниципальный округ": "варнавино",
        "тонкинский муниципальный округ": "тонкино",
        "дивеевский муниципальный округ": "дивеево",
        "сеченовский муниципальный округ": "сеченово",
        "вадский муниципальный округ": "вад",
        "большеболдинский муниципальный округ": "большое болдино",
        "гагинский муниципальный округ": "гагино",
        "спасский муниципальный округ": "спасское",
        "краснооктябрьский муниципальный округ": "уразовка",

        "ветлужский муниципальный округ": "ветлуга",
        "ветлужский": "ветлуга",

        "вознесенский муниципальный округ": "вознесенское",
        "вознесенский": "вознесенское",

        "шарангский муниципальный округ": "шаранга",
        "шарангский": "шаранга",

        # Кировская область: сопоставление муниципальных образований
        # с административными центрами, которые используются в файлах
        # population/distances. Это нужно, чтобы инфраструктура из results
        # корректно приклеивалась к территориям модели.
        "городской округ город вятские поляны": "вятские поляны",
        "городской округ город котельнич": "котельнич",
        "городской округ город слободской": "слободской",
        "арбажский муниципальный округ": "арбаж",
        "афанасьевский муниципальный округ": "афанасьево",
        "богородский муниципальный округ": "богородское",
        "верхнекамский муниципальный округ": "кирс",
        "кикнурский муниципальный округ": "кикнур",
        "лебяжский муниципальный округ": "лебяжье",
        "лузский муниципальный округ": "луза",
        "мурашинский муниципальный округ": "мураши",
        "немский муниципальный округ": "нема",
        "опаринский муниципальный округ": "опарино",
        "пижанский муниципальный округ": "пижанка",
        "санчурский муниципальный округ": "санчурск",
        "свечинский муниципальный округ": "свеча",
        "фаленский муниципальный округ": "фаленки",
        "унинский муниципальный округ": "уни",
        "белохолуницкий муниципальный район": "белая холуница",
        "верхошижемский муниципальный район": "верхошижемье",
        "вятскополянский муниципальный район": "вятские поляны",
        "даровской муниципальный район": "даровской",
        "зуевский муниципальный район": "зуевка",
        "кильмезский муниципальный район": "кильмезь",
        "кирово-чепецкий муниципальный район": "кирово-чепецк",
        "котельничский муниципальный район": "котельнич",
        "куменский муниципальный район": "кумены",
        "малмыжский муниципальный район": "малмыж",
        "нагорский муниципальный район": "нагорск",
        "нолинский муниципальный район": "нолинск",
        "омутнинский муниципальный район": "омутнинск",
        "оричевский муниципальный район": "оричи",
        "орловский муниципальный район": "орлов",
        "подосиновский муниципальный район": "подосиновец",
        "слободской муниципальный район": "слободской",
        "советский муниципальный район": "советск",
        "сунский муниципальный район": "суна",
        "тужинский муниципальный район": "тужа",
        "уржумский муниципальный район": "уржум",
        "шабалинский муниципальный район": "ленинское",
        "юрьянский муниципальный район": "юрья",
        "яранский муниципальный район": "яранск",
    }

    if text in aliases:
        return aliases[text]

    replacements = [
        "городской округ город ",
        "городской округ ",
        "муниципальный округ ",
        "муниципальный район ",
        "город ",
        "г. ",
        "го ",
        "мо ",
    ]

    for r in replacements:
        text = text.replace(r, "")

    text = text.strip()

    if text in aliases:
        return aliases[text]

    return text


def to_number(series: pd.Series, default=np.nan) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)

    out = pd.to_numeric(s, errors="coerce")

    if not np.isnan(default):
        out = out.fillna(default)

    return out


def robust_minmax(values, lower_q=0.05, upper_q=0.95):
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if len(arr) == 0:
        return arr

    lo = np.quantile(arr, lower_q)
    hi = np.quantile(arr, upper_q)

    if abs(hi - lo) < 1e-12:
        return np.zeros_like(arr)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1)

    return out


def weighted_average(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    weights = np.nan_to_num(weights, nan=0.0)
    weights = np.maximum(weights, 0)

    if weights.sum() <= 0:
        return float(np.mean(values))

    return float(np.average(values, weights=weights))


def find_column(df: pd.DataFrame, candidates: List[str], required=True, df_name=""):
    columns = list(df.columns)
    norm_cols = {col: normalize_text(col) for col in columns}

    for cand in candidates:
        cand_norm = normalize_text(cand)
        for col, col_norm in norm_cols.items():
            if cand_norm == col_norm or cand_norm in col_norm:
                return col

    for cand in candidates:
        words = normalize_text(cand).split()
        for col, col_norm in norm_cols.items():
            if all(w in col_norm for w in words):
                return col

    if required:
        raise ValueError(
            f"Не найдена колонка в {df_name}. "
            f"Искал {candidates}. Есть колонки: {columns}"
        )

    return None


# ============================================================
# 3. ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_population(path: Path) -> pd.DataFrame:
    print(f"\nЧитаю population: {path.name}")

    df = pd.read_excel(path)
    print("[population] колонки:", list(df.columns))

    try:
        name_col = find_column(
            df,
            [
                "населенный пункт",
                "населённый пункт",
                "пункт",
                "округ",
                "муниципальное образование",
                "территория",
                "name",
                "city",
                "town",
                "municipality",
            ],
            df_name="population",
        )

        pop_col = find_column(
            df,
            [
                "население",
                "численность",
                "population",
                "people",
                "total",
            ],
            df_name="population",
        )

        out = df[[name_col, pop_col]].copy()
        out.columns = ["town", "population"]

    except Exception:
        print("[population] Шапка не найдена. Читаю файл как таблицу без заголовков.")
        df_no_header = pd.read_excel(path, header=None)

        if df_no_header.shape[1] < 2:
            raise ValueError("В population должно быть минимум две колонки: территория и население.")

        out = df_no_header.iloc[:, :2].copy()
        out.columns = ["town", "population"]

    out["town"] = out["town"].astype(str).str.strip()
    out["town_key"] = out["town"].apply(normalize_text)
    out["population"] = to_number(out["population"], default=np.nan)

    out = out.dropna(subset=["population"])
    out = out[out["population"] > 0]
    out = out[out["population"] <= MAX_POPULATION]
    out = out[out["town_key"] != ""]
    out = out.drop_duplicates(subset=["town_key"], keep="first")
    out = out.reset_index(drop=True)

    print(f"[population] загружено территорий: {len(out)}")
    print(out.head().to_string(index=False))

    return out


def load_results(path: Path) -> pd.DataFrame:
    print(f"\nЧитаю results: {path.name}")

    df = pd.read_excel(path)
    print("[results] колонки:", list(df.columns))

    town_col = find_column(
        df,
        [
            "округ",
            "населенный пункт",
            "населённый пункт",
            "муниципальное образование",
            "территория",
            "город",
            "city",
            "town",
        ],
        df_name="results",
    )

    object_col = find_column(
        df,
        [
            "название объекта",
            "объект",
            "name",
            "object",
            "organization",
            "place",
        ],
        required=False,
        df_name="results",
    )

    type_col = find_column(
        df,
        [
            "тип здания",
            "тип объекта",
            "тип",
            "категория",
            "рубрика",
            "category",
            "type",
        ],
        df_name="results",
    )

    rating_col = find_column(
        df,
        [
            "рейтинг",
            "оценка",
            "rating",
            "rate",
            "stars",
        ],
        df_name="results",
    )

    reviews_col = find_column(
        df,
        [
            "кол-во отзывов",
            "кол во отзывов",
            "количество отзывов",
            "количество отзыв",
            "отзывы",
            "число отзывов",
            "reviews",
            "review_count",
            "count_reviews",
        ],
        df_name="results",
    )

    use_cols = [town_col, type_col, rating_col, reviews_col]
    rename = {
        town_col: "town",
        type_col: "object_type",
        rating_col: "rating",
        reviews_col: "reviews",
    }

    if object_col is not None:
        use_cols.append(object_col)
        rename[object_col] = "object_name"

    out = df[use_cols].copy()
    out = out.rename(columns=rename)

    if "object_name" not in out.columns:
        out["object_name"] = ""

    out["town"] = out["town"].astype(str).str.strip()
    out["town_key"] = out["town"].apply(normalize_text)

    out["object_name"] = out["object_name"].astype(str).str.strip()
    out["object_type"] = out["object_type"].astype(str).str.strip()

    out["rating"] = to_number(out["rating"], default=np.nan)
    out["reviews"] = to_number(out["reviews"], default=0)

    out.loc[(out["rating"] < 0) | (out["rating"] > 5), "rating"] = np.nan
    out["reviews"] = out["reviews"].fillna(0).clip(lower=0)

    print(f"[results] загружено объектов: {len(out)}")
    print(f"[results] уникальных территорий: {out['town_key'].nunique()}")

    return out


def load_distances(path: Path) -> pd.DataFrame:
    print(f"\nЧитаю distances: {path.name}")

    raw = pd.read_excel(path)

    first_col = raw.columns[0]
    raw = raw.set_index(first_col)

    raw = raw.dropna(how="all", axis=0)
    raw = raw.dropna(how="all", axis=1)

    raw.index = [normalize_text(x) for x in raw.index]
    raw.columns = [normalize_text(x) for x in raw.columns]

    dist = raw.copy()

    for col in dist.columns:
        dist[col] = to_number(dist[col], default=np.nan)

    dist = dist.loc[~pd.Index(dist.index).duplicated(keep="first")]
    dist = dist.loc[:, ~pd.Index(dist.columns).duplicated(keep="first")]

    common = [x for x in dist.index if x in set(dist.columns)]
    dist = dist.loc[common, common]

    arr = dist.values.astype(float)
    sym = np.nanmean(np.stack([arr, arr.T]), axis=0)

    dist = pd.DataFrame(sym, index=common, columns=common)

    for x in common:
        dist.loc[x, x] = 0.0

    print(f"[distances] размер матрицы: {dist.shape}")
    print("[distances] первые индексы:", list(dist.index[:5]))

    return dist


def align_all_data(population, results, distances):
    print_header("ПРОВЕРКА СОВПАДЕНИЯ НАЗВАНИЙ")

    pop_keys = set(population["town_key"])
    result_keys = set(results["town_key"])
    dist_keys = set(distances.index)

    print(f"Пунктов в population: {len(pop_keys)}")
    print(f"Пунктов в results:    {len(result_keys)}")
    print(f"Пунктов в distances:  {len(dist_keys)}")

    pop_not_results = sorted(pop_keys - result_keys)
    pop_not_dist = sorted(pop_keys - dist_keys)
    results_not_pop = sorted(result_keys - pop_keys)

    print("\nВ population есть, но в results нет:")
    if pop_not_results:
        for x in pop_not_results:
            print(" -", x)
    else:
        print(" - нет")

    print("\nВ population есть, но в distances нет:")
    if pop_not_dist:
        for x in pop_not_dist:
            print(" -", x)
    else:
        print(" - нет")

    print("\nВ results есть, но в population нет:")
    if results_not_pop:
        for x in results_not_pop[:50]:
            print(" -", x)
    else:
        print(" - нет")

    # Для модели обязательно нужны population + distances.
    # Если по территории нет объектов в results, она не выкидывается:
    # просто получит нулевые значения по инфраструктуре.
    common = sorted(list(pop_keys & dist_keys))

    if len(common) < 5:
        raise ValueError("Слишком мало совпавших территорий между population и distances.")

    population_aligned = population[population["town_key"].isin(common)].copy()
    population_aligned = population_aligned.sort_values("town_key").reset_index(drop=True)

    results_aligned = results[results["town_key"].isin(common)].copy()

    distances_aligned = distances.loc[
        population_aligned["town_key"],
        population_aligned["town_key"],
    ].copy()

    audit = {
        "population_not_in_results": pop_not_results,
        "population_not_in_distances": pop_not_dist,
        "results_not_in_population": results_not_pop,
        "model_towns": population_aligned["town_key"].tolist(),
    }

    with open(OUTPUT_DIR / "name_matching_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"\nИтоговое количество территорий для модели: {len(population_aligned)}")
    print(f"Аудит сопоставления сохранён: {OUTPUT_DIR / 'name_matching_audit.json'}")

    return population_aligned, results_aligned, distances_aligned


# ============================================================
# 4. СФЕРЫ ЖИЗНИ И ПРИЗНАКИ
# ============================================================

SECTOR_KEYWORDS = {
    "медицина": [
        "больниц", "поликлиник", "црб", "медицин", "медцентр", "мед центр",
        "стоматолог", "аптек", "фельдшер", "амбулатор", "здрав", "клиник",
        "лаборатор", "травмпункт", "оптик", "районная больница",
    ],
    "образование": [
        "школ", "гимнази", "лицей", "детский сад", "садик", "доу",
        "колледж", "техникум", "университет", "институт", "образован",
        "учеб", "центр развития", "репетитор", "курсы",
    ],
    "торговля": [
        "магазин", "супермаркет", "продукт", "рынок", "торгов", "тц",
        "пятерочка", "магнит", "fix price", "фикс прайс", "пункт выдачи",
        "wildberries", "ozon", "автозапчаст", "одежд", "обув", "хозтовар",
        "стройматериал", "мебел", "цвет", "канцтовар",
    ],
    "культура": [
        "дом культуры", "дк", "культур", "библиотек", "музе", "театр",
        "кино", "выстав", "клуб", "творчеств",
    ],
    "спорт": [
        "спорт", "стадион", "фитнес", "тренаж", "бассейн", "фок",
        "зал", "секци", "каток", "лыж", "единоборств", "физкультур",
    ],
    "административные_услуги": [
        "администрац", "мфц", "почта", "роспочта", "суд", "прокуратур",
        "полици", "гибдд", "мвд", "загс", "пенсион", "налог", "служба",
        "центр занятости", "соцзащит", "росреестр",
    ],
    "транспорт": [
        "автостанц", "автовокзал", "жд", "железнодорож", "вокзал",
        "останов", "такси", "транспорт", "заправ", "азс", "автосервис",
        "шиномонтаж", "сто", "мойка", "парков",
    ],
    "общепит": [
        "кафе", "ресторан", "столов", "пицц", "суши", "бар", "кофе",
        "пекар", "кулинар", "шаурм", "быстрое питание", "общепит",
    ],
    "бытовые_услуги": [
        "парикмах", "салон красоты", "космет", "ремонт", "ателье", "химчист",
        "прачеч", "ритуал", "фото", "типограф", "бытов", "услуг",
        "клининг", "сервисный центр", "мастерская",
    ],
    "жкх_и_коммунальные": [
        "жкх", "коммун", "водоканал", "теплосет", "электросет", "газ",
        "управляющая компания", "тсж", "котельн", "энерг",
    ],
    "досуг_и_туризм": [
        "парк", "сквер", "гостиниц", "отель", "туризм", "база отдыха",
        "развлеч", "баня", "сауна", "пляж", "достопримеч",
    ],
    "финансы": [
        "банк", "сбер", "втб", "банкомат", "финанс", "кредит", "страхов",
        "ломбард",
    ],
    "религия": [
        "церковь", "храм", "монастыр", "мечеть", "религи", "приход",
    ],
}


def detect_sector(object_type: str, object_name: str = "") -> str:
    text = normalize_text(f"{object_type} {object_name}")

    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if normalize_text(kw) in text:
                return sector

    return "прочее"


def add_smoothed_rating(results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Байесовское сглаживание рейтинга:
    R_smooth = (n * R + alpha * R_global) / (n + alpha)
    """
    df = results.copy()

    global_rating = df["rating"].dropna().mean()

    if pd.isna(global_rating):
        global_rating = 4.0

    df["rating_filled"] = df["rating"].fillna(global_rating)
    df["reviews"] = df["reviews"].fillna(0).clip(lower=0)

    df["smoothed_rating"] = (
        df["reviews"] * df["rating_filled"] + alpha * global_rating
    ) / (df["reviews"] + alpha)

    # Логарифмический вес, чтобы объекты с тысячами отзывов не задавили все остальные.
    df["rating_weight"] = np.log1p(df["reviews"]) + 1

    return df


def build_features(population: pd.DataFrame, results: pd.DataFrame, alpha: float):
    df = add_smoothed_rating(results, alpha=alpha)

    if len(df) > 0:
        df["sector"] = df.apply(
            lambda row: detect_sector(row["object_type"], row.get("object_name", "")),
            axis=1,
        )
    else:
        df["sector"] = []

    pop_keys = set(population["town_key"])
    df = df[df["town_key"].isin(pop_keys)].copy()

    sectors = sorted(df["sector"].dropna().unique().tolist())

    if "прочее" in sectors:
        sectors = [s for s in sectors if s != "прочее"] + ["прочее"]

    rows = []

    if len(df) > 0:
        for (town_key, sector), g in df.groupby(["town_key", "sector"]):
            rows.append(
                {
                    "town_key": town_key,
                    "sector": sector,
                    "object_count": len(g),
                    "quality": weighted_average(g["smoothed_rating"], g["rating_weight"]),
                    "reviews_sum": g["reviews"].sum(),
                    "reviews_mean": g["reviews"].mean(),
                }
            )

    agg = pd.DataFrame(rows)

    features = population[["town", "town_key", "population"]].copy()
    features["population_log"] = np.log1p(features["population"])

    for sector in sectors:
        tmp = agg[agg["sector"] == sector].copy()

        count_map = dict(zip(tmp["town_key"], tmp["object_count"]))
        quality_map = dict(zip(tmp["town_key"], tmp["quality"]))
        reviews_map = dict(zip(tmp["town_key"], tmp["reviews_sum"]))

        count_col = f"{sector}__count"
        per_col = f"{sector}__per_{PER_CAPITA_BASE}"
        quality_col = f"{sector}__quality"
        reviews_col = f"{sector}__reviews"

        features[count_col] = features["town_key"].map(count_map).fillna(0)
        features[quality_col] = features["town_key"].map(quality_map).fillna(0)
        features[reviews_col] = features["town_key"].map(reviews_map).fillna(0)
        features[per_col] = features[count_col] / features["population"] * PER_CAPITA_BASE

    # В основную модель не включаем "прочее".
    model_sectors = [s for s in sectors if s != "прочее"]

    sector_scores = pd.DataFrame(index=features["town_key"])

    for sector in model_sectors:
        per_col = f"{sector}__per_{PER_CAPITA_BASE}"
        quality_col = f"{sector}__quality"

        per_norm = robust_minmax(features[per_col].values)
        quality_norm = robust_minmax(features[quality_col].values)

        # Обеспеченность важнее рейтинга.
        score = 0.75 * per_norm + 0.25 * quality_norm

        sector_scores[sector] = score
        features[f"{sector}__score"] = score

    if sector_scores.shape[1] == 0:
        features["dummy_sector__score"] = 0.0
        sector_scores["dummy_sector"] = 0.0

    return features, sector_scores, sectors


def make_feature_matrix(features: pd.DataFrame):
    exclude = {"town", "town_key"}

    numeric_cols = [
        c for c in features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
    ]

    X = features[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X.values.astype(float), numeric_cols


# ============================================================
# 5. МАТРИЦЫ СВЯЗИ
# ============================================================

def compute_directed_help_matrices(sector_scores: pd.DataFrame):
    """
    help_i_to_j — насколько сильные стороны i закрывают слабые стороны j.
    complementarity — средняя двусторонняя взаимодополняемость.
    soft_reciprocity — мягкая двусторонность.
    """
    S = sector_scores.fillna(0).copy()

    Z = pd.DataFrame(index=S.index)

    for col in S.columns:
        values = S[col].values.astype(float)
        med = np.nanmedian(values)
        std = np.nanstd(values)

        if std < 1e-12:
            Z[col] = 0.0
        else:
            Z[col] = (values - med) / std

    z = Z.values.astype(float)
    strength = np.maximum(z, 0)
    deficit = np.maximum(-z, 0)

    n = z.shape[0]
    help_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            deficit_sum = np.sum(deficit[j])

            if deficit_sum <= 1e-12:
                help_matrix[i, j] = 0.0
            else:
                help_matrix[i, j] = np.sum(strength[i] * deficit[j]) / deficit_sum

    if np.nanmax(help_matrix) - np.nanmin(help_matrix) > 1e-12:
        help_norm = robust_minmax(help_matrix, 0.05, 0.95)
    else:
        help_norm = help_matrix

    np.fill_diagonal(help_norm, 0)

    comp = (help_norm + help_norm.T) / 2
    np.fill_diagonal(comp, 0)

    raw_rec = np.zeros((n, n), dtype=float)
    soft_rec = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = help_norm[i, j]
            b = help_norm[j, i]
            mx = max(a, b)

            if mx <= 1e-12:
                r = 0.0
            else:
                r = min(a, b) / mx

            raw_rec[i, j] = r
            soft_rec[i, j] = 0.5 + 0.5 * r

    np.fill_diagonal(raw_rec, 0)
    np.fill_diagonal(soft_rec, 0)

    towns = S.index

    return (
        pd.DataFrame(help_norm, index=towns, columns=towns),
        pd.DataFrame(comp, index=towns, columns=towns),
        pd.DataFrame(raw_rec, index=towns, columns=towns),
        pd.DataFrame(soft_rec, index=towns, columns=towns),
    )


def build_knn_mask(distances: pd.DataFrame, k: int):
    towns = list(distances.index)
    d = distances.values.astype(float)
    n = len(towns)

    mask = np.zeros((n, n), dtype=float)

    if k <= 0:
        return pd.DataFrame(mask, index=towns, columns=towns)

    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        nearest = np.argsort(row)[:k]

        for j in nearest:
            if np.isfinite(row[j]):
                mask[i, j] = 1

    mask = np.maximum(mask, mask.T)
    np.fill_diagonal(mask, 0)

    return pd.DataFrame(mask, index=towns, columns=towns)


def make_spatial_mask(distances: pd.DataFrame, threshold: float, k: int, hard_limit: float):
    d = distances.values.astype(float)
    towns = list(distances.index)

    dist_mask = (d <= threshold).astype(float)
    np.fill_diagonal(dist_mask, 0)

    knn_mask = build_knn_mask(distances, k).values.astype(float)

    mask = np.maximum(dist_mask, knn_mask)
    mask[d > hard_limit] = 0
    np.fill_diagonal(mask, 0)

    return pd.DataFrame(mask, index=towns, columns=towns)


def make_affinity(
    distances: pd.DataFrame,
    complementarity: pd.DataFrame,
    soft_reciprocity: pd.DataFrame,
    threshold: float,
    k: int,
    hard_limit: float,
    tau: float,
):
    d = distances.values.astype(float)
    comp = complementarity.values.astype(float)
    rec = soft_reciprocity.values.astype(float)

    mask = make_spatial_mask(distances, threshold, k, hard_limit)

    closeness = np.exp(-d / tau)
    np.fill_diagonal(closeness, 0)

    affinity = mask.values.astype(float) * comp * closeness * rec
    np.fill_diagonal(affinity, 1.0)

    affinity = pd.DataFrame(affinity, index=distances.index, columns=distances.columns)

    return affinity, mask


# ============================================================
# 6. МЕТРИКИ
# ============================================================

def labels_to_int(labels):
    labels = np.asarray(labels)
    unique = sorted(pd.unique(labels))

    mapping = {}
    next_id = 0

    for u in unique:
        if u == -1:
            mapping[u] = -1
        else:
            mapping[u] = next_id
            next_id += 1

    return np.array([mapping[x] for x in labels], dtype=int)


def cluster_pair_indices(labels):
    labels = np.asarray(labels)
    pairs = []

    for lab in sorted(set(labels)):
        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]

        if len(idx) < 2:
            continue

        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs.append((idx[a], idx[b]))

    return pairs


def compute_single_town_balance(sector_scores):
    S = sector_scores.values.astype(float)
    balances = []

    for row in S:
        mean_level = float(np.mean(row))
        std_level = float(np.std(row))
        evenness = 1 - min(std_level / 0.5, 1)
        coverage = float(np.mean(row >= 0.5))
        balance = 0.45 * mean_level + 0.35 * evenness + 0.20 * coverage
        balances.append(balance)

    return np.asarray(balances, dtype=float)


def build_graph_from_affinity(affinity: pd.DataFrame, distances: pd.DataFrame):
    towns = list(affinity.index)
    A = affinity.values.astype(float)
    D = distances.loc[towns, towns].values.astype(float)

    G = nx.Graph()

    for t in towns:
        G.add_node(t)

    for i in range(len(towns)):
        for j in range(i + 1, len(towns)):
            w = float(A[i, j])

            if w > 0:
                G.add_edge(
                    towns[i],
                    towns[j],
                    weight=w,
                    distance=float(D[i, j]),
                )

    return G


def graph_path_metrics_for_clusters(labels, towns, G):
    labels = labels_to_int(labels)

    avg_paths = []
    max_paths = []
    disconnected_clusters = 0
    connected_components_total = 0

    for lab in sorted(set(labels)):
        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]
        nodes = [towns[i] for i in idx]

        if len(nodes) < 2:
            continue

        H = G.subgraph(nodes).copy()

        if H.number_of_edges() == 0:
            disconnected_clusters += 1
            connected_components_total += len(nodes)
            continue

        comps = list(nx.connected_components(H))
        connected_components_total += len(comps)

        if len(comps) > 1:
            disconnected_clusters += 1

        for comp_nodes in comps:
            comp_nodes = list(comp_nodes)

            if len(comp_nodes) < 2:
                continue

            lengths = dict(
                nx.all_pairs_dijkstra_path_length(
                    H.subgraph(comp_nodes),
                    weight="distance",
                )
            )

            vals = []

            for a in range(len(comp_nodes)):
                for b in range(a + 1, len(comp_nodes)):
                    u = comp_nodes[a]
                    v = comp_nodes[b]

                    if v in lengths.get(u, {}):
                        vals.append(lengths[u][v])

            if vals:
                avg_paths.append(float(np.mean(vals)))
                max_paths.append(float(np.max(vals)))

    return {
        "avg_path_distance": float(np.mean(avg_paths)) if avg_paths else np.nan,
        "max_path_distance": float(np.max(max_paths)) if max_paths else np.nan,
        "disconnected_clusters": disconnected_clusters,
        "connected_components_total": connected_components_total,
    }


def cluster_dominance_penalty(labels, sector_scores):
    labels = labels_to_int(labels)
    S = sector_scores.values.astype(float)

    penalties = []

    for lab in sorted(set(labels)):
        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]

        if len(idx) < 2:
            continue

        cluster_scores = S[idx]
        town_power = cluster_scores.sum(axis=1)
        total_power = town_power.sum()

        if total_power <= 1e-12:
            penalties.append(0.0)
            continue

        max_share = float(town_power.max() / total_power)

        penalty = max(0.0, max_share - 0.45) / 0.55
        penalty = min(penalty, 1.0)

        penalties.append(penalty)

    if penalties:
        return float(np.mean(penalties))

    return 0.0


def evaluate_labels(
    labels,
    X_eval,
    distances,
    complementarity,
    soft_reciprocity,
    affinity,
    mask,
    sector_scores,
):
    labels = labels_to_int(labels)
    n = len(labels)
    towns = list(distances.index)

    non_noise_labels = [x for x in sorted(set(labels)) if x != -1]
    cluster_sizes = [int(np.sum(labels == lab)) for lab in non_noise_labels]

    n_clusters = len(non_noise_labels)
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / max(n, 1)

    singleton_count = sum(1 for s in cluster_sizes if s == 1)
    singleton_ratio = singleton_count / max(n_clusters, 1)

    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    mean_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0

    size_penalty = max(0.0, (max_cluster_size - STRICT_FILTERS["max_cluster_size_soft"]) / max(n, 1))

    result = {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": noise_ratio,
        "singleton_count": singleton_count,
        "singleton_ratio": singleton_ratio,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max_cluster_size,
        "mean_cluster_size": mean_cluster_size,
        "size_penalty": size_penalty,

        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,

        "avg_pairwise_distance": np.nan,
        "max_pairwise_distance": np.nan,
        "share_close_pairs": np.nan,

        "avg_edge_distance": np.nan,
        "max_edge_distance": np.nan,
        "edge_coverage": np.nan,

        "avg_path_distance": np.nan,
        "max_path_distance": np.nan,
        "disconnected_clusters": np.nan,
        "connected_components_total": np.nan,

        "avg_complementarity": np.nan,
        "avg_soft_reciprocity": np.nan,
        "avg_affinity": np.nan,

        "balance_score": np.nan,
        "agglomeration_gain": np.nan,
        "dominance_penalty": np.nan,
    }

    if n_clusters >= 2 and n_clusters < len(labels):
        try:
            result["silhouette"] = float(silhouette_score(X_eval, labels))
        except Exception:
            pass

        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels))
        except Exception:
            pass

        try:
            result["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, labels))
        except Exception:
            pass

    pairs = cluster_pair_indices(labels)

    if pairs:
        dist_arr = distances.values.astype(float)
        comp_arr = complementarity.values.astype(float)
        rec_arr = soft_reciprocity.values.astype(float)
        aff_arr = affinity.values.astype(float)
        mask_arr = mask.values.astype(float)

        pair_distances = np.asarray([dist_arr[i, j] for i, j in pairs])
        pair_comp = np.asarray([comp_arr[i, j] for i, j in pairs])
        pair_rec = np.asarray([rec_arr[i, j] for i, j in pairs])
        pair_aff = np.asarray([aff_arr[i, j] for i, j in pairs])
        pair_mask = np.asarray([mask_arr[i, j] for i, j in pairs])

        result["avg_pairwise_distance"] = float(np.mean(pair_distances))
        result["max_pairwise_distance"] = float(np.max(pair_distances))
        result["share_close_pairs"] = float(np.mean(pair_mask > 0))

        result["avg_complementarity"] = float(np.mean(pair_comp))
        result["avg_soft_reciprocity"] = float(np.mean(pair_rec))
        result["avg_affinity"] = float(np.mean(pair_aff))

        used = pair_aff > 0

        if np.any(used):
            result["avg_edge_distance"] = float(np.mean(pair_distances[used]))
            result["max_edge_distance"] = float(np.max(pair_distances[used]))
            result["edge_coverage"] = float(np.mean(used))

    G = build_graph_from_affinity(affinity, distances)
    result.update(graph_path_metrics_for_clusters(labels, towns, G))

    S = sector_scores.values.astype(float)
    single_balances = compute_single_town_balance(sector_scores)

    cluster_balances = []
    cluster_gains = []

    for lab in non_noise_labels:
        idx = np.where(labels == lab)[0]

        if len(idx) == 0:
            continue

        cluster_profile = np.mean(S[idx], axis=0)

        mean_level = float(np.mean(cluster_profile))
        std_level = float(np.std(cluster_profile))
        evenness = 1 - min(std_level / 0.5, 1)
        coverage = float(np.mean(cluster_profile >= 0.5))

        cluster_balance = 0.45 * mean_level + 0.35 * evenness + 0.20 * coverage
        before_balance = float(np.mean(single_balances[idx]))

        cluster_balances.append(cluster_balance)
        cluster_gains.append(cluster_balance - before_balance)

    if cluster_balances:
        result["balance_score"] = float(np.mean(cluster_balances))

    if cluster_gains:
        result["agglomeration_gain"] = float(np.mean(cluster_gains))

    result["dominance_penalty"] = cluster_dominance_penalty(labels, sector_scores)

    return result


# ============================================================
# 7. ЦЕЛЕВАЯ ФУНКЦИЯ OPTUNA
# ============================================================

def get_distance_stats(distances: pd.DataFrame) -> Dict[str, float]:
    arr = distances.values.astype(float)
    vals = arr[np.triu_indices_from(arr, k=1)]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if len(vals) == 0:
        return {
            "q10": 50.0,
            "q25": 100.0,
            "q50": 150.0,
            "q75": 200.0,
            "q90": 250.0,
            "max": 300.0,
        }

    return {
        "q10": float(np.quantile(vals, 0.10)),
        "q25": float(np.quantile(vals, 0.25)),
        "q50": float(np.quantile(vals, 0.50)),
        "q75": float(np.quantile(vals, 0.75)),
        "q90": float(np.quantile(vals, 0.90)),
        "max": float(np.max(vals)),
    }


def inv_distance_score(value, scale):
    if pd.isna(value):
        return 0.0

    return max(0.0, 1.0 - float(value) / max(float(scale), 1e-9))


def objective_score(metrics: Dict[str, float], dist_stats: Dict[str, float]) -> float:
    max_edge_scale = max(dist_stats["q50"], 1.0)
    max_pair_scale = max(dist_stats["q75"], 1.0)
    max_path_scale = max(dist_stats["q90"], 1.0)

    components = {
        "edge_coverage": float(metrics.get("edge_coverage", 0) or 0),
        "share_close_pairs": float(metrics.get("share_close_pairs", 0) or 0),
        "avg_complementarity": float(metrics.get("avg_complementarity", 0) or 0),
        "avg_soft_reciprocity": float(metrics.get("avg_soft_reciprocity", 0) or 0),
        "balance_score": float(metrics.get("balance_score", 0) or 0),
        "agglomeration_gain": max(float(metrics.get("agglomeration_gain", 0) or 0), -0.1),
        "avg_affinity": float(metrics.get("avg_affinity", 0) or 0),

        "avg_edge_distance_inv": inv_distance_score(metrics.get("avg_edge_distance", np.nan), max_edge_scale),
        "max_edge_distance_inv": inv_distance_score(metrics.get("max_edge_distance", np.nan), dist_stats["q75"]),
        "max_pairwise_distance_inv": inv_distance_score(metrics.get("max_pairwise_distance", np.nan), max_pair_scale),
        "max_path_distance_inv": inv_distance_score(metrics.get("max_path_distance", np.nan), max_path_scale),

        "dominance_penalty_inv": 1.0 - min(float(metrics.get("dominance_penalty", 0) or 0), 1.0),
        "size_penalty_inv": 1.0 - min(float(metrics.get("size_penalty", 0) or 0), 1.0),
    }

    score = 0.0

    for key, weight in OBJECTIVE_WEIGHTS.items():
        score += weight * components.get(key, 0.0)

    # Жёсткие штрафы
    if (metrics.get("disconnected_clusters", 0) or 0) > STRICT_FILTERS["disconnected_clusters_max"]:
        score -= 0.30 * float(metrics.get("disconnected_clusters", 0) or 0)

    if (metrics.get("edge_coverage", 0) or 0) < STRICT_FILTERS["edge_coverage_min"]:
        score -= 0.15

    if (metrics.get("share_close_pairs", 0) or 0) < STRICT_FILTERS["share_close_pairs_min"]:
        score -= 0.10

    if (metrics.get("agglomeration_gain", 0) or 0) < STRICT_FILTERS["agglomeration_gain_min"]:
        score -= 0.10

    if (metrics.get("singleton_count", 0) or 0) > 0:
        score -= 0.05 * float(metrics.get("singleton_count", 0) or 0)

    score -= 0.08 * float(metrics.get("size_penalty", 0) or 0)

    # Дополнительные штрафы для моделей, которые дали непригодную кластерную структуру.
    if (metrics.get("n_clusters", 0) or 0) < 2:
        score -= 0.50

    if (metrics.get("noise_ratio", 0) or 0) > 0.40:
        score -= 0.20 * float(metrics.get("noise_ratio", 0) or 0)

    return float(score)


def suggest_param_space_from_distances(distances: pd.DataFrame, n_towns: int):
    ds = get_distance_stats(distances)

    threshold_low = max(20.0, ds["q10"] * 0.75)
    threshold_high = max(threshold_low + 10.0, ds["q50"] * 0.90)

    hard_low = max(threshold_high + 10.0, ds["q25"])
    hard_high = max(hard_low + 20.0, min(ds["q90"], ds["q75"] * 1.35))

    tau_low = max(20.0, ds["q10"] * 0.75)
    tau_high = max(tau_low + 10.0, ds["q75"])

    min_clusters = max(3, min(5, n_towns // 8))
    max_clusters = min(12, max(5, n_towns // 3))

    return {
        "distance_stats": ds,
        "alpha_values": [5, 10, 20, 30, 50],
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "hard_low": hard_low,
        "hard_high": hard_high,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "knn_values": [1, 2, 3, 4, 5],
        "n_clusters_low": min_clusters,
        "n_clusters_high": max_clusters,
        "assign_labels_values": ["kmeans", "discretize", "cluster_qr"],
    }


# ============================================================
# 8. ЗАПУСК ОДНОЙ МОДЕЛИ
# ============================================================

def run_model_once(params: Dict, population: pd.DataFrame, results: pd.DataFrame, distances: pd.DataFrame):
    """
    Запускает одну конфигурацию выбранной модели.

    Универсальная версия пайплайна больше не сравнивает разные алгоритмы:
    в ВКР лучшей моделью уже выбрана SpectralClustering. Здесь Optuna
    подбирает только параметры построения полицентрической связи и
    параметры спектральной кластеризации.
    """
    params = params.copy()
    params["model_type"] = "spectral"

    alpha = params.get("alpha", 10)
    threshold = float(params.get("threshold", 100.0))
    knn = int(params.get("knn", 3))
    hard_limit = float(params.get("hard_limit", threshold + 50.0))
    tau = float(params.get("tau", 100.0))

    features, sector_scores, sectors = build_features(population, results, alpha=alpha)
    features = features.set_index("town_key").loc[population["town_key"]].reset_index()
    sector_scores = sector_scores.loc[population["town_key"]]

    X_raw, feature_cols = make_feature_matrix(features)
    X_eval = StandardScaler().fit_transform(X_raw)
    X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

    help_matrix, complementarity, raw_reciprocity, soft_reciprocity = compute_directed_help_matrices(sector_scores)

    affinity, mask = make_affinity(
        distances=distances,
        complementarity=complementarity,
        soft_reciprocity=soft_reciprocity,
        threshold=threshold,
        k=knn,
        hard_limit=hard_limit,
        tau=tau,
    )

    n_objects = len(population)
    n_clusters = int(params["n_clusters"])
    assign_labels = params.get("assign_labels", "cluster_qr")

    if n_clusters < 2 or n_clusters >= n_objects:
        raise ValueError("Некорректное число кластеров для SpectralClustering.")

    A = affinity.values.astype(float)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels=assign_labels,
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(A)
    labels = labels_to_int(labels)

    metrics = evaluate_labels(
        labels=labels,
        X_eval=X_eval,
        distances=distances,
        complementarity=complementarity,
        soft_reciprocity=soft_reciprocity,
        affinity=affinity,
        mask=mask,
        sector_scores=sector_scores,
    )

    return {
        "labels": labels,
        "features": features,
        "sector_scores": sector_scores,
        "sectors": sectors,
        "help_matrix": help_matrix,
        "complementarity": complementarity,
        "soft_reciprocity": soft_reciprocity,
        "affinity": affinity,
        "mask": mask,
        "metrics": metrics,
    }


# ============================================================
# 9. OPTUNA
# ============================================================

def run_optuna_search(population, results, distances):
    print_header("АДАПТИВНЫЙ ПОДБОР ГИПЕРПАРАМЕТРОВ SPECTRAL CLUSTERING")

    n_towns = len(population)
    param_space = suggest_param_space_from_distances(distances, n_towns)
    dist_stats = param_space["distance_stats"]

    print("Распределение расстояний региона:")
    for k, v in dist_stats.items():
        print(f"{k}: {v:.2f}")

    print("\nГраницы адаптивного подбора:")
    for k, v in param_space.items():
        if k != "distance_stats":
            print(f"{k}: {v}")

    all_rows = []
    best_score = -1e18
    best_params = None
    best_pack = None

    def evaluate_params(params, trial_number=None):
        nonlocal best_score, best_params, best_pack, all_rows

        params = params.copy()
        params["model_type"] = "spectral"

        try:
            pack = run_model_once(params, population, results, distances)
            metrics = pack["metrics"]
            score = objective_score(metrics, dist_stats)

            row = {
                **params,
                "trial": trial_number,
                "score": score,
                "model": "spectral",
            }
            row.update(metrics)
            all_rows.append(row)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_pack = pack

            return score

        except Exception as e:
            row = {
                **params,
                "trial": trial_number,
                "score": -1e9,
                "model": "spectral",
                "error": str(e),
            }
            all_rows.append(row)
            return -1e9

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        threshold = trial.suggest_float(
            "threshold",
            param_space["threshold_low"],
            param_space["threshold_high"],
        )

        hard_limit = trial.suggest_float(
            "hard_limit",
            max(threshold + 1.0, param_space["hard_low"]),
            param_space["hard_high"],
        )

        params = {
            "model_type": "spectral",
            "alpha": trial.suggest_categorical("alpha", param_space["alpha_values"]),
            "threshold": threshold,
            "knn": trial.suggest_categorical("knn", param_space["knn_values"]),
            "hard_limit": hard_limit,
            "tau": trial.suggest_float("tau", param_space["tau_low"], param_space["tau_high"]),
            "n_clusters": trial.suggest_int(
                "n_clusters",
                param_space["n_clusters_low"],
                param_space["n_clusters_high"],
            ),
            "assign_labels": trial.suggest_categorical(
                "assign_labels",
                param_space["assign_labels_values"],
            ),
        }

        return evaluate_params(params, trial_number=trial.number)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except TypeError:
        study.optimize(objective, n_trials=N_TRIALS)

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)

    results_df.to_csv(OUTPUT_DIR / "all_trials_results.csv", index=False, encoding="utf-8-sig")

    if best_params is None or best_pack is None:
        raise RuntimeError("Не удалось подобрать ни одной рабочей конфигурации SpectralClustering.")

    with open(OUTPUT_DIR / "adaptive_param_space.json", "w", encoding="utf-8") as f:
        json.dump(param_space, f, ensure_ascii=False, indent=2)

    print("\nЛучшие параметры SpectralClustering:")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    print(f"Лучший score: {best_score:.4f}")

    # Файл оставлен для совместимости с интерфейсом/архивом результатов,
    # но в универсальной версии в нем всегда одна строка: spectral.
    best_by_model = results_df[results_df["score"] > -1e8].sort_values("score", ascending=False).head(1)
    best_by_model.to_csv(OUTPUT_DIR / "best_by_model_type.csv", index=False, encoding="utf-8-sig")

    return best_params, best_pack, results_df


# ============================================================
# 10. ОПИСАНИЕ АГЛОМЕРАЦИЙ И РОЛЕЙ
# ============================================================

def describe_clusters(labels, population, sector_scores, distances, complementarity, soft_reciprocity, affinity, mask):
    labels = labels_to_int(labels)

    town_keys = population["town_key"].tolist()
    town_names = dict(zip(population["town_key"], population["town"]))
    pop_map = dict(zip(population["town_key"], population["population"]))

    rows = []
    members = []
    edges = []

    S = sector_scores.copy()

    for lab in sorted(set(labels)):
        idx = np.where(labels == lab)[0]
        keys = [town_keys[i] for i in idx]

        cluster_name = f"Агломерация {lab + 1}" if lab != -1 else "шум / не отнесено"

        if not keys:
            continue

        pops = np.asarray([pop_map[k] for k in keys], dtype=float)
        center_key = keys[int(np.argmax(pops))]

        internal_pair_distances = []
        internal_edge_distances = []
        internal_comps = []
        internal_recs = []
        internal_affs = []
        used_edges = 0
        all_pairs = 0

        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ka = keys[a]
                kb = keys[b]
                all_pairs += 1

                d = float(distances.loc[ka, kb])
                c = float(complementarity.loc[ka, kb])
                r = float(soft_reciprocity.loc[ka, kb])
                aff = float(affinity.loc[ka, kb])
                m = float(mask.loc[ka, kb])

                internal_pair_distances.append(d)
                internal_comps.append(c)
                internal_recs.append(r)
                internal_affs.append(aff)

                if aff > 0:
                    used_edges += 1
                    internal_edge_distances.append(d)

                    edges.append(
                        {
                            "cluster": cluster_name,
                            "from": town_names[ka],
                            "to": town_names[kb],
                            "distance_km": d,
                            "complementarity": c,
                            "soft_reciprocity": r,
                            "affinity": aff,
                            "spatial_mask": m,
                        }
                    )

        cluster_profile = S.loc[keys].mean(axis=0)

        strong = cluster_profile.sort_values(ascending=False).head(3)
        weak = cluster_profile.sort_values(ascending=True).head(3)

        rows.append(
            {
                "cluster": cluster_name,
                "label": lab,
                "n_towns": len(keys),
                "towns": ", ".join([town_names[k] for k in keys]),
                "center_by_population": town_names[center_key],
                "total_population": float(pops.sum()),

                "avg_pairwise_distance_km": float(np.mean(internal_pair_distances)) if internal_pair_distances else np.nan,
                "max_pairwise_distance_km": float(np.max(internal_pair_distances)) if internal_pair_distances else np.nan,
                "avg_edge_distance_km": float(np.mean(internal_edge_distances)) if internal_edge_distances else np.nan,
                "max_edge_distance_km": float(np.max(internal_edge_distances)) if internal_edge_distances else np.nan,
                "edge_coverage": used_edges / all_pairs if all_pairs > 0 else np.nan,

                "avg_internal_complementarity": float(np.mean(internal_comps)) if internal_comps else np.nan,
                "avg_internal_soft_reciprocity": float(np.mean(internal_recs)) if internal_recs else np.nan,
                "avg_internal_affinity": float(np.mean(internal_affs)) if internal_affs else np.nan,

                "strong_sectors": ", ".join([f"{s} ({v:.2f})" for s, v in strong.items()]),
                "weak_sectors": ", ".join([f"{s} ({v:.2f})" for s, v in weak.items()]),
            }
        )

        for k in keys:
            profile = S.loc[k].sort_values(ascending=False)

            members.append(
                {
                    "cluster": cluster_name,
                    "town": town_names[k],
                    "population": pop_map[k],
                    "strongest_sectors": ", ".join([f"{s} ({v:.2f})" for s, v in profile.head(3).items()]),
                    "weakest_sectors": ", ".join([f"{s} ({v:.2f})" for s, v in profile.tail(3).items()]),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(members), pd.DataFrame(edges)


def compute_territory_roles(labels, population, sector_scores, distances):
    labels = labels_to_int(labels)

    town_keys = population["town_key"].tolist()
    town_names = dict(zip(population["town_key"], population["town"]))
    pop_map = dict(zip(population["town_key"], population["population"]))

    S = sector_scores.copy()

    pop_values = population.set_index("town_key").loc[town_keys, "population"].values.astype(float)
    pop_norm = robust_minmax(np.log1p(pop_values), 0.05, 0.95)
    pop_norm_map = dict(zip(town_keys, pop_norm))

    all_rows = []

    for lab in sorted(set(labels)):
        idx = np.where(labels == lab)[0]
        keys = [town_keys[i] for i in idx]

        cluster_name = f"Агломерация {lab + 1}" if lab != -1 else "шум / не отнесено"

        if not keys:
            continue

        cluster_dist = distances.loc[keys, keys].copy()

        mean_dists = {}

        for k in keys:
            vals = cluster_dist.loc[k].drop(index=k, errors="ignore")
            vals = vals[vals > 0]

            if len(vals) == 0:
                mean_dists[k] = np.nan
            else:
                mean_dists[k] = float(vals.mean())

        dist_series = pd.Series(mean_dists)

        if dist_series.notna().sum() > 1:
            centrality_values = 1 - robust_minmax(
                dist_series.fillna(dist_series.median()).values,
                0.05,
                0.95,
            )
        else:
            centrality_values = np.ones(len(keys))

        centrality_map = dict(zip(keys, centrality_values))

        rows = []

        for k in keys:
            profile = S.loc[k]

            mean_score = float(profile.mean())
            diversity_score = float((profile > 0.25).mean())
            deficit_score = float((profile < 0.35).mean())
            strong_count = int((profile >= 0.55).sum())
            weak_count = int((profile < 0.35).sum())

            center_score = (
                0.30 * pop_norm_map.get(k, 0)
                + 0.35 * mean_score
                + 0.20 * diversity_score
                + 0.15 * centrality_map.get(k, 0)
            )

            periphery_score = (
                0.45 * deficit_score
                + 0.35 * (1 - centrality_map.get(k, 0))
                + 0.20 * (1 - pop_norm_map.get(k, 0))
            )

            rows.append(
                {
                    "cluster": cluster_name,
                    "town": town_names[k],
                    "town_key": k,
                    "population": pop_map[k],
                    "center_score": center_score,
                    "periphery_score": periphery_score,
                    "deficit_score": deficit_score,
                    "mean_sector_score": mean_score,
                    "diversity_score": diversity_score,
                    "centrality_score": centrality_map.get(k, 0),
                    "strong_sector_count": strong_count,
                    "weak_sector_count": weak_count,
                    "strongest_sectors": ", ".join(
                        [f"{s} ({v:.2f})" for s, v in profile.sort_values(ascending=False).head(3).items()]
                    ),
                    "weakest_sectors": ", ".join(
                        [f"{s} ({v:.2f})" for s, v in profile.sort_values(ascending=True).head(3).items()]
                    ),
                }
            )

        tmp = pd.DataFrame(rows)

        max_center = tmp["center_score"].max()

        roles = []

        for _, row in tmp.iterrows():
            if row["center_score"] == max_center:
                if row["deficit_score"] >= 0.50:
                    role = "условный локальный центр с инфраструктурными дефицитами"
                else:
                    role = "локальный центр"
            elif row["deficit_score"] >= 0.50 and row["periphery_score"] >= 0.55:
                role = "территория с выраженным инфраструктурным дефицитом"
            elif row["periphery_score"] >= 0.55:
                role = "периферийная территория"
            else:
                role = "зона тяготения / участник группы"

            roles.append(role)

        tmp["territory_role"] = roles
        all_rows.extend(tmp.to_dict("records"))

    return pd.DataFrame(all_rows)


# ============================================================
# 11. СОХРАНЕНИЕ
# ============================================================

def save_outputs(params: Dict, pack: Dict, results_df: pd.DataFrame, population: pd.DataFrame, distances: pd.DataFrame):
    labels = pack["labels"]
    features = pack["features"]
    sector_scores = pack["sector_scores"]
    complementarity = pack["complementarity"]
    soft_reciprocity = pack["soft_reciprocity"]
    affinity = pack["affinity"]
    mask = pack["mask"]
    metrics = pack["metrics"]

    cluster_df, member_df, edge_df = describe_clusters(
        labels=labels,
        population=population,
        sector_scores=sector_scores,
        distances=distances,
        complementarity=complementarity,
        soft_reciprocity=soft_reciprocity,
        affinity=affinity,
        mask=mask,
    )

    roles_df = compute_territory_roles(
        labels=labels,
        population=population,
        sector_scores=sector_scores,
        distances=distances,
    )

    labels_df = population[["town", "town_key", "population"]].copy()
    labels_df["cluster_label"] = labels
    labels_df["cluster_name"] = [
        "шум / не отнесено" if x == -1 else f"Агломерация {x + 1}"
        for x in labels
    ]

    best_info = {
        "region": REGION_NAME,
        "model": params.get("model_type", "unknown"),
        **params,
        "objective_score": objective_score(metrics, get_distance_stats(distances)),
        **metrics,
    }

    best_info_df = pd.DataFrame([best_info])

    save_model_plots(
        results_df=results_df,
        best_info_df=best_info_df,
        cluster_df=cluster_df,
        edge_df=edge_df,
        output_dir=OUTPUT_DIR,
    )

    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    best_xlsx = OUTPUT_DIR / "best_model_result.xlsx"

    with pd.ExcelWriter(best_xlsx, engine="openpyxl") as writer:
        best_info_df.to_excel(writer, sheet_name="best_model", index=False)
        cluster_df.to_excel(writer, sheet_name="agglomerations", index=False)
        member_df.to_excel(writer, sheet_name="members", index=False)
        roles_df.to_excel(writer, sheet_name="territory_roles", index=False)
        edge_df.to_excel(writer, sheet_name="used_edges", index=False)
        labels_df.to_excel(writer, sheet_name="labels", index=False)
        features.to_excel(writer, sheet_name="features", index=False)
        sector_scores.reset_index().to_excel(writer, sheet_name="sector_scores", index=False)

        if results_df is not None and len(results_df) > 0:
            export_trials = results_df.drop(columns=["labels"], errors="ignore")
            export_trials.head(TOP_TRIALS_TO_EXPORT).to_excel(writer, sheet_name="trials_top", index=False)

    best_info_df.to_csv(OUTPUT_DIR / "best_model_metrics.csv", index=False, encoding="utf-8-sig")
    cluster_df.to_csv(OUTPUT_DIR / "best_agglomerations.csv", index=False, encoding="utf-8-sig")
    member_df.to_csv(OUTPUT_DIR / "best_members.csv", index=False, encoding="utf-8-sig")
    roles_df.to_csv(OUTPUT_DIR / "territory_roles.csv", index=False, encoding="utf-8-sig")
    edge_df.to_csv(OUTPUT_DIR / "best_used_edges.csv", index=False, encoding="utf-8-sig")
    labels_df.to_csv(OUTPUT_DIR / "best_labels.csv", index=False, encoding="utf-8-sig")

    if results_df is not None and len(results_df) > 0:
        results_df.to_csv(OUTPUT_DIR / "all_trials_results.csv", index=False, encoding="utf-8-sig")

    print_header("ЛУЧШАЯ МОДЕЛЬ")
    print(best_info_df.T.to_string())

    print_header("АГЛОМЕРАЦИИ")
    print(cluster_df.to_string(index=False))

    print_header("РОЛИ ТЕРРИТОРИЙ")
    print(
        roles_df[
            ["cluster", "town", "territory_role", "center_score", "deficit_score"]
        ].to_string(index=False)
    )

    print_header("ФАЙЛЫ СОХРАНЕНЫ")
    print(f"Папка: {OUTPUT_DIR}")
    print("Главные файлы:")
    print(" - best_model_result.xlsx — итоговый Excel: модель, группы, роли, связи")
    print(" - best_model_metrics.csv — метрики лучшей модели")
    print(" - all_trials_results.csv — все попытки подбора параметров")
    print(" - best_agglomerations.csv — группы")
    print(" - territory_roles.csv — роли территорий")
    print(" - best_used_edges.csv — реально используемые связи")
    print(" - best_params.json — лучшие гиперпараметры")

# ============================================================
# 11.1. ГРАФИКИ ДЛЯ ВКР
# ============================================================

def setup_russian_plots():
    """
    Настройка matplotlib для русских подписей.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def save_barh(
    df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    path,
    value_fmt="{:.3f}",
    figsize=(10, 6),
):
    """
    Горизонтальная столбчатая диаграмма.
    """
    if df.empty:
        return

    plot_df = df.copy()

    plt.figure(figsize=figsize)
    plt.barh(plot_df[y_col].astype(str), plot_df[x_col])
    plt.gca().invert_yaxis()

    for i, value in enumerate(plot_df[x_col].values):
        if pd.notna(value):
            plt.text(
                value,
                i,
                " " + value_fmt.format(value),
                va="center",
                fontsize=9,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_model_plots(
    results_df: pd.DataFrame,
    best_info_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Сохраняет набор графиков для ВКР:
    1) топ-10 моделей по objective_score;
    2) динамика лучшего score по trials;
    3) размеры агломераций;
    4) среднее расстояние внутри агломераций;
    5) взаимодополняемость по агломерациям;
    6) сравнение ключевых метрик лучшей модели.
    """
    setup_russian_plots()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # 1. Топ-10 моделей по целевой функции
    # --------------------------------------------------------
    if results_df is not None and len(results_df) > 0 and "score" in results_df.columns:
        top_models = results_df.head(10).copy()
        top_models["model_name"] = [
            f"#{int(row['trial'])}: k={int(row['n_clusters'])}, "
            f"thr={row['threshold']:.1f}, knn={int(row['knn'])}"
            for _, row in top_models.iterrows()
        ]

        save_barh(
            df=top_models,
            x_col="score",
            y_col="model_name",
            title="Топ-10 вариантов модели по целевой функции",
            xlabel="Значение целевой функции",
            ylabel="Вариант модели",
            path=plots_dir / "01_top_10_models_score.png",
            value_fmt="{:.4f}",
            figsize=(12, 7),
        )

    # --------------------------------------------------------
    # 2. Динамика оптимизации Optuna
    # --------------------------------------------------------
    if results_df is not None and len(results_df) > 0 and {"trial", "score"}.issubset(results_df.columns):
        trial_df = results_df.copy()
        trial_df = trial_df.sort_values("trial")
        trial_df = trial_df[trial_df["score"] > -1e8].copy()

        if len(trial_df) > 0:
            trial_df["best_score_so_far"] = trial_df["score"].cummax()

            plt.figure(figsize=(12, 6))
            plt.plot(trial_df["trial"], trial_df["score"], alpha=0.35, label="Значение на попытке")
            plt.plot(trial_df["trial"], trial_df["best_score_so_far"], linewidth=2.5, label="Лучшее значение к текущей попытке")

            plt.title("Динамика подбора гиперпараметров Optuna")
            plt.xlabel("Номер попытки")
            plt.ylabel("Целевая функция")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "02_optuna_score_dynamics.png", dpi=300, bbox_inches="tight")
            plt.close()

    # --------------------------------------------------------
    # 3. Размеры агломераций
    # --------------------------------------------------------
    if cluster_df is not None and len(cluster_df) > 0 and {"cluster", "n_towns"}.issubset(cluster_df.columns):
        plot_df = cluster_df[["cluster", "n_towns"]].copy()
        plot_df = plot_df.sort_values("n_towns", ascending=False)

        save_barh(
            df=plot_df,
            x_col="n_towns",
            y_col="cluster",
            title="Размеры выделенных агломераций",
            xlabel="Количество территорий",
            ylabel="Агломерация",
            path=plots_dir / "03_agglomeration_sizes.png",
            value_fmt="{:.0f}",
            figsize=(10, 7),
        )

    # --------------------------------------------------------
    # 4. Среднее расстояние внутри агломераций
    # --------------------------------------------------------
    if cluster_df is not None and len(cluster_df) > 0 and {"cluster", "avg_pairwise_distance_km"}.issubset(cluster_df.columns):
        plot_df = cluster_df[["cluster", "avg_pairwise_distance_km"]].copy()
        plot_df = plot_df.sort_values("avg_pairwise_distance_km", ascending=False)

        save_barh(
            df=plot_df,
            x_col="avg_pairwise_distance_km",
            y_col="cluster",
            title="Среднее расстояние между территориями внутри агломераций",
            xlabel="Среднее расстояние, км",
            ylabel="Агломерация",
            path=plots_dir / "04_avg_pairwise_distance_by_agglomeration.png",
            value_fmt="{:.1f}",
            figsize=(11, 7),
        )

    # --------------------------------------------------------
    # 5. Взаимодополняемость по агломерациям
    # --------------------------------------------------------
    if cluster_df is not None and len(cluster_df) > 0 and {"cluster", "avg_internal_complementarity"}.issubset(cluster_df.columns):
        plot_df = cluster_df[["cluster", "avg_internal_complementarity"]].copy()
        plot_df = plot_df.sort_values("avg_internal_complementarity", ascending=False)

        save_barh(
            df=plot_df,
            x_col="avg_internal_complementarity",
            y_col="cluster",
            title="Средняя функциональная взаимодополняемость агломераций",
            xlabel="Взаимодополняемость",
            ylabel="Агломерация",
            path=plots_dir / "05_complementarity_by_agglomeration.png",
            value_fmt="{:.3f}",
            figsize=(11, 7),
        )

    # --------------------------------------------------------
    # 6. Ключевые метрики лучшей модели
    # --------------------------------------------------------
    if best_info_df is not None and len(best_info_df) > 0:
        row = best_info_df.iloc[0]

        metric_map = {
            "edge_coverage": "Плотность связей",
            "share_close_pairs": "Доля близких пар",
            "avg_complementarity": "Взаимодополняемость",
            "avg_soft_reciprocity": "Двусторонность",
            "balance_score": "Сбалансированность",
            "agglomeration_gain": "Прирост баланса",
            "avg_affinity": "Сила связи",
            "dominance_penalty": "Штраф доминирования",
        }

        metric_rows = []

        for col, label in metric_map.items():
            if col in row.index and pd.notna(row[col]):
                metric_rows.append(
                    {
                        "metric": label,
                        "value": float(row[col]),
                    }
                )

        metric_df = pd.DataFrame(metric_rows)

        if len(metric_df) > 0:
            metric_df = metric_df.sort_values("value", ascending=False)

            save_barh(
                df=metric_df,
                x_col="value",
                y_col="metric",
                title="Ключевые метрики лучшей модели",
                xlabel="Значение метрики",
                ylabel="Метрика",
                path=plots_dir / "06_best_model_key_metrics.png",
                value_fmt="{:.3f}",
                figsize=(11, 6),
            )

    # --------------------------------------------------------
    # 7. Связь расстояния и силы связи
    # --------------------------------------------------------
    if edge_df is not None and len(edge_df) > 0 and {"distance_km", "affinity"}.issubset(edge_df.columns):
        plot_df = edge_df.copy()
        plot_df = plot_df.dropna(subset=["distance_km", "affinity"])

        if len(plot_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(plot_df["distance_km"], plot_df["affinity"], alpha=0.75)

            plt.title("Зависимость силы связи от расстояния")
            plt.xlabel("Расстояние между территориями, км")
            plt.ylabel("Сила полицентрической связи")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "07_distance_vs_affinity.png", dpi=300, bbox_inches="tight")
            plt.close()

    # --------------------------------------------------------
    # 8. Связь расстояния и взаимодополняемости
    # --------------------------------------------------------
    if edge_df is not None and len(edge_df) > 0 and {"distance_km", "complementarity"}.issubset(edge_df.columns):
        plot_df = edge_df.copy()
        plot_df = plot_df.dropna(subset=["distance_km", "complementarity"])

        if len(plot_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(plot_df["distance_km"], plot_df["complementarity"], alpha=0.75)

            plt.title("Зависимость взаимодополняемости от расстояния")
            plt.xlabel("Расстояние между территориями, км")
            plt.ylabel("Функциональная взаимодополняемость")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "08_distance_vs_complementarity.png", dpi=300, bbox_inches="tight")
            plt.close()

    print(f"\nГрафики сохранены в папку: {plots_dir}")

# ============================================================
# 12. MAIN
# ============================================================

# ============================================================
# 13. ФУНКЦИЯ ДЛЯ STREAMLIT-ПРИЛОЖЕНИЯ
# ============================================================

def run_pipeline(
    population_path,
    results_path,
    distances_path,
    output_dir=None,
    n_trials=None,
    region_name="Нижегородская область",
):
    """
    Запуск полного пайплайна модели из Streamlit или другого Python-кода.

    Параметры:
    population_path  — путь к Excel-файлу с населением территорий;
    results_path     — путь к Excel-файлу с инфраструктурными объектами;
    distances_path   — путь к Excel-файлу с матрицей дорожных расстояний;
    output_dir       — папка, куда будут сохранены результаты модели;
    n_trials         — число попыток Optuna;
    region_name      — название региона для итоговых метрик.

    Возвращает словарь с основными путями и кратким результатом.
    """
    global POPULATION_FILE, RESULTS_FILE, DISTANCES_FILE, OUTPUT_DIR, N_TRIALS, REGION_NAME

    POPULATION_FILE = Path(population_path)
    RESULTS_FILE = Path(results_path)
    DISTANCES_FILE = Path(distances_path)

    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if n_trials is not None:
        N_TRIALS = int(n_trials)

    REGION_NAME = str(region_name)

    start_time = time.time()

    population = load_population(POPULATION_FILE)
    results = load_results(RESULTS_FILE)
    distances = load_distances(DISTANCES_FILE)

    population, results, distances = align_all_data(
        population=population,
        results=results,
        distances=distances,
    )

    best_params, best_pack, results_df = run_optuna_search(
        population=population,
        results=results,
        distances=distances,
    )

    save_outputs(
        params=best_params,
        pack=best_pack,
        results_df=results_df,
        population=population,
        distances=distances,
    )

    elapsed = time.time() - start_time

    return {
        "status": "ok",
        "region": REGION_NAME,
        "elapsed_seconds": elapsed,
        "output_dir": str(OUTPUT_DIR),
        "best_params_path": str(OUTPUT_DIR / "best_params.json"),
        "best_model_metrics_path": str(OUTPUT_DIR / "best_model_metrics.csv"),
        "best_agglomerations_path": str(OUTPUT_DIR / "best_agglomerations.csv"),
        "territory_roles_path": str(OUTPUT_DIR / "territory_roles.csv"),
        "best_used_edges_path": str(OUTPUT_DIR / "best_used_edges.csv"),
    }



def main():
    start_time = time.time()

    print_header("ПОЛИЦЕНТРИЧЕСКАЯ МОДЕЛЬ СЕЛЬСКИХ АГЛОМЕРАЦИЙ")
    print(f"Регион: {REGION_NAME}")
    print(f"N_TRIALS: {N_TRIALS}")
    print("Модель: SpectralClustering на предрассчитанной матрице полицентрической связи")

    for p in [POPULATION_FILE, RESULTS_FILE, DISTANCES_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл: {p}")

    print_header("ЗАГРУЗКА ДАННЫХ")

    population = load_population(POPULATION_FILE)
    results = load_results(RESULTS_FILE)
    distances = load_distances(DISTANCES_FILE)

    population, results, distances = align_all_data(
        population=population,
        results=results,
        distances=distances,
    )

    best_params, best_pack, results_df = run_optuna_search(
        population=population,
        results=results,
        distances=distances,
    )

    save_outputs(
        params=best_params,
        pack=best_pack,
        results_df=results_df,
        population=population,
        distances=distances,
    )

    elapsed = time.time() - start_time

    print_header("ГОТОВО")
    print(f"Время выполнения: {elapsed / 60:.2f} минут")


if __name__ == "__main__":
    main()
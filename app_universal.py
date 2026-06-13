# ============================================================
# Streamlit-приложение для ВКР
# Полицентрическая модель сельской агломерации
# ============================================================

import json
import math
import re
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Расчетная модель должна лежать рядом с app.py
import model_pipeline_universal as model_pipeline


# ============================================================
# 0. ОБЩИЕ НАСТРОЙКИ
# ============================================================

APP_TITLE = "Полицентрическая модель сельской агломерации"
APP_SUBTITLE = "для оценки факторов мотивации проживания на сельских территориях"
REGION_NAME = "Кировская область"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "model_outputs"

st.set_page_config(
    page_title="Полицентрическая модель сельской агломерации",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 1. СТИЛИ
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
            /* ============================================================
               СВЕТЛАЯ ТЕМА
               ============================================================ */

            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"] {
                background: #ffffff !important;
                color: #111827 !important;
            }

            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0) !important;
            }

            [data-testid="stToolbar"] {
                display: none !important;
            }

            [data-testid="stSidebar"] {
                display: none !important;
            }

            [data-testid="collapsedControl"] {
                display: none !important;
            }

            .main .block-container,
            .block-container {
                padding-top: 2.0rem !important;
                padding-bottom: 3.0rem !important;
                max-width: 1400px !important;
                background: #ffffff !important;
                color: #111827 !important;
            }

            html,
            body,
            [class*="css"] {
                font-family: "Segoe UI", sans-serif !important;
            }

            p,
            span,
            div,
            label,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            [data-testid="stMarkdownContainer"] {
                color: #111827 !important;
            }

            /* ============================================================
               ГРАДИЕНТНАЯ ШАПКА
               ============================================================ */

            .gradient-header {
                padding: 34px 38px !important;
                border-radius: 24px !important;
                background: linear-gradient(90deg, #bf003d 0%, #84205d 48%, #243875 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 18px 38px rgba(132, 32, 93, 0.22) !important;
                margin-bottom: 24px !important;
            }

            .gradient-header h1 {
                margin: 0 !important;
                font-size: 36px !important;
                line-height: 1.18 !important;
                font-weight: 850 !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                max-width: 1220px !important;
            }

            .gradient-header h1,
            .gradient-header h1 * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            /* ============================================================
               ГРАДИЕНТНЫЙ ТЕКСТ ДЛЯ ЗАГОЛОВКОВ
               ============================================================ */

            .gradient-text {
                background: linear-gradient(90deg, #bf003d 0%, #84205d 45%, #243875 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                color: transparent !important;
                font-weight: 850 !important;
            }

            .section-title {
                font-size: 28px !important;
                font-weight: 850 !important;
                margin: 28px 0 16px 0 !important;
                background: linear-gradient(90deg, #bf003d 0%, #84205d 45%, #243875 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                color: transparent !important;
            }

            .upload-title {
                font-weight: 850 !important;
                font-size: 18px !important;
                margin-bottom: 6px !important;
                background: linear-gradient(90deg, #bf003d 0%, #84205d 45%, #243875 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                color: transparent !important;
            }

            /* ============================================================
               ИНФОРМАЦИОННЫЕ ПОЛОСКИ
               ============================================================ */

            .info-strip {
                background: #ffffff !important;
                border: 1px solid #f0d3dd !important;
                border-left: 7px solid #bf003d !important;
                border-radius: 18px !important;
                padding: 18px 22px !important;
                margin: 12px 0 !important;
                box-shadow: 0 8px 24px rgba(0,0,0,0.045) !important;
                font-size: 18px !important;
                line-height: 1.55 !important;
                color: #111827 !important;
            }

            .info-strip,
            .info-strip p,
            .info-strip span,
            .info-strip div,
            .info-strip strong {
                color: #111827 !important;
            }

            .rose-note {
                background: #fff1f5 !important;
                border: 1px solid #f4c6d4 !important;
                border-left: 7px solid #bf003d !important;
                border-radius: 14px !important;
                padding: 16px 20px !important;
                margin: 18px 0 !important;
                color: #111827 !important;
                font-size: 17px !important;
                line-height: 1.45 !important;
                box-shadow: 0 8px 20px rgba(191, 0, 61, 0.06) !important;
            }

            .rose-note,
            .rose-note p,
            .rose-note span,
            .rose-note div,
            .rose-note strong {
                color: #111827 !important;
            }

            /* ============================================================
               БЛОКИ ЗАГРУЗКИ
               ============================================================ */

            .upload-card {
                background: #ffffff !important;
                border: 1px solid #ececf1 !important;
                border-radius: 20px !important;
                padding: 18px 18px 12px 18px !important;
                box-shadow: 0 10px 26px rgba(0,0,0,0.055) !important;
                min-height: 144px !important;
                color: #111827 !important;
            }

            .upload-card,
            .upload-card p,
            .upload-card div,
            .upload-card span,
            .upload-card strong {
                color: #111827 !important;
            }

            .upload-help {
                font-size: 14px !important;
                color: #475467 !important;
                line-height: 1.38 !important;
                margin-bottom: 10px !important;
            }

            .upload-help strong {
                color: #111827 !important;
            }

            .file-ok {
                margin-top: 6px !important;
                font-size: 14px !important;
                color: #0f7a3b !important;
                font-weight: 600 !important;
            }

            .file-bad {
                margin-top: 6px !important;
                font-size: 14px !important;
                color: #b42318 !important;
                font-weight: 600 !important;
            }

            /* ============================================================
               ЗАГРУЗЧИКИ ФАЙЛОВ
               ============================================================ */

            .stFileUploader {
                background: transparent !important;
                color: #111827 !important;
            }

            .stFileUploader label {
                color: #111827 !important;
                font-weight: 800 !important;
                font-size: 16px !important;
            }

            [data-testid="stFileUploader"] {
                background: transparent !important;
                color: #111827 !important;
            }

            [data-testid="stFileUploader"] * {
                color: #111827 !important;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: #ffffff !important;
                border: 1px dashed #c7cbd4 !important;
                border-radius: 15px !important;
                color: #111827 !important;
            }

            [data-testid="stFileUploaderDropzone"] * {
                color: #111827 !important;
            }

            [data-testid="stFileUploaderDropzone"] button {
                background: #ffffff !important;
                color: #111827 !important;
                border: 1px solid #d0d5dd !important;
                border-radius: 10px !important;
                box-shadow: none !important;
                font-weight: 700 !important;
            }

            [data-testid="stFileUploaderDropzone"] button * {
                color: #111827 !important;
            }

            [data-testid="stFileUploaderDropzone"] button:hover {
                background: #f9fafb !important;
                color: #111827 !important;
                border: 1px solid #bf003d !important;
            }

            [data-testid="stFileUploaderDropzone"] button:hover * {
                color: #111827 !important;
            }

            [data-testid="stFileUploaderDropzone"] small,
            [data-testid="stFileUploaderDropzone"] span,
            [data-testid="stFileUploaderDropzone"] div {
                color: #475467 !important;
            }

            [data-testid="stFileUploaderFile"] {
                background: #f8fafc !important;
                border: 1px solid #e5e7eb !important;
                border-radius: 12px !important;
                color: #111827 !important;
            }

            [data-testid="stFileUploaderFile"] * {
                color: #111827 !important;
            }

            /* ============================================================
               БЛОК ОЖИДАНИЯ
               ============================================================ */

            .run-box {
                background: #fff8fb !important;
                border: 1px solid #f1c9d6 !important;
                border-left: 7px solid #bf003d !important;
                border-radius: 18px !important;
                padding: 20px 22px !important;
                margin: 18px 0 16px 0 !important;
                font-size: 18px !important;
                line-height: 1.55 !important;
                color: #111827 !important;
            }

            .run-box,
            .run-box p,
            .run-box div,
            .run-box span,
            .run-box strong {
                color: #111827 !important;
            }

            /* ============================================================
               КАРТОЧКИ МЕТРИК
               ============================================================ */

            .metric-card {
                background: #ffffff !important;
                border: 1px solid #ececf1 !important;
                border-radius: 18px !important;
                padding: 18px 18px !important;
                box-shadow: 0 10px 26px rgba(0,0,0,0.055) !important;
                min-height: 118px !important;
                color: #111827 !important;
            }

            .metric-card,
            .metric-card p,
            .metric-card div,
            .metric-card span {
                color: #111827 !important;
            }

            .metric-label {
                color: #596273 !important;
                font-size: 14px !important;
                font-weight: 650 !important;
                margin-bottom: 8px !important;
            }

            .metric-value {
                color: #111827 !important;
                font-size: 28px !important;
                font-weight: 850 !important;
                line-height: 1.2 !important;
            }

            .metric-caption {
                color: #6b7280 !important;
                font-size: 13px !important;
                margin-top: 8px !important;
                line-height: 1.35 !important;
            }

            /* ============================================================
               ОБЩИЕ КАРТОЧКИ
               ============================================================ */

            .soft-card {
                background: #ffffff !important;
                border: 1px solid #ececf1 !important;
                border-radius: 18px !important;
                padding: 18px 20px !important;
                box-shadow: 0 10px 26px rgba(0,0,0,0.045) !important;
                margin-bottom: 14px !important;
                color: #111827 !important;
            }

            .soft-card,
            .soft-card p,
            .soft-card div,
            .soft-card span,
            .soft-card strong {
                color: #111827 !important;
            }

            .pill {
                display: inline-block !important;
                padding: 6px 11px !important;
                border-radius: 999px !important;
                background: #f7edf2 !important;
                color: #8c1749 !important;
                font-size: 13px !important;
                font-weight: 700 !important;
                margin: 3px 4px 3px 0 !important;
            }

            /* ============================================================
               КНОПКИ
               ============================================================ */

            .stButton > button {
                width: 100% !important;
                border: none !important;
                border-radius: 16px !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                height: 56px !important;
                background: linear-gradient(90deg, #bf003d 0%, #84205d 48%, #243875 100%) !important;
                box-shadow: 0 12px 24px rgba(191, 0, 61, 0.18) !important;
            }

            .stButton > button * {
                color: #ffffff !important;
            }

            .stButton > button:hover {
                border: none !important;
                color: #ffffff !important;
                filter: brightness(1.04) !important;
            }

            .stButton > button:hover * {
                color: #ffffff !important;
            }

            /* ============================================================
               EXPANDER – НАСТРОЙКИ РАСЧЕТА
               ============================================================ */

            [data-testid="stExpander"] {
                background: #ffffff !important;
                border: 1px solid #ececf1 !important;
                border-radius: 16px !important;
                box-shadow: 0 8px 20px rgba(0,0,0,0.035) !important;
                overflow: hidden !important;
            }

            [data-testid="stExpander"] details {
                background: #ffffff !important;
                color: #111827 !important;
            }

            [data-testid="stExpander"] summary {
                background: #ffffff !important;
                color: #111827 !important;
                font-weight: 800 !important;
                border-radius: 16px !important;
            }

            [data-testid="stExpander"] summary * {
                color: #111827 !important;
            }

            [data-testid="stExpander"] div,
            [data-testid="stExpander"] p,
            [data-testid="stExpander"] span {
                color: #111827 !important;
            }

            /* ============================================================
               SLIDER
               ============================================================ */

            [data-testid="stSlider"] {
                color: #111827 !important;
            }

            [data-testid="stSlider"] label,
            [data-testid="stSlider"] label *,
            [data-testid="stSlider"] div,
            [data-testid="stSlider"] span {
                color: #111827 !important;
            }

            [data-baseweb="slider"] div[role="slider"] {
                background-color: #bf003d !important;
                border-color: #bf003d !important;
                box-shadow: 0 0 0 3px rgba(191, 0, 61, 0.15) !important;
            }

            /* ============================================================
               TABS, ALERTS, DATAFRAME, INPUTS
               ============================================================ */

            button[role="tab"] {
                background: #ffffff !important;
                color: #111827 !important;
                border-radius: 12px 12px 0 0 !important;
                font-weight: 800 !important;
            }

            button[role="tab"] * {
                color: #111827 !important;
            }

            button[role="tab"][aria-selected="true"] {
                color: #bf003d !important;
                border-bottom: 3px solid #bf003d !important;
            }

            button[role="tab"][aria-selected="true"] * {
                color: #bf003d !important;
            }

            .stAlert {
                border-radius: 16px !important;
            }

            [data-testid="stAlert"] * {
                color: #111827 !important;
            }

            [data-testid="stDataFrame"] {
                color: #111827 !important;
            }

            [data-testid="stDataFrame"] * {
                color: inherit;
            }

            [data-baseweb="select"] * {
                color: #111827 !important;
            }

            input,
            textarea {
                color: #111827 !important;
                background: #ffffff !important;
            }


            /* ============================================================
               ФИНАЛЬНЫЕ ДОРАБОТКИ ИНТЕРФЕЙСА
               ============================================================ */

            .file-ok,
            .file-bad {
                border-radius: 14px !important;
                padding: 12px 16px !important;
                margin: 14px 0 20px 0 !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                line-height: 1.35 !important;
                box-shadow: 0 8px 18px rgba(0,0,0,0.035) !important;
            }

            .file-ok {
                background: #ecfdf3 !important;
                border: 1px solid #86efac !important;
                border-left: 6px solid #10b981 !important;
                color: #067647 !important;
            }

            .file-bad {
                background: #fff1f3 !important;
                border: 1px solid #fecdd3 !important;
                border-left: 6px solid #e11d48 !important;
                color: #b42318 !important;
            }

            [data-testid="stExpander"] {
                margin-top: 18px !important;
            }

            .explain-card {
                position: relative !important;
                background: #ffffff !important;
                border: 1px solid #ececf1 !important;
                border-radius: 18px !important;
                padding: 18px 22px 18px 24px !important;
                margin: 16px 0 18px 0 !important;
                box-shadow: 0 10px 26px rgba(0,0,0,0.045) !important;
                color: #111827 !important;
                overflow: hidden !important;
            }

            .explain-card::before {
                content: "" !important;
                position: absolute !important;
                left: 0 !important;
                top: 0 !important;
                bottom: 0 !important;
                width: 5px !important;
                background: linear-gradient(180deg, #bf003d 0%, #84205d 50%, #243875 100%) !important;
            }

            .explain-card p {
                margin: 0 0 10px 0 !important;
                line-height: 1.55 !important;
                color: #111827 !important;
            }

            .explain-card p:last-child {
                margin-bottom: 0 !important;
            }

            .small-note {
                color: #667085 !important;
                font-size: 14px !important;
                line-height: 1.5 !important;
                margin-top: 12px !important;
            }

            .small-note b,
            .small-note strong {
                color: #475467 !important;
            }

            .white-table-wrap {
                width: 100% !important;
                overflow-x: auto !important;
                border: 1px solid #ececf1 !important;
                border-radius: 16px !important;
                background: #ffffff !important;
                box-shadow: 0 10px 26px rgba(0,0,0,0.045) !important;
                margin-top: 12px !important;
            }

            table.white-table {
                width: 100% !important;
                border-collapse: collapse !important;
                background: #ffffff !important;
                color: #111827 !important;
                font-size: 14px !important;
            }

            table.white-table thead th {
                background: #fff1f5 !important;
                color: #8c1749 !important;
                font-weight: 800 !important;
                text-align: left !important;
                border-bottom: 1px solid #f4c6d4 !important;
                padding: 12px 12px !important;
                white-space: nowrap !important;
            }

            table.white-table tbody td {
                background: #ffffff !important;
                color: #111827 !important;
                border-bottom: 1px solid #f2f4f7 !important;
                padding: 11px 12px !important;
                vertical-align: top !important;
            }

            table.white-table tbody tr:hover td {
                background: #fff8fb !important;
            }

            [data-testid="stFileUploaderFile"] {
                background: #fff1f5 !important;
                border: 1px solid #f4c6d4 !important;
                border-radius: 12px !important;
                color: #111827 !important;
            }

            [data-testid="stFileUploaderFile"] * {
                color: #111827 !important;
            }

            [data-testid="stFileUploaderFile"] svg {
                color: #bf003d !important;
                fill: #bf003d !important;
            }

            [data-testid="stFileUploaderFile"] button {
                background: #ffffff !important;
                border: 1px solid #f4c6d4 !important;
                color: #bf003d !important;
            }

            /* selectbox в светлой теме */
            [data-baseweb="select"] > div {
                background: #ffffff !important;
                border: 1px solid #e5e7eb !important;
                border-radius: 12px !important;
                color: #111827 !important;
            }

            [data-baseweb="popover"],
            [data-baseweb="popover"] * {
                background: #ffffff !important;
                color: #111827 !important;
            }


            /* ============================================================
               ИКОНКА ЗАГРУЖЕННОГО ФАЙЛА
               ============================================================ */

            [data-testid="stFileUploaderFile"] {
                background: #fff1f5 !important;
                border: 1px solid #f4c6d4 !important;
                border-radius: 12px !important;
                color: #111827 !important;
            }

            [data-testid="stFileUploaderFile"] * {
                color: #111827 !important;
            }

            [data-testid="stFileUploaderFile"] > div:first-child,
            [data-testid="stFileUploaderFile"] div:has(svg) {
                background: #ffffff !important;
                border: 1px solid #f4c6d4 !important;
                border-radius: 10px !important;
            }

            [data-testid="stFileUploaderFile"] svg,
            [data-testid="stFileUploaderFile"] svg *,
            [data-testid="stFileUploaderFile"] path {
                color: #bf003d !important;
                fill: #bf003d !important;
                stroke: #bf003d !important;
            }

            [data-testid="stFileUploaderFile"] [data-testid="stIconMaterial"] {
                color: #bf003d !important;
            }


            /* ============================================================
               ФИНАЛЬНАЯ ЧИСТКА: ИКОНКИ, ПОЛОСКИ, ФАЙЛЫ
               ============================================================ */

            .info-strip,
            .explain-card,
            .run-box,
            .rose-note {
                border-left: none !important;
                position: relative !important;
            }

            .info-strip::before,
            .explain-card::before,
            .run-box::before,
            .rose-note::before {
                content: "" !important;
                position: absolute !important;
                left: 0 !important;
                top: 0 !important;
                bottom: 0 !important;
                width: 4px !important;
                background: linear-gradient(180deg, #bf003d 0%, #84205d 48%, #243875 100%) !important;
                border-radius: 18px 0 0 18px !important;
            }

            .file-ok,
            .file-bad {
                font-weight: 500 !important;
                margin: 18px 0 26px 0 !important;
                border-left-width: 5px !important;
            }

            [data-testid="stExpander"] {
                margin-top: 22px !important;
            }

            [data-testid="stFileUploaderFile"] {
                position: relative !important;
                background: #fff1f5 !important;
                border: 1px solid #f4c6d4 !important;
                border-radius: 12px !important;
                color: #111827 !important;
                padding-left: 12px !important;
            }

            [data-testid="stFileUploaderFile"] * {
                color: #111827 !important;
            }

            /* Полностью закрываем темную встроенную иконку файла */
            [data-testid="stFileUploaderFile"]::before {
                content: "" !important;
                position: absolute !important;
                left: 0 !important;
                top: 0 !important;
                bottom: 0 !important;
                width: 56px !important;
                background: #fff1f5 !important;
                border-radius: 12px 0 0 12px !important;
                z-index: 20 !important;
                pointer-events: none !important;
            }

            [data-testid="stFileUploaderFile"] > div:first-child,
            [data-testid="stFileUploaderFile"] div:has(svg) {
                background: transparent !important;
                border: none !important;
            }

            [data-testid="stFileUploaderFile"] svg:first-child,
            [data-testid="stFileUploaderFile"] [data-testid="stIconMaterial"] {
                display: none !important;
            }

            [data-testid="stFileUploaderFile"] button {
                position: relative !important;
                z-index: 50 !important;
                background: #ffffff !important;
                border: 1px solid #f4c6d4 !important;
                color: #bf003d !important;
                border-radius: 999px !important;
            }

            [data-testid="stFileUploaderFile"] button * {
                color: #bf003d !important;
            }


        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 2. МЕЛКИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def fmt_number(value, digits: int = 2, suffix: str = "") -> str:
    """Красивый вывод чисел: 99, а не 99,00; 0,13, а не 0,130."""
    if value is None:
        return "–"
    try:
        if pd.isna(value):
            return "–"
    except Exception:
        pass
    try:
        val = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(val):
        return "–"

    if abs(val - round(val)) < 1e-9:
        text = str(int(round(val)))
    else:
        text = f"{val:.{digits}f}".rstrip("0").rstrip(".").replace(".", ",")
    return text + suffix

def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("е", "е")
    text = text.replace("–", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[«»\"']", "", text)
    return text.strip()


def read_csv_smart(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1251"]
    seps = [",", ";", "\t"]
    last_error = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception as e:
                last_error = e
    raise last_error


def read_table_smart(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return read_csv_smart(path)


def find_col(df: pd.DataFrame, variants: List[str], required: bool = False) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {c: normalize_text(c) for c in cols}
    for v in variants:
        nv = normalize_text(v)
        for c, nc in norm_map.items():
            if nc == nv or nv in nc:
                return c
    if required:
        raise ValueError(f"Не найдена колонка. Искал: {variants}. Есть: {cols}")
    return None


def safe_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return []
    for sep in [";", "|", ","]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]
    return [text]


def copy_uploaded_file(uploaded_file, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())



def clean_group_label(value) -> str:
    """Не допускает вида «Агломерация Агломерация 5»."""
    text = str(value).strip()
    if normalize_text(text).startswith("агломерация"):
        return text
    return f"Агломерация {text}"


def group_sort_key(value):
    text = str(value)
    m = re.search(r"\d+", text)
    if m:
        return int(m.group(0))
    return text


def clean_sector_name(value) -> str:
    text = str(value).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_sector_list(items: List[str]) -> List[str]:
    out = []
    for item in items:
        text = clean_sector_name(item)
        if text and text.lower() not in ["nan", "none", "null"]:
            out.append(text)
    return out


def translate_role(value) -> str:
    text = str(value).strip()
    n = normalize_text(text)
    if "center" in n or "центр" in n or "опор" in n:
        return "опорная территория"
    if "periphery" in n or "перифер" in n:
        return "периферийная территория"
    if "deficit" in n or "дефиц" in n:
        return "территория с инфраструктурным дефицитом"
    if "attraction" in n or "тягот" in n:
        return "зона тяготения"
    return text


def format_cell_value(value, col_name: str = "") -> str:
    if value is None:
        return "–"
    try:
        if pd.isna(value):
            return "–"
    except Exception:
        pass

    col_norm = normalize_text(col_name)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if "population" in col_norm or "насел" in col_norm:
            return fmt_number(value, 0)
        return fmt_number(value, 3)

    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return "–"

    # Красиво выводим списки сфер.
    if any(x in col_norm for x in ["sector", "сфер"]):
        return ", ".join(clean_sector_list(safe_list(text)))

    if "role" in col_norm or "роль" in col_norm:
        return translate_role(text)

    return text


def dataframe_to_white_html(df: pd.DataFrame) -> str:
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].apply(lambda x: format_cell_value(x, str(col)))
    html = safe.to_html(index=False, escape=True, classes="white-table")
    return f'<div class="white-table-wrap">{html}</div>'


def get_population_sum(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    pop_col = find_col(df, ["population", "население", "численность"], required=False)
    if not pop_col:
        return None
    vals = pd.to_numeric(df[pop_col], errors="coerce")
    if vals.notna().any():
        return float(vals.sum())
    return None


# Координаты административных центров территорий.
# Используются для демонстрационной карты. Для нового региона можно расширить словарь.
REGION_COORDS = {
    "володарск": (56.226, 43.186),
    "семенов": (56.789, 44.490),
    "кулебаки": (55.430, 42.512),
    "лысково": (56.032, 45.042),
    "шахунья": (57.676, 46.612),
    "урень": (57.464, 45.785),
    "сергач": (55.520, 45.481),
    "лукоянов": (55.028, 44.493),
    "починки": (54.700, 44.867),
    "шатки": (55.188, 44.125),
    "ардатов": (55.239, 43.096),
    "навашино": (55.543, 42.200),
    "чкаловск": (56.765, 43.242),
    "дальнее константиново": (55.810, 44.090),
    "пильна": (55.555, 45.921),
    "красные баки": (57.131, 45.159),
    "сосновское": (55.805, 43.165),
    "ковернино": (57.127, 43.817),
    "первомайск": (54.868, 43.802),
    "вача": (55.803, 42.770),
    "перевоз": (55.596, 44.545),
    "воскресенское": (56.838, 45.430),
    "сокольское": (57.141, 43.160),
    "тоншаево": (57.736, 47.012),
    "бутурлино": (55.566, 44.897),
    "княгинино": (55.820, 45.033),
    "большое мурашкино": (55.783, 44.770),
    "варнавино": (57.403, 45.091),
    "тонкино": (57.372, 46.463),
    "дивеево": (55.043, 43.245),
    "сеченово": (55.224, 45.892),
    "вад": (55.530, 44.210),
    "большое болдино": (55.005, 45.314),
    "гагино": (55.230, 45.030),
    "спасское": (55.860, 45.700),
    "уразовка": (55.401, 45.618),
    "ветлуга": (57.856, 45.781),
    "вознесенское": (54.890, 42.760),
    "шаранга": (57.179, 46.539),

    # Кировская область
    "вятские поляны": (56.228, 51.061),
    "котельнич": (58.304, 48.347),
    "слободской": (58.731, 50.177),
    "арбаж": (57.681, 48.306),
    "афанасьево": (58.840, 53.250),
    "богородское": (57.827, 50.748),
    "кирс": (59.339, 52.244),
    "кикнур": (57.301, 47.201),
    "лебяжье": (57.412, 49.515),
    "луза": (60.629, 47.252),
    "мураши": (59.395, 48.963),
    "нема": (57.506, 50.501),
    "опарино": (59.851, 48.277),
    "пижанка": (57.461, 48.543),
    "санчурск": (56.941, 47.249),
    "свеча": (58.278, 47.516),
    "фаленки": (58.361, 51.594),
    "уни": (57.752, 51.491),
    "белая холуница": (58.842, 50.846),
    "верхошижемье": (57.993, 49.105),
    "даровской": (58.770, 47.957),
    "зуевка": (58.405, 51.133),
    "кильмезь": (56.946, 51.066),
    "кирово-чепецк": (58.553, 50.031),
    "кумены": (58.108, 49.916),
    "малмыж": (56.525, 50.678),
    "нагорск": (59.316, 50.801),
    "нолинск": (57.559, 49.935),
    "омутнинск": (58.670, 52.189),
    "оричи": (58.403, 49.057),
    "орлов": (58.539, 48.891),
    "подосиновец": (60.277, 47.065),
    "советск": (57.584, 48.959),
    "суна": (57.833, 50.058),
    "тужа": (57.606, 47.936),
    "уржум": (57.114, 49.999),
    "ленинское": (58.316, 47.088),
    "юрья": (59.044, 49.279),
    "яранск": (57.303, 47.886),
}

# алиасы для карты: район/округ -> центр территории
REGION_COORD_ALIASES = {
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


# Дополнительные правила сопоставления названий территорий с координатами.
# В результатах модели есть town_key – именно его лучше использовать для карты.
def coord_key_from_row(row, tcol: str, key_col: Optional[str] = None) -> str:
    raw_key = ""
    if key_col is not None and key_col in row.index:
        raw_key = normalize_text(row.get(key_col, ""))
        if raw_key in REGION_COORDS:
            return raw_key

    text = normalize_text(row.get(tcol, ""))

    if text in REGION_COORD_ALIASES:
        return REGION_COORD_ALIASES[text]

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
    simplified = text
    for r in replacements:
        simplified = simplified.replace(r, "")
    simplified = simplified.strip()

    if simplified in aliases:
        return aliases[simplified]
    return simplified


# ============================================================
# 3. ПРОВЕРКА ФАЙЛОВ ДО ЗАПУСКА
# ============================================================

def inspect_population_file(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Файл не загружен"
    name = uploaded_file.name.lower()
    if "population" in name or "насел" in name:
        return True, "Файл похож на таблицу населения"
    try:
        df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        cols = [normalize_text(c) for c in df.columns]
        has_name = any(any(x in c for x in ["насел", "округ", "территор", "пункт", "municipality", "town"]) for c in cols)
        has_pop = any(any(x in c for x in ["население", "числен", "population"]) for c in cols)
        if has_name and has_pop:
            return True, "Файл похож на таблицу населения"
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return False, "В этот блок нужен файл с населением территорий, например population.xlsx"


def inspect_results_file(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Файл не загружен"
    name = uploaded_file.name.lower()
    if "results" in name or "object" in name or "инфра" in name:
        return True, "Файл похож на таблицу инфраструктурных объектов"
    try:
        df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        cols = [normalize_text(c) for c in df.columns]
        has_place = any(any(x in c for x in ["округ", "территор", "город", "town", "city"]) for c in cols)
        has_type = any(any(x in c for x in ["тип", "катег", "рубрика", "type", "category"]) for c in cols)
        has_rating = any(any(x in c for x in ["рейтинг", "оцен", "rating"]) for c in cols)
        if has_place and has_type and has_rating:
            return True, "Файл похож на таблицу инфраструктурных объектов"
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return False, "В этот блок нужен файл с инфраструктурными объектами, например results.xlsx"


def inspect_distances_file(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Файл не загружен"
    name = uploaded_file.name.lower()
    if "distance" in name or "matrix" in name or "расст" in name:
        return True, "Файл похож на матрицу расстояний"
    try:
        df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        if df.shape[0] >= 5 and df.shape[1] >= 5:
            numeric_share = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").notna().mean().mean()
            if numeric_share > 0.45:
                return True, "Файл похож на матрицу расстояний"
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return False, "В этот блок нужна матрица дорожных расстояний, например distances_matrix.xlsx"


# ============================================================
# 4. ЗАПУСК МОДЕЛИ
# ============================================================

def run_model_from_uploads(population_file, results_file, distances_file, n_trials: int) -> Path:
    run_dir = BASE_DIR / "streamlit_run"
    input_dir = run_dir / "input"
    output_dir = run_dir / "model_outputs"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    population_path = input_dir / "population.xlsx"
    results_path = input_dir / "results.xlsx"
    distances_path = input_dir / "distances_matrix.xlsx"

    copy_uploaded_file(population_file, population_path)
    copy_uploaded_file(results_file, results_path)
    copy_uploaded_file(distances_file, distances_path)

    # Важно: run_pipeline должен принимать эти аргументы.
    model_pipeline.run_pipeline(
        population_path=population_path,
        results_path=results_path,
        distances_path=distances_path,
        output_dir=output_dir,
        n_trials=n_trials,
        region_name=REGION_NAME,
    )

    return output_dir


# ============================================================
# 5. ЗАГРУЗКА РЕЗУЛЬТАТОВ МОДЕЛИ
# ============================================================

def load_outputs(output_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {}
    names = {
        "agglomerations": "best_agglomerations.csv",
        "members": "best_members.csv",
        "labels": "best_labels.csv",
        "metrics": "best_model_metrics.csv",
        "edges": "best_used_edges.csv",
        "roles": "territory_roles.csv",
    }
    for key, filename in names.items():
        path = output_dir / filename
        if path.exists():
            try:
                files[key] = read_csv_smart(path)
            except Exception:
                files[key] = pd.DataFrame()
        else:
            files[key] = pd.DataFrame()
    return files


def get_metric_value(metrics: pd.DataFrame, names: List[str]):
    if metrics is None or metrics.empty:
        return None

    # Вариант 1: одна строка, много колонок
    for name in names:
        col = find_col(metrics, [name], required=False)
        if col is not None:
            val = metrics[col].iloc[0]
            if pd.notna(val):
                return val

    # Вариант 2: две колонки metric/value
    metric_col = find_col(metrics, ["metric", "метрика", "показатель", "name"], required=False)
    value_col = find_col(metrics, ["value", "значение", "score"], required=False)
    if metric_col and value_col:
        temp = metrics.copy()
        temp["__m"] = temp[metric_col].apply(normalize_text)
        for name in names:
            n = normalize_text(name)
            row = temp[temp["__m"].str.contains(n, na=False)]
            if not row.empty:
                return row[value_col].iloc[0]

    return None


def detect_group_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["agglomeration", "cluster", "group", "label", "агломерация", "кластер", "группа"], required=False)


def detect_town_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["town", "territory", "name", "municipality", "территория", "округ", "населенный", "населенный"], required=False)


def compute_summary(outputs: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    aggl = outputs.get("agglomerations", pd.DataFrame())
    labels = outputs.get("labels", pd.DataFrame())
    members = outputs.get("members", pd.DataFrame())
    metrics = outputs.get("metrics", pd.DataFrame())

    group_count = None
    territory_count = None

    for df in [labels, members, aggl]:
        if df is not None and not df.empty:
            gcol = detect_group_col(df)
            tcol = detect_town_col(df)
            if group_count is None and gcol:
                group_count = int(df[gcol].nunique())
            if territory_count is None and tcol:
                territory_count = int(df[tcol].nunique())

    if group_count is None:
        group_count = get_metric_value(metrics, ["n_clusters", "clusters", "agglomerations", "количество кластеров"])
    if territory_count is None:
        territory_count = get_metric_value(metrics, ["n_towns", "territories", "objects", "количество территорий"])

    avg_distance = get_metric_value(
        metrics,
        ["avg_pairwise_distance", "avg_distance", "mean_distance", "среднее расстояние"],
    )
    if avg_distance is None and aggl is not None and not aggl.empty:
        col = find_col(aggl, ["avg_pairwise_distance", "avg_distance", "mean_distance", "среднее расстояние"], required=False)
        if col:
            avg_distance = pd.to_numeric(aggl[col], errors="coerce").mean()

    avg_comp = get_metric_value(
        metrics,
        ["avg_complementarity", "complementarity", "средняя взаимодополняемость", "взаимодополняемость"],
    )
    if avg_comp is None and aggl is not None and not aggl.empty:
        col = find_col(aggl, ["avg_complementarity", "complementarity", "взаимодополняемость"], required=False)
        if col:
            avg_comp = pd.to_numeric(aggl[col], errors="coerce").mean()

    disconnected = get_metric_value(metrics, ["disconnected_clusters", "разорванные"])
    if disconnected is None:
        disconnected = 0

    singleton = None
    if labels is not None and not labels.empty:
        gcol = detect_group_col(labels)
        if gcol:
            counts = labels[gcol].value_counts()
            singleton = int((counts == 1).sum())
    if singleton is None:
        singleton = get_metric_value(metrics, ["singletons", "singleton_clusters", "одиночные"])
    if singleton is None:
        singleton = 0

    return {
        "group_count": int(group_count) if group_count is not None and not pd.isna(group_count) else None,
        "territory_count": int(territory_count) if territory_count is not None and not pd.isna(territory_count) else None,
        "avg_distance": avg_distance,
        "avg_complementarity": avg_comp,
        "disconnected_clusters": int(float(disconnected)) if disconnected is not None and not pd.isna(disconnected) else 0,
        "singleton_clusters": int(float(singleton)) if singleton is not None and not pd.isna(singleton) else 0,
    }


# ============================================================
# 6. ОТОБРАЖЕНИЕ ИНТЕРФЕЙСА
# ============================================================

def render_header() -> None:
    st.markdown(
        """
        <div class="gradient-header">
            <h1>Полицентрическая модель сельской агломерации для оценки факторов мотивации проживания на сельских территориях</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_block():
    st.markdown('<div class="section-title">Загрузка данных</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="upload-title">1. Данные о населении территорий</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-help">Загрузите таблицу с названиями округов и численностью населения.</div>', unsafe_allow_html=True)
        population_file = st.file_uploader(
            "Файл населения",
            type=["xlsx", "xls", "csv"],
            key="population_file",
            label_visibility="collapsed",
        )
        ok, msg = inspect_population_file(population_file)
        if population_file is not None:
            st.markdown(f'<div class="{"file-ok" if ok else "file-bad"}">{msg}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-title">2. Данные об инфраструктурных объектах</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-help">Загрузите таблицу с объектами, типами объектов, рейтингами и отзывами.</div>', unsafe_allow_html=True)
        results_file = st.file_uploader(
            "Файл инфраструктуры",
            type=["xlsx", "xls", "csv"],
            key="results_file",
            label_visibility="collapsed",
        )
        ok2, msg2 = inspect_results_file(results_file)
        if results_file is not None:
            st.markdown(f'<div class="{"file-ok" if ok2 else "file-bad"}">{msg2}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="upload-title">3. Матрица дорожных расстояний</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-help">Загрузите матрицу расстояний между территориями в километрах.</div>', unsafe_allow_html=True)
        distances_file = st.file_uploader(
            "Файл расстояний",
            type=["xlsx", "xls", "csv"],
            key="distances_file",
            label_visibility="collapsed",
        )
        ok3, msg3 = inspect_distances_file(distances_file)
        if distances_file is not None:
            st.markdown(f'<div class="{"file-ok" if ok3 else "file-bad"}">{msg3}</div>', unsafe_allow_html=True)

    return population_file, results_file, distances_file, (ok, ok2, ok3)


def render_waiting_text() -> None:
    st.markdown(
        """
        <div class="run-box">
            <p><b>Ожидайте, идет работа модели.</b></p>
            <p>Выполняется сопоставление территорий, расчет инфраструктурных показателей, построение матрицы полицентрической связи, подбор параметров и выделение агломерационных групп.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(summary: Dict[str, object]) -> None:
    st.markdown('<div class="section-title">Результаты моделирования</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        render_metric_card("Выделено агломераций", fmt_number(summary["group_count"], 0), "Количество пространственно-функциональных групп")
    with c2:
        render_metric_card("Территорий в модели", fmt_number(summary["territory_count"], 0), "Количество округов, вошедших в расчет")
    with c3:
        render_metric_card("Среднее расстояние внутри групп", fmt_number(summary["avg_distance"], 2, " км"), "Пространственная компактность агломераций")
    with c4:
        render_metric_card("Средняя взаимодополняемость", fmt_number(summary["avg_complementarity"], 3), "Функциональная связь территорий")
    with c5:
        render_metric_card("Разорванные кластеры", fmt_number(summary["disconnected_clusters"], 0), "Хороший результат – 0")
    with c6:
        render_metric_card("Одиночные кластеры", fmt_number(summary["singleton_clusters"], 0), "Хороший результат – 0")

    st.markdown('<div class="section-title">Как читать показатели</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="explain-card">
            <p><b>Территорий в модели</b> – количество округов, которые удалось сопоставить во всех загруженных файлах.</p>
            <p><b>Среднее расстояние внутри групп</b> – средняя дорожная удаленность территорий внутри выделенных агломераций.</p>
            <p><b>Средняя взаимодополняемость</b> – показатель от 0 до 1: чем он выше, тем лучше сильные сферы одних территорий закрывают дефициты других.</p>
            <p><b>Разорванные кластеры</b> – группы, внутри которых есть территории без связей с остальными участниками. Хороший результат – 0.</p>
            <p><b>Одиночные кластеры</b> – территории, которые модель выделила отдельно от остальных. Хороший результат – 0.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. СПРАВКА ПО АГЛОМЕРАЦИИ
# ============================================================

def render_agglomeration_details(outputs: Dict[str, pd.DataFrame]) -> None:
    aggl = outputs.get("agglomerations", pd.DataFrame())
    members = outputs.get("members", pd.DataFrame())
    roles = outputs.get("roles", pd.DataFrame())

    st.markdown('<div class="section-title">Справка по агломерации</div>', unsafe_allow_html=True)

    base_df = members if members is not None and not members.empty else roles
    if base_df is None or base_df.empty:
        st.markdown('<div class="rose-note">Таблица состава агломераций не найдена.</div>', unsafe_allow_html=True)
        return

    gcol = detect_group_col(base_df)
    tcol = detect_town_col(base_df)
    if not gcol or not tcol:
        st.markdown('<div class="rose-note">Не удалось определить колонки с номером агломерации и названием территории.</div>', unsafe_allow_html=True)
        st.markdown(dataframe_to_white_html(base_df), unsafe_allow_html=True)
        return

    groups = sorted(base_df[gcol].dropna().unique().tolist(), key=group_sort_key)
    selected = st.selectbox("Выберите агломерацию", groups, format_func=clean_group_label)

    selected_members = base_df[base_df[gcol] == selected].copy()

    # Роли берем из territory_roles, если она есть.
    selected_roles = pd.DataFrame()
    if roles is not None and not roles.empty:
        rgcol = detect_group_col(roles)
        rtcol = detect_town_col(roles)
        if rgcol and rtcol:
            selected_roles = roles[roles[rgcol] == selected].copy()

    towns = selected_members[tcol].dropna().astype(str).tolist()

    role_col = None
    if selected_roles is not None and not selected_roles.empty:
        role_col = find_col(selected_roles, ["role", "роль", "status", "тип", "territory_role"], required=False)

    def towns_by_role(keywords: List[str]) -> List[str]:
        if selected_roles is None or selected_roles.empty or role_col is None:
            return []
        rtcol = detect_town_col(selected_roles)
        temp = selected_roles.copy()
        temp["__role"] = temp[role_col].apply(normalize_text)
        mask = False
        for kw in keywords:
            mask = mask | temp["__role"].str.contains(normalize_text(kw), na=False)
        return temp.loc[mask, rtcol].dropna().astype(str).tolist()

    support = towns_by_role(["центр", "опор", "local_center", "support"])
    periphery = towns_by_role(["перифер", "periphery"])
    deficit = towns_by_role(["дефицит", "deficit"])

    attraction = [t for t in towns if t not in set(support + periphery + deficit)]

    # Метрики группы из best_agglomerations
    group_row = pd.DataFrame()
    if aggl is not None and not aggl.empty:
        agcol = detect_group_col(aggl)
        if agcol:
            group_row = aggl[aggl[agcol] == selected].copy()

    def group_value(names: List[str]):
        if group_row is None or group_row.empty:
            return None
        col = find_col(group_row, names, required=False)
        if col:
            return group_row[col].iloc[0]
        return None

    avg_dist = group_value(["avg_pairwise_distance", "avg_distance", "mean_distance", "среднее расстояние"])
    max_dist = group_value(["max_pairwise_distance", "max_distance", "максимальное расстояние"])
    avg_comp = group_value(["avg_complementarity", "complementarity", "взаимодополняемость"])
    edge_cov = group_value(["edge_coverage", "coverage", "доля связей"])
    reciprocity = group_value(["avg_soft_reciprocity", "reciprocity", "двусторон"])

    population = get_population_sum(selected_roles)
    if population is None:
        population = get_population_sum(selected_members)
    if population is None:
        raw_pop = group_value(["population_sum", "population_total", "population", "население"])
        try:
            population = float(raw_pop)
        except Exception:
            population = None

    strong_col = find_col(group_row, ["strongest_sectors", "strong", "strength", "сильные"], required=False) if group_row is not None and not group_row.empty else None
    weak_col = find_col(group_row, ["weakest_sectors", "weak", "deficit", "слабые", "дефицитные сферы"], required=False) if group_row is not None and not group_row.empty else None
    strong = clean_sector_list(safe_list(group_row[strong_col].iloc[0])) if strong_col else []
    weak = clean_sector_list(safe_list(group_row[weak_col].iloc[0])) if weak_col else []

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown(
            f"""
            <div class="explain-card">
                <p><b>Состав группы:</b> {', '.join(towns) if towns else '–'}</p>
                <p><b>Опорная территория:</b> {', '.join(support) if support else '–'}</p>
                <p><b>Зоны тяготения / участники группы:</b> {', '.join(attraction) if attraction else '–'}</p>
                <p><b>Периферийные территории:</b> {', '.join(periphery) if periphery else '–'}</p>
                <p><b>Территории с инфраструктурным дефицитом:</b> {', '.join(deficit) if deficit else '–'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="explain-card">
                <p><b>Среднее расстояние внутри группы:</b> {fmt_number(avg_dist, 2, ' км')}</p>
                <p><b>Максимальное расстояние внутри группы:</b> {fmt_number(max_dist, 2, ' км')}</p>
                <p><b>Средняя взаимодополняемость:</b> {fmt_number(avg_comp, 3)}</p>
                <p><b>Доля внутренних связей:</b> {fmt_number(edge_cov, 3)}</p>
                <p><b>Средняя двусторонность:</b> {fmt_number(reciprocity, 3)}</p>
                <p><b>Население группы:</b> {fmt_number(population, 0, ' чел.')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="explain-card">
            <p><b>Среднее расстояние внутри группы</b> показывает среднюю дорожную удаленность территорий друг от друга внутри выбранной агломерации.</p>
            <p><b>Максимальное расстояние внутри группы</b> показывает самую дальнюю пару территорий внутри группы.</p>
            <p><b>Средняя взаимодополняемость</b> принимает значения от 0 до 1 и отражает, насколько сильные сферы одних территорий закрывают дефициты других.</p>
            <p><b>Доля внутренних связей</b> показывает, какая часть возможных связей внутри группы была использована моделью.</p>
            <p><b>Средняя двусторонность</b> показывает, насколько взаимодействие территорий является взаимным, а не односторонним.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if strong or weak:
        strong_html = " ".join([f'<span class="pill">{x}</span>' for x in strong]) if strong else "–"
        weak_html = " ".join([f'<span class="pill">{x}</span>' for x in weak]) if weak else "–"
        st.markdown(
            f"""
            <div class="explain-card">
                <p><b>Сильные сферы:</b></p>
                <p>{strong_html}</p>
                <p><b>Слабые сферы:</b></p>
                <p>{weak_html}</p>
                <p class="small-note"><b>Как читать:</b> сильные сферы – направления, где средняя оценка группы выше. Слабые сферы – направления, которые выражены слабее. Число в скобках находится в диапазоне от 0 до 1: чем оно выше, тем сильнее выражена сфера.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if selected_roles is not None and not selected_roles.empty:
        table_df = selected_roles.copy()
        rename_cols = {
            "cluster": "Агломерация",
            "agglomeration": "Агломерация",
            "group": "Агломерация",
            "town": "Территория",
            "territory": "Территория",
            "population": "Население",
            "center_score": "Оценка опорности",
            "periphery_score": "Оценка периферийности",
            "deficit_score": "Оценка дефицита",
            "mean_sector_score": "Средний уровень сфер",
            "diversity_score": "Разнообразие сфер",
            "centrality_score": "Связность в группе",
            "strong_sector_count": "Количество сильных сфер",
            "weak_sector_count": "Количество слабых сфер",
            "strongest_sectors": "Сильные сферы",
            "weakest_sectors": "Слабые сферы",
            "territory_role": "Роль территории",
            "role": "Роль территории",
        }

        # town_key нужен для расчетов, но в итоговой демонстрационной таблице только мешает.
        for col in list(table_df.columns):
            if normalize_text(col) == "town_key":
                table_df = table_df.drop(columns=[col])

        table_df = table_df.rename(columns={c: rename_cols.get(c, c) for c in table_df.columns})
        wanted_order = [
            "Агломерация", "Территория", "Население", "Оценка опорности",
            "Оценка периферийности", "Оценка дефицита", "Средний уровень сфер",
            "Разнообразие сфер", "Связность в группе", "Количество сильных сфер",
            "Количество слабых сфер", "Сильные сферы", "Слабые сферы", "Роль территории",
        ]
        table_df = table_df[[c for c in wanted_order if c in table_df.columns] + [c for c in table_df.columns if c not in wanted_order]]

        st.markdown('<div class="section-title">Таблица территорий выбранной агломерации</div>', unsafe_allow_html=True)
        st.markdown(dataframe_to_white_html(table_df), unsafe_allow_html=True)
        st.markdown(
            """
            <div class="explain-card">
                <p><b>Агломерация</b> – номер группы, в которую вошла территория.</p>
                <p><b>Территория</b> – муниципальный или городской округ, участвующий в расчете.</p>
                <p><b>Население</b> – численность населения территории, использованная при расчете обеспеченности инфраструктурой.</p>
                <p><b>Оценка опорности</b> – насколько территория может выполнять роль опорной внутри группы с учетом уровня сфер и положения в связях.</p>
                <p><b>Оценка периферийности</b> – насколько территория слабее включена в структуру группы и может зависеть от соседних территорий.</p>
                <p><b>Оценка дефицита</b> – выраженность инфраструктурных слабых мест территории.</p>
                <p><b>Средний уровень сфер</b> – средняя оценка развития функциональных сфер территории.</p>
                <p><b>Разнообразие сфер</b> – насколько равномерно представлены разные направления инфраструктуры.</p>
                <p><b>Связность в группе</b> – положение территории внутри сети связей агломерации.</p>
                <p><b>Количество сильных сфер</b> – число направлений, где территория имеет относительно высокую оценку.</p>
                <p><b>Количество слабых сфер</b> – число направлений, где у территории заметен дефицит.</p>
                <p><b>Сильные сферы</b> – направления, которые являются преимуществами территории.</p>
                <p><b>Слабые сферы</b> – направления, которые требуют усиления.</p>
                <p><b>Роль территории</b> – итоговая интерпретация положения территории в группе: опорная, периферийная, зона тяготения или территория с дефицитом.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 8. КАРТА – ЭКСПЕРИМЕНТАЛЬНЫЕ ПОЛИГОНЫ ЧЕРЕЗ OSMnx
# ============================================================

def render_map_placeholder(outputs: Dict[str, pd.DataFrame]) -> None:
    st.markdown('<div class="section-title">Карта агломераций</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="explain-card">
            <p>Карта показывает пространственное расположение территорий. Цвет точки соответствует агломерации, а линии показывают связи между территориями, использованные моделью.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    roles = outputs.get("roles", pd.DataFrame())
    members = outputs.get("members", pd.DataFrame())
    edges = outputs.get("edges", pd.DataFrame())

    base_df = roles if roles is not None and not roles.empty else members
    if base_df is None or base_df.empty:
        st.markdown('<div class="rose-note">Нет таблицы территорий для построения карты.</div>', unsafe_allow_html=True)
        return

    gcol = detect_group_col(base_df)
    tcol = detect_town_col(base_df)
    key_col = find_col(base_df, ["town_key", "ключ", "key"], required=False)

    if not gcol or not tcol:
        st.markdown('<div class="rose-note">Не удалось определить колонки территории и агломерации для карты.</div>', unsafe_allow_html=True)
        return

    points = []
    missing = []
    for _, row in base_df[[c for c in [tcol, gcol, key_col] if c]].drop_duplicates().iterrows():
        town = str(row[tcol])
        group = clean_group_label(row[gcol])
        key = coord_key_from_row(row, tcol=tcol, key_col=key_col)
        coords = REGION_COORDS.get(key)
        if coords is None:
            missing.append(town)
            continue
        lat, lon = coords
        points.append({
            "town": town,
            "key": key,
            "group": group,
            "lat": float(lat),
            "lon": float(lon),
        })

    if not points:
        st.markdown(
            '<div class="rose-note">Для территорий не удалось подобрать координаты. Проверьте, что в результатах есть колонка town_key или стандартные названия округов.</div>',
            unsafe_allow_html=True,
        )
        return

    point_by_key = {p["key"]: p for p in points}

    lines = []
    if edges is not None and not edges.empty:
        from_col = find_col(edges, ["source", "from", "town_i", "i", "territory_i", "город 1", "территория 1"], required=False)
        to_col = find_col(edges, ["target", "to", "town_j", "j", "territory_j", "город 2", "территория 2"], required=False)
        if from_col and to_col:
            for _, row in edges.iterrows():
                a = coord_key_from_row(row, tcol=from_col, key_col=None)
                b = coord_key_from_row(row, tcol=to_col, key_col=None)
                if a in point_by_key and b in point_by_key and a != b:
                    pa = point_by_key[a]
                    pb = point_by_key[b]
                    lines.append({
                        "from": pa["town"],
                        "to": pb["town"],
                        "group": pa["group"],
                        "coords": [[pa["lat"], pa["lon"]], [pb["lat"], pb["lon"]]],
                    })

    # Если файл связей не распознался, соединяем территории внутри одной агломерации с опорной.
    if not lines:
        role_col = find_col(base_df, ["role", "роль", "status", "тип", "territory_role"], required=False)
        for group_value, group_df in base_df.groupby(gcol):
            group_points = []
            for _, row in group_df.iterrows():
                key = coord_key_from_row(row, tcol=tcol, key_col=key_col)
                if key in point_by_key:
                    group_points.append(point_by_key[key])
            if len(group_points) < 2:
                continue

            center_point = None
            if role_col:
                temp = group_df.copy()
                temp["__role"] = temp[role_col].apply(normalize_text)
                center_rows = temp[temp["__role"].str.contains("центр|опор|center|support", regex=True, na=False)]
                if not center_rows.empty:
                    center_key = coord_key_from_row(center_rows.iloc[0], tcol=tcol, key_col=key_col)
                    center_point = point_by_key.get(center_key)
            if center_point is None:
                center_point = group_points[0]

            for p in group_points:
                if p["key"] != center_point["key"]:
                    lines.append({
                        "from": center_point["town"],
                        "to": p["town"],
                        "group": center_point["group"],
                        "coords": [[center_point["lat"], center_point["lon"]], [p["lat"], p["lon"]]],
                    })

    palette = [
        "#bf003d", "#2563eb", "#16a34a", "#d97706", "#7a1f5c", "#0f766e",
        "#9333ea", "#dc2626", "#64748b", "#be185d", "#243875", "#ea580c",
    ]
    groups = sorted({p["group"] for p in points}, key=group_sort_key)
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    points_json = json.dumps(points, ensure_ascii=False)
    lines_json = json.dumps(lines, ensure_ascii=False)
    colors_json = json.dumps(color_map, ensure_ascii=False)

    center_lat = float(np.mean([p["lat"] for p in points]))
    center_lon = float(np.mean([p["lon"] for p in points]))

    html = f"""
    <div class="map-wrapper" style="position: relative; width: 100%; height: 650px; border-radius: 18px; overflow: hidden; border: 1px solid #ececf1; background: #ffffff;">
        <div id="map" style="width: 100%; height: 650px;"></div>
    </div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        .leaflet-control-attribution,
        .leaflet-control-container .leaflet-control-attribution,
        .leaflet-bottom.leaflet-right {{
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
        }}
    </style>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const points = {points_json};
        const lines = {lines_json};
        const colors = {colors_json};

        const map = L.map('map', {{attributionControl: false}}).setView([{center_lat}, {center_lon}], 7);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 18,
            attribution: ''
        }}).addTo(map);

        lines.forEach(function(line) {{
            const groupColor = colors[line.group] || '#667085';
            L.polyline(line.coords, {{
                color: groupColor,
                weight: 2,
                opacity: 0.32
            }}).bindTooltip(line.from + ' – ' + line.to).addTo(map);
        }});

        const bounds = [];
        points.forEach(function(p) {{
            const color = colors[p.group] || '#bf003d';
            bounds.push([p.lat, p.lon]);
            L.circleMarker([p.lat, p.lon], {{
                radius: 8,
                color: '#ffffff',
                weight: 2,
                fillColor: color,
                fillOpacity: 0.92
            }})
            .bindTooltip('<b>' + p.town + '</b><br>' + p.group)
            .addTo(map);
        }});

        if (bounds.length > 1) {{
            map.fitBounds(bounds, {{padding: [28, 28]}});
        }}

        const legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            const div = L.DomUtil.create('div', 'legend');
            div.style.background = 'white';
            div.style.padding = '10px 12px';
            div.style.borderRadius = '12px';
            div.style.boxShadow = '0 8px 20px rgba(0,0,0,0.12)';
            div.style.fontSize = '13px';
            div.style.lineHeight = '1.45';
            div.innerHTML = '<b>Агломерации</b><br>';
            Object.keys(colors).forEach(function(g) {{
                div.innerHTML += '<span style="display:inline-block;width:10px;height:10px;background:' + colors[g] + ';border-radius:50%;margin-right:6px;"></span>' + g + '<br>';
            }});
            return div;
        }};
        legend.addTo(map);
    </script>
    """
    components.html(html, height=650)


# ============================================================
# 9. ОСНОВНОЙ ЗАПУСК
# ============================================================

def main() -> None:
    inject_css()
    render_header()

    population_file, results_file, distances_file, checks = render_upload_block()

    with st.expander("Настройки расчета", expanded=False):
        n_trials = st.slider(
            "Количество итераций автоматического подбора параметров",
            min_value=50,
            max_value=1000,
            value=1000,
            step=50,
            help="Для быстрого теста можно поставить 50–100. Для финального результата лучше 1000.",
        )

    all_uploaded = (
        population_file is not None
        and results_file is not None
        and distances_file is not None
    )
    all_ok = all(checks)

    if not all_uploaded:
        st.markdown(
            """
            <div class="rose-note">
                Загрузите три входных файла, чтобы запустить модель.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not all_ok:
        st.error(
            "Проверьте, что каждый файл загружен в свой блок: "
            "население, инфраструктурные объекты, матрица расстояний."
        )
        return

    if st.button("Запустить модель"):
        render_waiting_text()

        try:
            with st.spinner("Модель выполняет расчет. Это может занять несколько минут."):
                output_dir = run_model_from_uploads(
                    population_file=population_file,
                    results_file=results_file,
                    distances_file=distances_file,
                    n_trials=n_trials,
                )

            st.session_state.output_dir = str(output_dir)
            st.success("Расчет завершен. Результаты сформированы.")

        except Exception as e:
            st.error("Во время расчета возникла ошибка.")
            st.code(str(e))

            with st.expander("Показать технические подробности"):
                st.code(traceback.format_exc())

            return

    if "output_dir" in st.session_state:
        output_dir = Path(st.session_state.output_dir)
        outputs = load_outputs(output_dir)
        summary = compute_summary(outputs)

        render_summary(summary)
        render_map_placeholder(outputs)
        render_agglomeration_details(outputs)


if __name__ == "__main__":
    main()

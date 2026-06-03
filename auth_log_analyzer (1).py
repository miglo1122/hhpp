"""
auth_log_analyzer.py — SOC-аналізатор журналів автентифікації
================================================================
Веб-додаток для виявлення підозрілої активності у журналах входу
користувачів. Реалізує шість детекторів загроз, інтерактивні
візуалізації та формування звіту.

Запуск:  streamlit run auth_log_analyzer.py

Автор:   Куліуш Д.О., гр. ННІ-4-24-203 Кб, ХНУВС, 2026
"""

from __future__ import annotations

import io
import ipaddress
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
#  Глобальні константи та палітра оформлення
# ============================================================
APP_TITLE = "Sentinel · Аналіз журналів автентифікації"
APP_ICON = "🛡️"

# Кольорова схема SOC-дашборду (єдине джерело правди для UI та графіків)
PALETTE = {
    "bg": "#0b1020",
    "surface": "#141a2e",
    "surface_2": "#1b2238",
    "border": "#26304d",
    "text": "#e8edf7",
    "muted": "#8b97b5",
    "accent": "#34e0c4",      # бірюзовий акцент
    "accent_dim": "#1c8f7d",
    "success": "#34d399",
    "failure": "#f87171",
    "grid": "rgba(139,151,181,0.12)",
}

# Кольори рівнів небезпеки
SEVERITY_COLORS = {
    "CRITICAL": "#ff4d6d",
    "HIGH": "#ff8a4c",
    "MEDIUM": "#ffd166",
    "LOW": "#5ad1ff",
}
SEVERITY_WEIGHT = {"CRITICAL": 10, "HIGH": 6, "MEDIUM": 3, "LOW": 1}
SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

RENAME_MAP = {
    "time": "timestamp", "date": "timestamp", "ts": "timestamp", "datetime": "timestamp",
    "user": "username", "login": "username", "account": "username", "user_name": "username",
    "src_ip": "ip_address", "source_ip": "ip_address", "ip": "ip_address", "client_ip": "ip_address",
    "status": "result", "outcome": "result", "event": "result",
    "svc": "service", "app": "service",
}
REQUIRED_COLS = ["timestamp", "username", "ip_address", "result"]


# ============================================================
#  Стан додатку
# ============================================================
@dataclass
class DetectorConfig:
    """Параметри роботи детекторів (керуються зі сторінки налаштувань)."""
    bf_threshold: int = 5        # поріг невдалих спроб для брутфорсу
    bf_window: int = 60          # вікно брутфорсу, секунди
    spray_min_users: int = 5     # мін. користувачів для розпилення паролів
    night_start: int = 0         # початок «нічних» годин
    night_end: int = 6           # кінець «нічних» годин
    multiip_threshold: int = 4   # мін. унікальних IP для одного користувача
    saf_window: int = 300        # success-after-failure вікно, секунди
    saf_min_failures: int = 3    # мін. попередніх невдач
    profile_eps: float = 0.01    # поріг рідкісності години для профілю

    def signature(self) -> tuple:
        """Хешований підпис конфігурації для кешу."""
        return (self.bf_threshold, self.bf_window, self.spray_min_users,
                self.night_start, self.night_end, self.multiip_threshold,
                self.saf_window, self.saf_min_failures, self.profile_eps)


@dataclass
class AppState:
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cfg: DetectorConfig = field(default_factory=DetectorConfig)
    source_name: str = ""


# ============================================================
#  Завантаження та нормалізація даних  (кешується)
# ============================================================
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Зводить довільну схему журналу до стандартного формату."""
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Відсутні обовʼязкові колонки: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["username"] = df["username"].astype(str).str.strip()
    df["result"] = df["result"].astype(str).str.upper().str.strip()
    df["result"] = np.where(
        df["result"].isin(["SUCCESS", "OK", "ALLOW", "ACCEPTED", "1", "TRUE"]),
        "SUCCESS", "FAILURE")

    # Векторизована валідація IP-адрес
    def _valid_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(str(value))
            return True
        except ValueError:
            return False

    df["ip_address"] = df["ip_address"].astype(str).str.strip()
    df = df[df["ip_address"].map(_valid_ip)]
    df = df.dropna(subset=["timestamp", "username", "ip_address"])

    if "service" not in df.columns:
        df["service"] = "unknown"
    df["service"] = df["service"].astype(str)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    return df


@st.cache_data(show_spinner=False, max_entries=4)
def load_log(raw: bytes, fmt: str) -> pd.DataFrame:
    """Зчитує байти файлу журналу та повертає нормалізований DataFrame.

    Кешується за вмістом файлу — повторні перерендери не перечитують дані.
    """
    if len(raw) > 80 * 1024 * 1024:
        raise ValueError("Файл перевищує 80 МБ — завеликий для обробки в памʼяті.")

    if fmt == "CSV":
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    else:  # JSON
        try:
            df = pd.read_json(io.BytesIO(raw), lines=True)
        except ValueError:
            df = pd.read_json(io.BytesIO(raw))

    if df.empty:
        raise ValueError("Файл порожній або не містить рядків даних.")
    return _normalize(df)


@st.cache_data(show_spinner=False)
def generate_demo_log(seed: int = 42, n_users: int = 40, n_days: int = 7) -> pd.DataFrame:
    """Генерує реалістичний демонстраційний журнал із вбудованими атаками.

    Дозволяє повноцінно протестувати застосунок без зовнішнього файлу.
    """
    rng = np.random.default_rng(seed)
    users = [f"user{i:02d}" for i in range(n_users)]
    base = pd.Timestamp("2026-03-01", tz="UTC")
    rows: list[dict] = []

    # У кожного користувача стабільна «домашня» та зрідка вторинна IP
    home_ip = {u: f"10.0.{i % 5}.{10 + i}" for i, u in enumerate(users)}
    alt_ip = {u: f"10.0.9.{20 + i}" for i, u in enumerate(users)}

    # 1. Фоновий нормальний трафік (переважно робочі години 8–19)
    for _ in range(6000):
        u = rng.choice(users)
        day = rng.integers(0, n_days)
        hour = int(np.clip(rng.normal(13, 3), 0, 23))
        minute = rng.integers(0, 60)
        ts = base + pd.Timedelta(days=int(day), hours=hour, minutes=int(minute))
        # 92% — домашня IP, інакше вторинна (робота з дому / телефон)
        ip = home_ip[u] if rng.random() < 0.92 else alt_ip[u]
        result = "SUCCESS" if rng.random() > 0.06 else "FAILURE"
        rows.append({"timestamp": ts, "username": u, "ip_address": ip,
                     "result": result, "service": rng.choice(["vpn", "owa", "ssh"])})

    # 2. Брутфорс: одна IP штурмує одного користувача
    bf_ip = "203.0.113.66"
    t0 = base + pd.Timedelta(days=2, hours=3, minutes=10)
    for k in range(25):
        rows.append({"timestamp": t0 + pd.Timedelta(seconds=k * 4),
                     "username": "user07", "ip_address": bf_ip,
                     "result": "FAILURE", "service": "ssh"})
    rows.append({"timestamp": t0 + pd.Timedelta(seconds=110),
                 "username": "user07", "ip_address": bf_ip,
                 "result": "SUCCESS", "service": "ssh"})  # success-after-failure

    # 3. Розпилення паролів: одна IP пробує багато акаунтів
    spray_ip = "198.51.100.23"
    t1 = base + pd.Timedelta(days=4, hours=2)
    for k, u in enumerate(rng.choice(users, 18, replace=False)):
        rows.append({"timestamp": t1 + pd.Timedelta(seconds=k * 7),
                     "username": u, "ip_address": spray_ip,
                     "result": "FAILURE", "service": "owa"})

    # 4. Один акаунт із багатьох IP (можлива компрометація)
    for k in range(6):
        ts = base + pd.Timedelta(days=5, hours=10, minutes=k * 5)
        rows.append({"timestamp": ts, "username": "user11",
                     "ip_address": f"185.220.{k}.{rng.integers(2, 250)}",
                     "result": "SUCCESS", "service": "vpn"})

    df = pd.DataFrame(rows)
    return _normalize(df)


# ============================================================
#  Детектори загроз
# ============================================================
class BruteForceDetector:
    """Повторювані невдалі спроби з однієї IP проти одного користувача
    у межах ковзного часового вікна."""

    def __init__(self, threshold: int, window_sec: int):
        self.threshold = max(2, threshold)
        self.window_sec = max(10, window_sec)

    def detect(self, df: pd.DataFrame) -> list[dict]:
        failures = df[df["result"] == "FAILURE"]
        if failures.empty:
            return []
        out: list[dict] = []
        window = pd.Timedelta(seconds=self.window_sec)
        for (ip, user), grp in failures.groupby(["ip_address", "username"], sort=False):
            times = grp["timestamp"].to_numpy()
            i, n = 0, len(times)
            while i < n:
                limit = times[i] + window.to_timedelta64()
                j = i
                while j < n and times[j] <= limit:
                    j += 1
                count = j - i
                if count >= self.threshold:
                    out.append({
                        "type": "Брутфорс",
                        "ip_address": ip, "username": user,
                        "timestamp": str(pd.Timestamp(times[i])),
                        "details": f"{count} невдалих спроб за {self.window_sec} с",
                        "severity": "CRITICAL" if count >= 10 else "HIGH"})
                    i = j
                else:
                    i += 1
        return out


class PasswordSprayDetector:
    """Одна IP-адреса намагається увійти під багатьма різними акаунтами."""

    def __init__(self, min_users: int):
        self.min_users = max(2, min_users)

    def detect(self, df: pd.DataFrame) -> list[dict]:
        failures = df[df["result"] == "FAILURE"]
        if failures.empty:
            return []
        agg = failures.groupby("ip_address").agg(
            users=("username", "nunique"), attempts=("username", "size"))
        flagged = agg[agg["users"] >= self.min_users]
        return [{
            "type": "Розпилення паролів",
            "ip_address": ip, "username": "—",
            "timestamp": "—",
            "details": f"{row.users} акаунтів, {row.attempts} спроб",
            "severity": "HIGH" if row.users >= self.min_users * 2 else "MEDIUM",
        } for ip, row in flagged.iterrows()]


class OffHoursDetector:
    """Успішні входи у неробочі (нічні) години."""

    def __init__(self, night_start: int, night_end: int):
        self.start = night_start
        self.end = night_end

    def detect(self, df: pd.DataFrame) -> list[dict]:
        success = df[df["result"] == "SUCCESS"]
        if success.empty:
            return []
        if self.start <= self.end:
            mask = success["hour"].between(self.start, self.end, inclusive="left")
        else:  # вікно через північ, напр. 22→6
            mask = (success["hour"] >= self.start) | (success["hour"] < self.end)
        night = success[mask]
        return [{
            "type": "Нічний вхід",
            "ip_address": r.ip_address, "username": r.username,
            "timestamp": str(r.timestamp),
            "details": f"Успішний вхід о {int(r.hour):02d}:00",
            "severity": "LOW",
        } for r in night.itertuples()]


class MultiIPDetector:
    """Один акаунт автентифікується з підозріло великої кількості IP-адрес."""

    def __init__(self, threshold: int):
        self.threshold = max(2, threshold)

    def detect(self, df: pd.DataFrame) -> list[dict]:
        success = df[df["result"] == "SUCCESS"]
        if success.empty:
            return []
        per_user = success.groupby("username")["ip_address"].nunique()
        flagged = per_user[per_user >= self.threshold]
        out = []
        for user, cnt in flagged.items():
            ips = success[success["username"] == user]["ip_address"].unique()[:5]
            out.append({
                "type": "Багато IP",
                "ip_address": ", ".join(ips), "username": user,
                "timestamp": "—",
                "details": f"{cnt} унікальних IP-адрес",
                "severity": "HIGH" if cnt >= self.threshold * 2 else "MEDIUM"})
        return out


class SuccessAfterFailureDetector:
    """Успішний вхід одразу після серії невдач — ознака підбору пароля."""

    def __init__(self, window_sec: int, min_failures: int):
        self.window = pd.Timedelta(seconds=window_sec)
        self.min_failures = max(1, min_failures)

    def detect(self, df: pd.DataFrame) -> list[dict]:
        out: list[dict] = []
        for (user, ip), grp in df.groupby(["username", "ip_address"], sort=False):
            g = grp.sort_values("timestamp")
            results = g["result"].to_numpy()
            times = g["timestamp"].to_numpy()
            for idx in np.where(results == "SUCCESS")[0]:
                low = times[idx] - self.window.to_timedelta64()
                prior = [k for k in range(idx) if times[k] >= low and results[k] == "FAILURE"]
                if len(prior) >= self.min_failures:
                    out.append({
                        "type": "Успіх після невдач",
                        "ip_address": ip, "username": user,
                        "timestamp": str(pd.Timestamp(times[idx])),
                        "details": f"{len(prior)} невдач перед успіхом",
                        "severity": "CRITICAL"})
        return out


class ProfileAnomalyDetector:
    """Статистична аномалія: вхід у годину, нетипову для профілю користувача
    (повністю векторизовано — без iterrows)."""

    def __init__(self, eps: float):
        self.eps = eps

    def detect(self, df: pd.DataFrame) -> list[dict]:
        success = df[df["result"] == "SUCCESS"].copy()
        if success.empty:
            return []
        totals = success.groupby("username")["result"].transform("size")
        hour_counts = success.groupby(["username", "hour"])["result"].transform("size")
        success["profile_freq"] = hour_counts / totals
        anomalies = success[(success["profile_freq"] < self.eps) & (totals >= 20)]
        return [{
            "type": "Аномалія профілю",
            "ip_address": r.ip_address, "username": r.username,
            "timestamp": str(r.timestamp),
            "details": f"частота години {r.profile_freq:.3f} < {self.eps}",
            "severity": "LOW",
        } for r in anomalies.itertuples()]


@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame, cfg_sig: tuple) -> pd.DataFrame:
    """Запускає всі детектори та повертає єдиний DataFrame алертів.

    Кешується за (даними, підписом конфігурації) — повторний клік без
    змін не перераховує нічого.
    """
    (bf_t, bf_w, spray_u, n_start, n_end, mip_t, saf_w, saf_f, eps) = cfg_sig
    detectors = [
        BruteForceDetector(bf_t, bf_w),
        PasswordSprayDetector(spray_u),
        OffHoursDetector(n_start, n_end),
        MultiIPDetector(mip_t),
        SuccessAfterFailureDetector(saf_w, saf_f),
        ProfileAnomalyDetector(eps),
    ]
    alerts: list[dict] = []
    for det in detectors:
        try:
            alerts.extend(det.detect(df))
        except Exception:  # детектор не повинен валити весь аналіз
            continue
    if not alerts:
        return pd.DataFrame(columns=[
            "type", "severity", "username", "ip_address", "timestamp", "details"])
    out = pd.DataFrame(alerts)
    out["severity"] = pd.Categorical(out["severity"], SEVERITY_ORDER, ordered=True)
    cols = ["type", "severity", "username", "ip_address", "timestamp", "details"]
    return out[cols].sort_values("severity").reset_index(drop=True)


def risk_score(alerts: pd.DataFrame) -> int:
    """Сукупний бал ризику 0–100 на основі ваг рівнів небезпеки."""
    if alerts.empty:
        return 0
    raw = sum(SEVERITY_WEIGHT.get(str(s), 0) for s in alerts["severity"])
    return int(min(100, round(100 * (1 - np.exp(-raw / 60)))))


# ============================================================
#  Візуалізації  (легкі агрегації кешуються)
# ============================================================
def _style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["text"], size=13),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=dict(font=dict(size=15)),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _timeline_data(df: pd.DataFrame) -> pd.DataFrame:
    return (df.set_index("timestamp").resample("1h")["result"]
              .value_counts().unstack(fill_value=0).reset_index())


def plot_timeline(df: pd.DataFrame) -> go.Figure:
    ts = _timeline_data(df)
    cols = [c for c in ["SUCCESS", "FAILURE"] if c in ts.columns]
    fig = px.area(ts, x="timestamp", y=cols,
                  color_discrete_map={"SUCCESS": PALETTE["success"],
                                      "FAILURE": PALETTE["failure"]},
                  labels={"value": "Події", "timestamp": "Час", "variable": ""},
                  title="Динаміка подій автентифікації (погодинно)")
    fig.update_traces(line=dict(width=1.5), opacity=0.85)
    return _style_fig(fig)


@st.cache_data(show_spinner=False)
def _top_ips_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return (df[df["result"] == "FAILURE"].groupby("ip_address").size()
              .nlargest(n).reset_index(name="count"))


def plot_top_ips(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = _top_ips_data(df, n)
    fig = px.bar(top, x="count", y="ip_address", orientation="h",
                 color="count", color_continuous_scale=["#3a2330", PALETTE["failure"]],
                 labels={"ip_address": "", "count": "Невдалих спроб"},
                 title=f"Топ-{n} IP за невдалими спробами")
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return _style_fig(fig)


@st.cache_data(show_spinner=False)
def _heatmap_data(df: pd.DataFrame, top_users: int) -> pd.DataFrame:
    success = df[df["result"] == "SUCCESS"]
    if success.empty:
        return pd.DataFrame()
    busiest = success["username"].value_counts().nlargest(top_users).index
    sub = success[success["username"].isin(busiest)]
    pivot = sub.groupby(["username", "hour"]).size().unstack(fill_value=0)
    return pivot.reindex(columns=range(24), fill_value=0)


def plot_user_heatmap(df: pd.DataFrame, top_users: int = 20) -> go.Figure:
    pivot = _heatmap_data(df, top_users)
    if pivot.empty:
        return _style_fig(go.Figure())
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Teal",
                    labels={"x": "Година доби", "y": "", "color": "Входів"},
                    title=f"Теплова карта активності (топ-{top_users} користувачів)")
    return _style_fig(fig, height=max(360, 22 * len(pivot)))


def plot_severity_donut(alerts: pd.DataFrame) -> go.Figure:
    counts = alerts["severity"].value_counts()
    counts = counts.reindex(SEVERITY_ORDER, fill_value=0)
    counts = counts[counts > 0]
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values), hole=0.62,
        marker=dict(colors=[SEVERITY_COLORS[s] for s in counts.index]),
        textinfo="label+value", sort=False))
    fig.update_layout(title="Розподіл за рівнем небезпеки", showlegend=False)
    return _style_fig(fig, height=320)


def plot_risk_gauge(score: int) -> go.Figure:
    if score >= 70:
        color = SEVERITY_COLORS["CRITICAL"]
    elif score >= 40:
        color = SEVERITY_COLORS["HIGH"]
    elif score >= 15:
        color = SEVERITY_COLORS["MEDIUM"]
    else:
        color = PALETTE["accent"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(suffix="", font=dict(size=44)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=PALETTE["muted"]),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0, 15], color="rgba(52,224,196,0.10)"),
                dict(range=[15, 40], color="rgba(255,209,102,0.10)"),
                dict(range=[40, 70], color="rgba(255,138,76,0.12)"),
                dict(range=[70, 100], color="rgba(255,77,109,0.14)"),
            ])))
    fig.update_layout(title="Загальний бал ризику")
    return _style_fig(fig, height=300)


# ============================================================
#  Звіт
# ============================================================
def build_report(df: pd.DataFrame, alerts: pd.DataFrame, source: str) -> str:
    total = len(df)
    succ = int((df["result"] == "SUCCESS").sum())
    fail = total - succ
    lines = [
        "=" * 64,
        "  ЗВІТ ПРО АНАЛІЗ ЖУРНАЛІВ АВТЕНТИФІКАЦІЇ",
        "=" * 64,
        f"Джерело даних .......... {source or 'демонстраційний набір'}",
        f"Згенеровано ............ {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC",
        f"Інтервал ............... {df['timestamp'].min()} — {df['timestamp'].max()}",
        "",
        "ЗАГАЛЬНА СТАТИСТИКА",
        "-" * 64,
        f"Всього записів ......... {total:,}",
        f"Успішних входів ........ {succ:,} ({succ / max(total, 1):.1%})",
        f"Невдалих входів ........ {fail:,} ({fail / max(total, 1):.1%})",
        f"Унікальних користувачів  {df['username'].nunique():,}",
        f"Унікальних IP-адрес .... {df['ip_address'].nunique():,}",
        f"Бал ризику ............. {risk_score(alerts)} / 100",
        "",
        "ВИЯВЛЕНІ ЗАГРОЗИ",
        "-" * 64,
    ]
    if alerts.empty:
        lines.append("Підозрілої активності не виявлено.")
    else:
        for atype, grp in alerts.groupby("type", sort=False):
            lines.append(f"\n[{atype}] — {len(grp)} алертів")
            for r in grp.head(8).itertuples():
                stamp = "" if r.timestamp == "—" else f" @ {r.timestamp}"
                lines.append(f"   • [{r.severity}] {r.username} / {r.ip_address}"
                             f"{stamp} — {r.details}")
            if len(grp) > 8:
                lines.append(f"   … та ще {len(grp) - 8} записів")
    lines += ["", "=" * 64, "Згенеровано Sentinel · ХНУВС"]
    return "\n".join(lines)


# ============================================================
#  Оформлення (CSS) та UI-компоненти
# ============================================================
def inject_css() -> None:
    p = PALETTE
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

      .stApp {{
        background:
          radial-gradient(1200px 600px at 85% -10%, rgba(52,224,196,0.08), transparent 60%),
          radial-gradient(900px 500px at 0% 0%, rgba(91,134,255,0.06), transparent 55%),
          {p['bg']};
        color: {p['text']};
        font-family: 'IBM Plex Sans', sans-serif;
      }}
      .block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
      h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }}

      /* Бренд-шапка */
      .brand {{
        display: flex; align-items: center; gap: 14px; margin-bottom: 4px;
      }}
      .brand .logo {{
        width: 46px; height: 46px; border-radius: 12px;
        display: grid; place-items: center; font-size: 24px;
        background: linear-gradient(135deg, {p['accent']}, {p['accent_dim']});
        box-shadow: 0 6px 22px rgba(52,224,196,0.30);
      }}
      .brand h1 {{ font-size: 1.6rem; margin: 0; }}
      .brand .sub {{ color: {p['muted']}; font-size: 0.86rem; margin-top: -2px; }}

      /* KPI-картки */
      .kpi-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px; margin: 18px 0 6px;
      }}
      .kpi {{
        background: linear-gradient(160deg, {p['surface']}, {p['surface_2']});
        border: 1px solid {p['border']}; border-radius: 16px;
        padding: 16px 18px; position: relative; overflow: hidden;
        transition: transform .18s ease, border-color .18s ease;
      }}
      .kpi:hover {{ transform: translateY(-3px); border-color: {p['accent_dim']}; }}
      .kpi::after {{
        content: ''; position: absolute; right: -20px; top: -20px;
        width: 70px; height: 70px; border-radius: 50%;
        background: var(--glow, rgba(52,224,196,0.12)); filter: blur(6px);
      }}
      .kpi .label {{ color: {p['muted']}; font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 0.06em; }}
      .kpi .value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem;
        font-weight: 700; margin-top: 4px; }}
      .kpi .delta {{ font-size: 0.78rem; color: {p['muted']}; margin-top: 2px; }}

      /* Бейджі рівнів небезпеки */
      .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.74rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}

      /* Сайдбар */
      section[data-testid="stSidebar"] {{
        background: {p['surface']}; border-right: 1px solid {p['border']};
      }}
      section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

      /* Кнопки */
      .stButton > button {{
        border-radius: 10px; border: 1px solid {p['border']};
        background: {p['surface_2']}; color: {p['text']}; font-weight: 600;
        transition: all .15s ease;
      }}
      .stButton > button:hover {{ border-color: {p['accent']}; color: {p['accent']}; }}
      .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {p['accent']}, {p['accent_dim']});
        color: #04201b; border: none; box-shadow: 0 6px 18px rgba(52,224,196,0.25);
      }}

      /* DataFrame та елементи */
      [data-testid="stDataFrame"] {{ border: 1px solid {p['border']}; border-radius: 12px; }}
      hr {{ border-color: {p['border']}; }}
      .section-title {{ font-family: 'Space Grotesk'; font-size: 1.15rem;
        margin: 18px 0 6px; color: {p['text']}; }}
      .hint {{ color: {p['muted']}; font-size: 0.86rem; }}
    </style>
    """, unsafe_allow_html=True)


def badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, PALETTE["muted"])
    return (f"<span class='badge' style='background:{color}22;color:{color};"
            f"border:1px solid {color}55'>{severity}</span>")


def kpi_card(label: str, value: str, delta: str = "", glow: str = None) -> str:
    glow = glow or "rgba(52,224,196,0.12)"
    return (f"<div class='kpi' style='--glow:{glow}'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"<div class='delta'>{delta}</div></div>")


def kpi_row(cards: list[str]) -> None:
    st.markdown("<div class='kpi-grid'>" + "".join(cards) + "</div>",
                unsafe_allow_html=True)


# ============================================================
#  Сторінки
# ============================================================
def page_upload(state: AppState) -> None:
    st.markdown("<div class='section-title'>Джерело даних</div>", unsafe_allow_html=True)
    st.markdown("<p class='hint'>Завантажте журнал у форматі CSV або JSON. "
                "Обовʼязкові поля (із синонімами): час, користувач, IP, результат. "
                "Або скористайтесь демонстраційним набором.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fmt = st.radio("Формат файлу", ["CSV", "JSON"], horizontal=True)
        uploaded = st.file_uploader("Файл журналу", type=["csv", "json"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            try:
                state.raw_df = load_log(uploaded.getvalue(), fmt)
                state.source_name = uploaded.name
                state.alerts_df = pd.DataFrame()
                st.success(f"Завантажено {len(state.raw_df):,} записів із «{uploaded.name}».")
            except Exception as exc:
                st.error(f"Не вдалося обробити файл: {exc}")
    with c2:
        st.markdown("<p class='hint'>Демонстрація</p>", unsafe_allow_html=True)
        if st.button("Завантажити демо-журнал", width='stretch'):
            state.raw_df = generate_demo_log()
            state.source_name = "демонстраційний набір"
            state.alerts_df = pd.DataFrame()
            st.success(f"Згенеровано {len(state.raw_df):,} демо-записів із вбудованими атаками.")

    if not state.raw_df.empty:
        st.markdown("<div class='section-title'>Перші записи</div>", unsafe_allow_html=True)
        st.dataframe(state.raw_df.head(12), width='stretch', hide_index=True)


def page_overview(state: AppState) -> None:
    df = state.raw_df
    total = len(df)
    succ = int((df["result"] == "SUCCESS").sum())
    fail = total - succ
    kpi_row([
        kpi_card("Всього подій", f"{total:,}"),
        kpi_card("Успішних", f"{succ:,}", f"{succ / max(total,1):.0%} від усіх",
                 glow="rgba(52,211,153,0.16)"),
        kpi_card("Невдалих", f"{fail:,}", f"{fail / max(total,1):.0%} від усіх",
                 glow="rgba(248,113,113,0.16)"),
        kpi_card("Унікальних IP", f"{df['ip_address'].nunique():,}"),
        kpi_card("Користувачів", f"{df['username'].nunique():,}"),
    ])
    st.plotly_chart(plot_timeline(df), width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_top_ips(df), width='stretch')
    with c2:
        st.plotly_chart(plot_user_heatmap(df), width='stretch')


def page_analysis(state: AppState) -> None:
    df = state.raw_df
    top = st.columns([1, 1, 4])
    with top[0]:
        run = st.button("▶ Запустити аналіз", type="primary", width='stretch')
    if run:
        with st.spinner("Виконується аналіз шести детекторів…"):
            state.alerts_df = run_analysis(df, state.cfg.signature())

    alerts = state.alerts_df
    if alerts is None or alerts.empty:
        if run:
            st.success("Аналіз завершено — підозрілої активності не виявлено.")
        else:
            st.info("Натисніть «Запустити аналіз», щоб перевірити журнал.")
        return

    score = risk_score(alerts)
    sev_counts = alerts["severity"].value_counts()
    kpi_row([
        kpi_card("Всього алертів", f"{len(alerts):,}"),
        kpi_card("Критичних", f"{int(sev_counts.get('CRITICAL', 0))}",
                 glow="rgba(255,77,109,0.18)"),
        kpi_card("Високих", f"{int(sev_counts.get('HIGH', 0))}",
                 glow="rgba(255,138,76,0.16)"),
        kpi_card("Типів загроз", f"{alerts['type'].nunique()}"),
    ])

    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(plot_risk_gauge(score), width='stretch')
    with c2:
        st.plotly_chart(plot_severity_donut(alerts), width='stretch')

    st.markdown("<div class='section-title'>Деталізація алертів</div>", unsafe_allow_html=True)
    type_filter = st.multiselect("Фільтр за типом",
                                 options=sorted(alerts["type"].unique()),
                                 default=sorted(alerts["type"].unique()))
    view = alerts[alerts["type"].isin(type_filter)]
    st.dataframe(view, width='stretch', hide_index=True,
                 column_config={
                     "type": "Тип", "severity": "Небезпека", "username": "Користувач",
                     "ip_address": "IP-адреса", "timestamp": "Час", "details": "Деталі"})

    cols = st.columns(2)
    cols[0].download_button("⬇ Алерти (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                            file_name="alerts.csv", width='stretch')
    cols[1].download_button("⬇ Алерти (JSON)",
                            view.to_json(orient="records", force_ascii=False, indent=2),
                            file_name="alerts.json", width='stretch')


def page_settings(state: AppState) -> None:
    cfg = state.cfg
    st.markdown("<p class='hint'>Зміни застосовуються при наступному запуску аналізу.</p>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Брутфорс**")
        cfg.bf_threshold = st.slider("Поріг невдалих спроб", 2, 50, cfg.bf_threshold)
        cfg.bf_window = st.slider("Вікно, секунди", 10, 3600, cfg.bf_window, 10)
        st.markdown("**Розпилення паролів**")
        cfg.spray_min_users = st.slider("Мін. атакованих акаунтів", 2, 50, cfg.spray_min_users)
        st.markdown("**Багато IP**")
        cfg.multiip_threshold = st.slider("Поріг унікальних IP", 2, 20, cfg.multiip_threshold)
    with c2:
        st.markdown("**Нічні входи**")
        cfg.night_start, cfg.night_end = st.slider(
            "Нічні години (від → до)", 0, 23, (cfg.night_start, cfg.night_end))
        st.markdown("**Успіх після невдач**")
        cfg.saf_min_failures = st.slider("Мін. попередніх невдач", 1, 20, cfg.saf_min_failures)
        cfg.saf_window = st.slider("Вікно success-after-failure, с", 30, 1800, cfg.saf_window, 30)
        st.markdown("**Аномалія профілю**")
        cfg.profile_eps = st.slider("Поріг рідкісності (ε)", 0.001, 0.1, cfg.profile_eps, 0.001)


def page_report(state: AppState) -> None:
    if state.alerts_df is None or state.alerts_df.empty:
        st.info("Спочатку виконайте аналіз на сторінці «Аналіз загроз».")
    report = build_report(state.raw_df, state.alerts_df if state.alerts_df is not None
                          else pd.DataFrame(), state.source_name)
    st.code(report, language=None)
    st.download_button("⬇ Завантажити звіт (TXT)", report.encode("utf-8"),
                       file_name="auth_analysis_report.txt", width='stretch')


# ============================================================
#  Головна функція
# ============================================================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    inject_css()

    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    state: AppState = st.session_state.app_state

    # Шапка
    st.markdown(f"""
      <div class='brand'>
        <div class='logo'>{APP_ICON}</div>
        <div><h1>Sentinel</h1>
        <div class='sub'>Аналіз журналів автентифікації · виявлення підозрілої активності</div></div>
      </div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Навігація
    with st.sidebar:
        st.markdown("### Навігація")
        page = st.radio("Розділ", ["Завантаження", "Огляд", "Аналіз загроз",
                                   "Налаштування", "Звіт"], label_visibility="collapsed")
        st.markdown("<hr>", unsafe_allow_html=True)
        if not state.raw_df.empty:
            st.markdown(f"<p class='hint'>Активний набір:<br><b>{state.source_name}</b><br>"
                        f"{len(state.raw_df):,} записів</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='hint'>Дані не завантажено</p>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='hint'>Детектори: брутфорс · розпилення паролів · "
                    "нічні входи · багато IP · успіх після невдач · аномалія профілю</p>",
                    unsafe_allow_html=True)

    needs_data = page in {"Огляд", "Аналіз загроз", "Звіт"}
    if needs_data and state.raw_df.empty:
        st.info("Спочатку завантажте журнал на сторінці «Завантаження» "
                "або скористайтесь демо-набором.")
        return

    try:
        if page == "Завантаження":
            page_upload(state)
        elif page == "Огляд":
            page_overview(state)
        elif page == "Аналіз загроз":
            page_analysis(state)
        elif page == "Налаштування":
            page_settings(state)
        elif page == "Звіт":
            page_report(state)
    except Exception as exc:
        st.error(f"Виникла помилка під час обробки: {exc}")


if __name__ == "__main__":
    main()

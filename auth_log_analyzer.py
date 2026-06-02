# auth_log_analyzer.py
# Автор: Куліуш Д.О., гр. ННІ-4-24-203 Кб, ХНУВС, 2026
# Веб-додаток для аналізу журналів автентифікації користувачів

import io, ipaddress
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
#  Структури даних та нормалізація
# ============================================================
@dataclass
class AppState:
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    filtered_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: list = field(default_factory=list)
    bf_threshold: int = 5
    bf_window: int = 60
    profile_eps: float = 0.01


RENAME_MAP = {
    'time': 'timestamp', 'date': 'timestamp', 'ts': 'timestamp',
    'user': 'username', 'login': 'username', 'account': 'username',
    'src_ip': 'ip_address', 'source_ip': 'ip_address', 'ip': 'ip_address',
    'status': 'result', 'outcome': 'result',
}
REQUIRED_COLS = ['timestamp', 'username', 'ip_address', 'result']


# ============================================================
#  Завантаження та нормалізація даних
# ============================================================
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Нормалізує схему DataFrame до стандартного формату."""
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items()
                            if k in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Відсутні колонки: {missing}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['result'] = df['result'].astype(str).str.upper().str.strip()
    df['result'] = df['result'].apply(
        lambda x: 'SUCCESS' if x in {'SUCCESS', 'OK'} else 'FAILURE')

    def is_valid_ip(ip):
        try:
            ipaddress.ip_address(str(ip))
            return True
        except ValueError:
            return False

    df = df[df['ip_address'].apply(is_valid_ip)]
    df = df.dropna(subset=['timestamp', 'username', 'ip_address'])
    if 'service' not in df.columns:
        df['service'] = 'unknown'
    return df.sort_values('timestamp').reset_index(drop=True)


def load_log(uploaded_file, fmt: str) -> pd.DataFrame:
    """Зчитує завантажений файл журналу та повертає DataFrame."""
    try:
        raw = uploaded_file.read()
        if len(raw) > 50 * 1024 * 1024:
            st.warning('Файл перевищує 50 МБ. Обробка може тривати довго.')
        if fmt == 'CSV':
            df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
        elif fmt == 'JSON':
            try:
                df = pd.read_json(io.BytesIO(raw), lines=True)
            except ValueError:
                df = pd.read_json(io.BytesIO(raw))
        if df.empty:
            st.error('Файл порожній або не містить даних.')
            return pd.DataFrame()
        return _normalize(df)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f'Помилка зчитування журналу: {exc}')
        return pd.DataFrame()


# ============================================================
#  Детектори аномалій
# ============================================================
class BruteForceDetector:
    """Виявлення брутфорс-атак за методом ковзного вікна."""

    def __init__(self, threshold: int = 5, window_sec: int = 60):
        if threshold < 1:
            raise ValueError('threshold має бути >= 1')
        if window_sec < 10:
            raise ValueError('window_sec має бути >= 10')
        self.threshold = threshold
        self.window_sec = window_sec

    def detect(self, df: pd.DataFrame) -> list[dict]:
        failures = df[df['result'] == 'FAILURE'].copy()
        if failures.empty:
            return []
        alerts = []
        for (ip, user), grp in failures.groupby(['ip_address', 'username']):
            times = grp['timestamp'].sort_values().tolist()
            i = 0
            while i < len(times):
                window_end = times[i] + pd.Timedelta(seconds=self.window_sec)
                in_window = [t for t in times[i:] if t <= window_end]
                if len(in_window) >= self.threshold:
                    alerts.append({
                        'type': 'Брутфорс', 'ip_address': ip,
                        'username': user, 'first_attempt': str(times[i]),
                        'attempts_in_window': len(in_window),
                        'severity': 'HIGH' if len(in_window) >= 10 else 'MEDIUM'})
                    i += len(in_window)
                else:
                    i += 1
        return alerts


class PasswordSprayDetector:
    """Виявлення атак розпилення паролів."""

    def __init__(self, min_users: int = 5, window_min: int = 10):
        self.min_users = min_users
        self.window = pd.Timedelta(minutes=window_min)

    def detect(self, df: pd.DataFrame) -> list[dict]:
        failures = df[df['result'] == 'FAILURE']
        alerts = []
        for ip, grp in failures.groupby('ip_address'):
            unique_users = grp['username'].nunique()
            if unique_users >= self.min_users:
                alerts.append({
                    'type': 'Розпилення паролів',
                    'ip_address': ip, 'targeted_users': unique_users,
                    'total_attempts': len(grp), 'severity': 'HIGH'})
        return alerts


class ProfileAnomalyDetector:
    """Виявлення аномалій за часовим профілем поведінки."""

    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def build_profiles(self, df: pd.DataFrame) -> dict:
        success = df[df['result'] == 'SUCCESS'].copy()
        success['hour'] = success['timestamp'].dt.hour
        profiles = {}
        for user, grp in success.groupby('username'):
            counts = Counter(grp['hour'])
            total = sum(counts.values())
            if total > 0:
                profiles[user] = {h: c / total for h, c in counts.items()}
        return profiles

    def detect(self, df: pd.DataFrame) -> list[dict]:
        profiles = self.build_profiles(df)
        success = df[df['result'] == 'SUCCESS'].copy()
        success['hour'] = success['timestamp'].dt.hour
        alerts = []
        for _, row in success.iterrows():
            user, h = row['username'], row['hour']
            if user not in profiles:
                continue
            freq = profiles[user].get(h, 0.0)
            if freq < self.eps:
                alerts.append({
                    'type': 'Аномалія профілю',
                    'username': user, 'timestamp': str(row['timestamp']),
                    'hour': h, 'profile_freq': round(freq, 4),
                    'severity': 'LOW'})
        return alerts


def run_all_detectors(df: pd.DataFrame, state: AppState) -> list[dict]:
    """Запускає всі детектори та повертає список алертів."""
    alerts = []
    for Cls, name, args in [
        (BruteForceDetector,     'брутфорсу',   (state.bf_threshold, state.bf_window)),
        (PasswordSprayDetector,  'розпилення',  ()),
        (ProfileAnomalyDetector, 'профілю',     (state.profile_eps,)),
    ]:
        try:
            alerts.extend(Cls(*args).detect(df))
        except Exception as exc:
            st.warning(f'Помилка: {exc}')
    return alerts


# ============================================================
#  Візуалізація та формування звіту
# ============================================================
def plot_timeline(df: pd.DataFrame) -> go.Figure:
    ts = (df.set_index('timestamp').resample('1h')['result']
            .value_counts().unstack(fill_value=0).reset_index())
    color_map = {'SUCCESS': '#2ecc71', 'FAILURE': '#e74c3c'}
    cols = [c for c in ['SUCCESS', 'FAILURE'] if c in ts.columns]
    return px.line(ts, x='timestamp', y=cols, color_discrete_map=color_map,
                   labels={'value': 'Кількість', 'timestamp': 'Час'},
                   title='Динаміка подій автентифікації (погодинно)')


def plot_top_ips(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = (df[df['result'] == 'FAILURE'].groupby('ip_address').size()
             .nlargest(n).reset_index(name='count'))
    return px.bar(top, x='ip_address', y='count',
                  color='count', color_continuous_scale='Reds',
                  labels={'ip_address': 'IP-адреса', 'count': 'Невдалих спроб'},
                  title=f'Топ-{n} IP за кількістю невдалих спроб')


def plot_user_heatmap(df: pd.DataFrame) -> go.Figure:
    success = df[df['result'] == 'SUCCESS'].copy()
    success['hour'] = success['timestamp'].dt.hour
    pivot = success.groupby(['username', 'hour']).size().unstack(fill_value=0)
    if pivot.empty:
        return go.Figure()
    return px.imshow(pivot,
                     labels={'x': 'Година доби', 'y': 'Користувач', 'color': 'Входів'},
                     title='Теплова карта активності (успішні входи)',
                     color_continuous_scale='Blues', aspect='auto')


def generate_report(df: pd.DataFrame, alerts: list[dict]) -> str:
    lines = ['=' * 60,
             'ЗВІТ ПРО АНАЛІЗ ЖУРНАЛІВ АВТЕНТИФІКАЦІЇ',
             '=' * 60,
             f'Всього записів: {len(df):,}',
             f'Успішних: {(df["result"]=="SUCCESS").sum():,}',
             f'Невдалих: {(df["result"]=="FAILURE").sum():,}']
    by_type = {}
    for a in alerts:
        by_type.setdefault(a['type'], []).append(a)
    for atype, items in by_type.items():
        lines.append(f'{atype}: {len(items)} алертів')
        for item in items[:5]:
            lines.append(f'  {item}')
    return '\n'.join(lines)


# ============================================================
#  Сторінки Streamlit та головна функція
# ============================================================
def page_overview(df: pd.DataFrame) -> None:
    st.header('Огляд журналу автентифікації')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Всього подій', len(df))
    col2.metric('Успішних', (df['result'] == 'SUCCESS').sum())
    col3.metric('Невдалих', (df['result'] == 'FAILURE').sum())
    col4.metric('Унікальних IP', df['ip_address'].nunique())

    st.subheader('Динаміка подій у часі')
    st.plotly_chart(plot_timeline(df), use_container_width=True)

    st.subheader('Топ-10 IP за кількістю невдалих спроб')
    st.plotly_chart(plot_top_ips(df), use_container_width=True)


def page_analysis(df: pd.DataFrame, state: AppState) -> None:
    st.header('Аналіз загроз')
    if st.button('Запустити аналіз', type='primary'):
        with st.spinner('Виконується аналіз...'):
            state.alerts = run_all_detectors(df, state)
        st.success(f'Виявлено {len(state.alerts)} алертів.')
    if state.alerts:
        df_alerts = pd.DataFrame(state.alerts)
        for atype in df_alerts['type'].unique():
            st.subheader(atype)
            st.dataframe(df_alerts[df_alerts['type'] == atype],
                         use_container_width=True)


def page_settings(state: AppState) -> None:
    st.header('Налаштування алгоритмів')
    state.bf_threshold = st.slider(
        'Поріг брутфорсу (спроб у вікні)', 2, 50, state.bf_threshold)
    state.bf_window = st.slider(
        'Вікно брутфорсу (секунди)', 10, 3600, state.bf_window)
    state.profile_eps = st.slider(
        'Поріг рідкісності (epsilon)', 0.001, 0.1, state.profile_eps, 0.001)
    st.info('Зміни застосовуються при наступному запуску аналізу.')


def main() -> None:
    st.set_page_config(
        page_title='Аналіз журналів автентифікації',
        page_icon='🛡', layout='wide')
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    state = st.session_state.app_state

    page = st.sidebar.radio('Навігація',
                            ['Завантаження', 'Огляд', 'Аналіз загроз',
                             'Профілі', 'Фільтрація', 'Звіт', 'Налаштування'])
    try:
        if page == 'Завантаження':
            st.header('Завантаження журналу')
            fmt = st.radio('Формат файлу', ['CSV', 'JSON'], horizontal=True)
            uploaded = st.file_uploader('Оберіть файл журналу',
                                        type=['csv', 'json'])
            if uploaded is not None:
                state.raw_df = load_log(uploaded, fmt)
                if not state.raw_df.empty:
                    st.success(f'Завантажено {len(state.raw_df)} записів.')
                    st.dataframe(state.raw_df.head(), use_container_width=True)

        elif page == 'Огляд':
            if state.raw_df.empty:
                st.info('Спочатку завантажте журнал.')
            else:
                page_overview(state.raw_df)

        elif page == 'Аналіз загроз':
            if state.raw_df.empty:
                st.info('Спочатку завантажте журнал.')
            else:
                page_analysis(state.raw_df, state)

        elif page == 'Профілі':
            if state.raw_df.empty:
                st.info('Спочатку завантажте журнал.')
            else:
                st.header('Профілі користувачів')
                st.plotly_chart(plot_user_heatmap(state.raw_df),
                                use_container_width=True)

        elif page == 'Фільтрація':
            if state.raw_df.empty:
                st.info('Спочатку завантажте журнал.')
            else:
                st.header('Фільтрація')
                users = ['(всі)'] + sorted(state.raw_df['username'].unique().tolist())
                sel = st.selectbox('Користувач', users)
                df = state.raw_df
                if sel != '(всі)':
                    df = df[df['username'] == sel]
                state.filtered_df = df
                st.dataframe(df, use_container_width=True)

        elif page == 'Звіт':
            if state.raw_df.empty:
                st.info('Спочатку завантажте журнал.')
            else:
                st.header('Звіт')
                report = generate_report(state.raw_df, state.alerts)
                st.code(report)
                st.download_button('Завантажити TXT', report,
                                   file_name='report.txt')

        elif page == 'Налаштування':
            page_settings(state)

    except Exception as exc:
        st.error(f'Виникла помилка: {exc}')
        st.exception(exc)


if __name__ == '__main__':
    main()

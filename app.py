
from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Optional

import gspread
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials
from plotly.subplots import make_subplots

logger = logging.getLogger("gold_dashboard")

# ============================================================================
# 1. CONFIG
# ============================================================================
st.set_page_config(page_title="Galeri24 Aurum Analytics", page_icon="🥇", layout="wide")

APP_TITLE = "Galeri24 Aurum Price Analytics"
SPREADSHEET_NAME = "Data Emas"
EXCLUDED_CATEGORIES = {"SENTRA BUYBACK - SENTRA BUYBACK"}
DATA_START_LABEL = "05 Juni 2026"
DATA_TTL_SECONDS = 15 * 60
REFRESH_COOLDOWN_S = 120
STALE_AFTER_HOURS = 24
SESSION_HOURS_PER_DAY = 12  # 08:00..19:00
TIMEZONE = "Asia/Jakarta"
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

# spreadsheets.readonly cukup untuk membaca; drive.readonly dibutuhkan agar
# gspread bisa membuka spreadsheet berdasarkan nama.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

RAW_REQUIRED = {"timestamp", "kategori", "berat_gram", "harga_jual", "harga_buyback"}
CURRENT_REQUIRED = {
    "Kategori", "Berat (gr)", "Harga Jual (Per Batang)",
    "Harga Jual (Per Gram)", "Harga Buyback", "Latest Update",
}
DAILY_REQUIRED = {
    "tanggal", "kategori", "berat_gram", "harga_buka", "harga_tertinggi",
    "harga_terendah", "harga_tutup", "frekuensi_update", "pergerakan_rupiah",
}

_ID_MONTHS = {
    "januari": "January", "februari": "February", "maret": "March", "april": "April",
    "mei": "May", "juni": "June", "juli": "July", "agustus": "August",
    "september": "September", "oktober": "October", "november": "November",
    "desember": "December", "agu": "Aug", "okt": "Oct", "des": "Dec",
}


class DataError(Exception):
    """Error data/konfigurasi yang pesannya aman ditampilkan ke pengguna."""


# ============================================================================
# 2. DATA ACCESS
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_google_credentials() -> Credentials:
    try:
        info = dict(st.secrets["gcp_service_account"])
    except KeyError as exc:
        raise DataError("Secret 'gcp_service_account' tidak ditemukan.") from exc
    return Credentials.from_service_account_info(info, scopes=SCOPES)


@st.cache_resource(show_spinner=False)
def app_store() -> dict:
    """Penyimpanan in-memory lintas sesi: snapshot valid terakhir + cooldown refresh."""
    return {"data": None, "loaded_at": None, "last_refresh": 0.0}


def load_raw_sheets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = gspread.authorize(get_google_credentials())
    book = client.open(SPREADSHEET_NAME)
    frames = []
    for tab in ("Data Mentah", "Harga Terkini", "Fluktuasi Harian"):
        frames.append(pd.DataFrame(book.worksheet(tab).get_all_records()))
    return tuple(frames)  # type: ignore[return-value]


# ============================================================================
# 3. PREPARATION & VALIDATION
# ============================================================================
def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataError(f"Tab '{name}' tidak memiliki kolom wajib: {', '.join(missing)}")


def parse_id_datetime(series: pd.Series) -> pd.Series:
    """Parse tanggal yang mungkin memakai nama bulan Indonesia / akhiran WIB."""
    text = series.astype(str).str.replace(r"\bWIB\b", "", regex=True).str.strip()

    def _sub(match: re.Match) -> str:
        return _ID_MONTHS.get(match.group(0).lower(), match.group(0))

    text = text.str.replace(r"[A-Za-z]+", _sub, regex=True)
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def _clean_frame(
    df: pd.DataFrame, *, category: str, numeric: list[str], required_numeric: list[str],
    dates: list[str],
) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df[category] = df[category].astype(str).str.strip()
    df = df[~df[category].isin(EXCLUDED_CATEGORIES | {"", "nan", "None"})]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in dates:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=required_numeric + dates)
    return df.reset_index(drop=True), before - len(df)


def enrich_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Fitur data per jam: return log, penanda gap semalam, jam, hari."""
    key = ["kategori", "berat_gram"]
    raw = raw[raw["harga_jual"] > 0].sort_values(key + ["timestamp"]).copy()
    raw["tanggal"] = raw["timestamp"].dt.normalize()
    raw["jam"] = raw["timestamp"].dt.hour
    raw["hari_idx"] = raw["timestamp"].dt.dayofweek
    raw["log_p"] = np.log(raw["harga_jual"])
    grouped = raw.groupby(key, sort=False)
    raw["ret"] = grouped["log_p"].diff()
    raw["overnight"] = grouped["tanggal"].diff().dt.days.gt(0)
    raw["gap_h"] = grouped["timestamp"].diff().dt.total_seconds() / 3600
    return raw


def prepare_data(
    raw: pd.DataFrame, current: pd.DataFrame, daily: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    require_columns(raw, RAW_REQUIRED, "Data Mentah")
    require_columns(current, CURRENT_REQUIRED, "Harga Terkini")
    require_columns(daily, DAILY_REQUIRED, "Fluktuasi Harian")

    raw, raw_dropped = _clean_frame(
        raw, category="kategori",
        numeric=["berat_gram", "harga_jual", "harga_buyback"],
        required_numeric=["berat_gram", "harga_jual", "harga_buyback"],
        dates=["timestamp"],
    )
    price_cols = ["Berat (gr)", "Harga Jual (Per Batang)", "Harga Jual (Per Gram)", "Harga Buyback"]
    current, current_dropped = _clean_frame(
        current, category="Kategori", numeric=price_cols, required_numeric=price_cols, dates=[],
    )
    daily_num = [
        "berat_gram", "harga_buka", "harga_tertinggi", "harga_terendah",
        "harga_tutup", "frekuensi_update", "pergerakan_rupiah",
    ]
    daily, daily_dropped = _clean_frame(
        daily, category="kategori", numeric=daily_num,
        required_numeric=["berat_gram", "harga_buka", "harga_tertinggi", "harga_terendah", "harga_tutup"],
        dates=["tanggal"],
    )
    for name, df in (("Data Mentah", raw), ("Harga Terkini", current), ("Fluktuasi Harian", daily)):
        if df.empty:
            raise DataError(f"Tab '{name}' kosong setelah proses pembersihan.")

    raw = enrich_raw(raw)

    # ---- Harga terkini: spread & break-even dihitung dari harga, bukan dari kolom sheet
    current["Latest Update Parsed"] = parse_id_datetime(current["Latest Update"])
    sale = current["Harga Jual (Per Batang)"]
    buyback = current["Harga Buyback"]
    current["spread_rp_calc"] = sale - buyback
    current["spread_pct_calc"] = np.where(sale > 0, (sale - buyback) / sale * 100, np.nan)
    # Berapa persen buyback harus naik agar menyamai harga jual sekarang.
    current["breakeven_buyback_rise_pct"] = np.where(buyback > 0, (sale / buyback - 1) * 100, np.nan)
    current = (
        current.sort_values("Latest Update Parsed", na_position="first")
        .drop_duplicates(subset=["Kategori", "Berat (gr)"], keep="last")
        .reset_index(drop=True)
    )

    # ---- Harian: return close-to-close & std 7 observasi terakhir
    daily = daily.sort_values(["kategori", "berat_gram", "tanggal"]).copy()
    grouped = daily.groupby(["kategori", "berat_gram"], sort=False)
    daily["return_pct"] = grouped["harga_tutup"].pct_change(fill_method=None) * 100
    daily["vol_7obs_pct"] = grouped["return_pct"].transform(
        lambda s: s.rolling(7, min_periods=3).std()
    )
    daily["frekuensi_update"] = daily["frekuensi_update"].fillna(0)

    quality = {
        "raw_dropped": raw_dropped, "current_dropped": current_dropped,
        "daily_dropped": daily_dropped,
    }
    return raw, current, daily, quality


@st.cache_data(ttl=DATA_TTL_SECONDS, show_spinner="Mengambil & memproses data...")
def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Ambil + bersihkan sekali per TTL. Exception tidak pernah ikut ter-cache."""
    return prepare_data(*load_raw_sheets())


# ============================================================================
# 4. HELPERS
# ============================================================================
def rupiah(value: float) -> str:
    return "—" if pd.isna(value) else f"Rp {value:,.0f}"


def get_quote(current: pd.DataFrame, category: str, weight: float) -> Optional[pd.Series]:
    match = current[(current["Kategori"] == category) & np.isclose(current["Berat (gr)"], weight)]
    return None if match.empty else match.iloc[0]


def now_wib() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)


def latest_update_text(current: pd.DataFrame, raw: pd.DataFrame) -> str:
    if not raw.empty:
        return f"{raw['timestamp'].max():%d %b %Y %H:%M} WIB"
    parsed = current["Latest Update Parsed"].dropna()
    return f"{parsed.max():%d %b %Y %H:%M}" if not parsed.empty else "Tidak tersedia"


def robust_z(s: pd.Series) -> pd.Series:
    """Z-score robust (median/MAD) hanya dari return non-nol (harga ritel 'berundak')."""
    nonzero = s[s.notna() & s.ne(0)]
    if len(nonzero) < 10:
        return pd.Series(0.0, index=s.index)
    med = nonzero.median()
    mad = (nonzero - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def chart_config() -> dict:
    return {"displaylogo": False, "responsive": True, "scrollZoom": True}


def style(fig: go.Figure, title: str | None = None, height: int = 420, rp_axis: bool = True) -> go.Figure:
    fig.update_layout(
        title=title, height=height, hovermode="x unified", template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    if rp_axis:
        fig.update_yaxes(tickprefix="Rp ", tickformat=",.0f")
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=chart_config())


# ============================================================================
# 5. DATA HEALTH & METHODOLOGY
# ============================================================================
def coverage_frame(raw: pd.DataFrame) -> pd.DataFrame:
    per_day = raw.groupby("tanggal")["jam"].nunique().rename("jam_terobservasi").reset_index()
    per_day["kelengkapan_pct"] = per_day["jam_terobservasi"].clip(upper=SESSION_HOURS_PER_DAY) / SESSION_HOURS_PER_DAY * 100
    return per_day


def render_data_health(raw, current, daily, quality) -> None:
    cov = coverage_frame(raw)
    with st.expander("🔎 Data Quality & Coverage", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observasi per jam", f"{len(raw):,}")
        c2.metric("Quote terkini", f"{len(current):,}")
        c3.metric("Baris harian", f"{len(daily):,}")
        c4.metric("Kelengkapan window", f"{cov['kelengkapan_pct'].mean():.0f}%",
                  help="Rata-rata jam (dari 12 jam 08:00-19:00) yang punya snapshot per hari.")
        st.write(
            f"**Rentang per jam:** {raw['timestamp'].min():%d %b %Y} → {raw['timestamp'].max():%d %b %Y}  \n"
            f"**Baris dibuang saat cleaning** (nilai tidak valid): "
            f"Data Mentah {quality['raw_dropped']:,} · Harga Terkini {quality['current_dropped']:,} · "
            f"Fluktuasi Harian {quality['daily_dropped']:,}"
        )


def render_methodology(raw: pd.DataFrame) -> None:
    st.subheader("🧭 Metodologi & Keterbatasan Data")
    cov = coverage_frame(raw)
    fig = px.bar(cov, x="tanggal", y="jam_terobservasi",
                 title="Jumlah jam terobservasi per hari (maks. 12)")
    fig.add_hline(y=SESSION_HOURS_PER_DAY, line_dash="dot")
    show(style(fig, height=300, rp_axis=False))

    st.markdown(
        f"""
**Sumber & cakupan**
- Harga ritel emas batangan dari galeri24.co.id, di-scrape tiap jam pukul 08:00–19:00 WIB sejak {DATA_START_LABEL}.
- Ini **bukan** harga spot XAU/USD dan tidak mengandung tick pasar.

**Definisi metrik**
- *Spread (%)* = (harga jual − buyback) / harga jual.
- *Break-even buyback (%)* = harga jual / buyback − 1 (kenaikan buyback agar menyamai harga jual saat ini).
- *Open / Close harian* = snapshot pertama / terakhir dalam window scraper, bukan open/close pasar.
- *Realized volatility harian* = √Σ(return log per jam²) dalam satu hari, tanpa return semalam.
- *Vol 7 observasi* = std return harian dari 7 baris terakhir (bukan 7 hari kalender jika ada hari kosong).
- *Persentil harga* = posisi harga terkini terhadap close 30 hari terakhir; bukan "all-time" (data baru {DATA_START_LABEL}).
- *Gap semalam* = return dari snapshot terakhir hari t ke snapshot pertama hari t+1.

**Keterbatasan**
- Sesi London-New York (± 19:00–04:00 WIB) tidak terobservasi; gap semalam dipakai sebagai proksi, bukan pengganti.
- Harga ritel berubah berundak (banyak return = 0), sehingga statistik berbasis rata-rata perlu dibaca hati-hati.
- Sampel masih pendek (beberapa bulan) — belum layak untuk kesimpulan statistik yang kuat atau model prediktif.
- Dashboard ini bersifat informasional, **bukan saran investasi**.
"""
    )


# ============================================================================
# 6. TAB: RINGKASAN
# ============================================================================
def position_table(quotes: pd.DataFrame, daily_w: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    rows = []
    for _, q in quotes.iterrows():
        cat, cur = q["Kategori"], q["Harga Jual (Per Batang)"]
        hist = daily_w[daily_w["kategori"] == cat].sort_values("tanggal")
        pct = chg = np.nan
        if not hist.empty:
            recent = hist[hist["tanggal"] >= hist["tanggal"].max() - pd.Timedelta(days=window_days)]
            if len(recent) >= 5:
                pct = (recent["harga_tutup"] <= cur).mean() * 100
            prev = hist[hist["tanggal"] < hist["tanggal"].max()]["harga_tutup"]
            if len(prev):
                chg = (cur / prev.iloc[-1] - 1) * 100
        rows.append({
            "Merek": cat, "Harga Jual": rupiah(cur), "Buyback": rupiah(q["Harga Buyback"]),
            "Spread (%)": q["spread_pct_calc"], "Break-even (%)": q["breakeven_buyback_rise_pct"],
            "Persentil 30H": pct, "Δ vs close sebelumnya (%)": chg,
        })
    return pd.DataFrame(rows)


def render_summary(quotes: pd.DataFrame, daily_w: pd.DataFrame, raw: pd.DataFrame, weight: float) -> None:
    if quotes.empty:
        st.warning("Tidak ada quote aktif untuk berat ini.")
        return
    best_spread = quotes.loc[quotes["spread_pct_calc"].idxmin()]
    best_buyback = quotes.loc[quotes["Harga Buyback"].idxmax()]
    cheapest = quotes.loc[quotes["Harga Jual (Per Batang)"].idxmin()]

    st.subheader(f"Ringkasan Harga {weight:g} gram")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median Harga Jual", rupiah(quotes["Harga Jual (Per Batang)"].median()))
    k2.metric("Termurah (beli)", rupiah(cheapest["Harga Jual (Per Batang)"]),
              delta=str(cheapest["Kategori"]), delta_color="off")
    k3.metric("Buyback Tertinggi", rupiah(best_buyback["Harga Buyback"]),
              delta=str(best_buyback["Kategori"]), delta_color="off")
    k4.metric("Spread Terendah", f"{best_spread['spread_pct_calc']:.2f}%",
              delta=str(best_spread["Kategori"]), delta_color="off")

    st.caption(
        "Spread = biaya masuk-keluar saat ini, bukan prediksi return. "
        "Persentil dihitung dari close 30 hari terakhir, bukan all-time."
    )
    table = position_table(quotes, daily_w)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Spread (%)": st.column_config.NumberColumn(format="%.2f"),
            "Break-even (%)": st.column_config.NumberColumn(
                format="%.2f",
                help="Kenaikan buyback yang dibutuhkan agar menyamai harga jual sekarang."),
            "Persentil 30H": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f",
                help="0 = terendah, 100 = tertinggi dalam 30 hari terakhir."),
            "Δ vs close sebelumnya (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )


# ============================================================================
# 7. TAB: TREN
# ============================================================================
@st.fragment
def candlestick_panel(daily_v: pd.DataFrame, weight: float) -> None:
    cats = sorted(daily_v["kategori"].unique())
    cat = st.selectbox("Merek", cats, key="candle_category")
    d = daily_v[daily_v["kategori"] == cat].sort_values("tanggal").copy()
    if len(d) < 2:
        st.info("Data belum cukup untuk candlestick.")
        return
    d["ma7"] = d["harga_tutup"].rolling(7, min_periods=3).mean()
    missing = pd.date_range(d["tanggal"].min(), d["tanggal"].max()).difference(pd.DatetimeIndex(d["tanggal"]))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=d["tanggal"], open=d["harga_buka"], high=d["harga_tertinggi"], low=d["harga_terendah"],
        close=d["harga_tutup"], name="OHLC",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["tanggal"], y=d["ma7"], name="MA7", line=dict(width=1.6)), row=1, col=1)
    fig.add_trace(go.Bar(x=d["tanggal"], y=d["frekuensi_update"], name="Jumlah update",
                         marker_color="rgba(100,116,139,.55)"), row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    if len(missing):
        fig.update_xaxes(rangebreaks=[dict(values=missing.strftime("%Y-%m-%d").tolist())])
    fig.update_xaxes(rangeselector=dict(buttons=[
        dict(count=7, label="1W", step="day", stepmode="backward"),
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(step="all", label="Semua"),
    ]), row=1, col=1)
    fig.update_yaxes(tickprefix="Rp ", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Update", row=2, col=1)
    style(fig, f"{cat} — {weight:g} gram", height=560, rp_axis=False)
    show(fig)
    st.caption(
        "Open/Close = snapshot pertama/terakhir 08:00–19:00 WIB (bukan OHLC pasar). "
        "Batang kecil + jumlah update rendah berarti harga jarang berubah dalam sehari."
    )


def render_trend(daily_v: pd.DataFrame, weight: float) -> None:
    if daily_v.empty:
        st.warning("Tidak ada observasi pada filter yang dipilih.")
        return
    st.subheader("📈 Harga Penutupan Harian")
    indexed = st.toggle("Tampilkan sebagai indeks (basis 100 = awal periode)", key="trend_indexed")
    plot = daily_v.sort_values("tanggal").copy()
    if indexed:
        base = plot.groupby("kategori")["harga_tutup"].transform("first")
        plot["nilai"] = plot["harga_tutup"] / base * 100
        fig = px.line(plot, x="tanggal", y="nilai", color="kategori")
        fig.update_traces(hovertemplate="%{fullData.name}: %{y:.2f}<extra></extra>")
        style(fig, f"Indeks harga tutup — {weight:g} gram", rp_axis=False)
    else:
        fig = px.line(plot, x="tanggal", y="harga_tutup", color="kategori")
        fig.update_traces(hovertemplate="%{fullData.name}: Rp %{y:,.0f}<extra></extra>")
        style(fig, f"Observed daily close — {weight:g} gram")
    show(fig)

    st.divider()
    st.subheader("🕯️ Candlestick Harian")
    candlestick_panel(daily_v, weight)

    st.divider()
    st.subheader("📊 Return Harian & Volatilitas")
    ret = daily_v.dropna(subset=["return_pct"])
    if ret.empty:
        st.info("Belum cukup data untuk return harian.")
        return
    fig = px.bar(ret, x="tanggal", y="return_pct", color="kategori", barmode="group")
    fig.add_hline(y=0, line_width=1)
    fig.update_traces(hovertemplate="%{fullData.name}: %{y:.2f}%<extra></extra>")
    show(style(fig, "Return close-to-close harian (%)", rp_axis=False))

    fig = px.line(ret.dropna(subset=["vol_7obs_pct"]), x="tanggal", y="vol_7obs_pct", color="kategori")
    fig.update_traces(hovertemplate="%{fullData.name}: %{y:.2f}%<extra></extra>")
    show(style(fig, "Std return harian, 7 observasi terakhir (%)", rp_axis=False))
    st.caption("Untuk volatilitas berbasis data per jam, lihat tab Volatilitas & Sesi.")


# ============================================================================
# 8. TAB: VOLATILITAS & SESI (data per jam)
# ============================================================================
def render_volatility(raw_v: pd.DataFrame, weight: float) -> None:
    st.subheader("⚡ Volatilitas & Sesi (berbasis data per jam)")
    valid = raw_v[raw_v["ret"].notna()]
    intraday = valid[~valid["overnight"]]
    overnight = valid[valid["overnight"]]
    if len(intraday) < 30:
        st.info("Data per jam belum cukup pada filter ini.")
        return

    # ---- Realized volatility harian
    rv = (intraday.assign(r2=intraday["ret"] ** 2)
          .groupby(["kategori", "tanggal"], as_index=False)["r2"].sum())
    rv["rv_pct"] = np.sqrt(rv["r2"]) * 100
    fig = px.line(rv, x="tanggal", y="rv_pct", color="kategori")
    fig.update_traces(hovertemplate="%{fullData.name}: %{y:.3f}%<extra></extra>")
    show(style(fig, f"Realized volatility harian — {weight:g} gram (%)", rp_axis=False))
    st.caption("RV = √Σ(return log per jam²) dalam window 08:00–19:00; return semalam tidak dihitung.")

    st.divider()

    # ---- Heatmap jam x hari
    heat = (intraday.assign(abs_ret=intraday["ret"].abs() * 100)
            .pivot_table(index="hari_idx", columns="jam", values="abs_ret", aggfunc="mean")
            .sort_index())
    heat.index = [HARI[i] for i in heat.index]
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Jam (WIB)", y="Hari", color="|ret| rata-rata (%)"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=55, b=10),
                      title="Rata-rata |return| per jam × hari")
    show(fig)
    st.caption("Sampel per sel masih kecil; baca sebagai eksplorasi, bukan pola yang terbukti.")

    st.divider()

    # ---- Gap semalam vs intraday (vol per √jam)
    st.markdown("**Seberapa informatif sesi yang tidak terobservasi?**")

    def vol_per_sqrt_hour(df: pd.DataFrame) -> float:
        hours = df["gap_h"].sum()
        return float(np.sqrt((df["ret"] ** 2).sum() / hours) * 100) if hours > 0 else np.nan

    rows = []
    for cat in sorted(valid["kategori"].unique()):
        sub = valid[valid["kategori"] == cat]
        rows.append({
            "kategori": cat,
            "Intraday": vol_per_sqrt_hour(sub[~sub["overnight"]]),
            "Gap semalam": vol_per_sqrt_hour(sub[sub["overnight"]]),
        })
    comp = pd.DataFrame(rows).dropna()
    overall_intra, overall_night = vol_per_sqrt_hour(intraday), vol_per_sqrt_hour(overnight)
    if len(overnight) >= 5 and overall_intra > 0:
        m1, m2, m3 = st.columns(3)
        m1.metric("Vol intraday (%/√jam)", f"{overall_intra:.4f}")
        m2.metric("Vol gap semalam (%/√jam)", f"{overall_night:.4f}")
        m3.metric("Rasio semalam / intraday", f"{overall_night / overall_intra:.2f}×",
                  help="≈1 berarti sesi malam tidak lebih volatil per satuan waktu; >1 berarti lebih volatil.")
    if not comp.empty:
        long = comp.melt(id_vars="kategori", var_name="Jenis", value_name="vol")
        fig = px.bar(long, x="kategori", y="vol", color="Jenis", barmode="group")
        fig.update_traces(hovertemplate="%{fullData.name}: %{y:.4f}%/√jam<extra></extra>")
        show(style(fig, "Volatilitas per √jam: intraday vs gap semalam", height=340, rp_axis=False))
    st.caption(
        "Varians bersifat aditif, jadi dinormalisasi per √jam agar adil terhadap gap 13 jam. "
        "Rasio jauh di atas 1 menunjukkan bahwa window 08:00–19:00 melewatkan pergerakan yang berarti."
    )

    st.divider()

    # ---- Anomali
    st.markdown("**Deteksi anomali (robust z-score, |z| ≥ 4)**")
    an = intraday.copy()
    an["z"] = an.groupby("kategori")["ret"].transform(robust_z)
    an = an[an["z"].abs() >= 4].sort_values("z", key=lambda s: s.abs(), ascending=False).head(15)
    if an.empty:
        st.success("Tidak ada return per jam yang tergolong anomali pada filter ini.")
    else:
        out = pd.DataFrame({
            "Waktu (WIB)": an["timestamp"].dt.strftime("%d %b %Y %H:%M"),
            "Merek": an["kategori"], "Harga Jual": an["harga_jual"].map(rupiah),
            "Return (%)": an["ret"] * 100, "Robust z": an["z"],
        })
        st.dataframe(out, width="stretch", hide_index=True, column_config={
            "Return (%)": st.column_config.NumberColumn(format="%.3f"),
            "Robust z": st.column_config.NumberColumn(format="%.1f"),
        })
        st.caption("Bisa berupa pergerakan nyata atau glitch scraper — verifikasi sebelum menyimpulkan.")


# ============================================================================
# 9. TAB: SPREAD
# ============================================================================
def render_spread(quotes: pd.DataFrame, current_all: pd.DataFrame, raw_v: pd.DataFrame, weight: float) -> None:
    st.subheader("💸 Analisis Spread")
    if quotes.empty:
        st.warning("Tidak ada quote untuk berat ini.")
        return
    spread = quotes.sort_values("spread_pct_calc")
    fig = px.bar(spread, x="spread_pct_calc", y="Kategori", orientation="h", text="spread_pct_calc")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside",
                      hovertemplate="%{y}: %{x:.2f}%<extra></extra>")
    fig.update_layout(xaxis_title="Spread (%)", yaxis_title=None)
    show(style(fig, f"Spread jual → buyback saat ini — {weight:g} gram", rp_axis=False))

    if not raw_v.empty:
        st.divider()
        hist = raw_v.assign(spread_pct=(raw_v["harga_jual"] - raw_v["harga_buyback"]) / raw_v["harga_jual"] * 100)
        daily_spread = hist.groupby(["kategori", "tanggal"], as_index=False)["spread_pct"].mean()
        fig = px.line(daily_spread, x="tanggal", y="spread_pct", color="kategori")
        fig.update_traces(hovertemplate="%{fullData.name}: %{y:.2f}%<extra></extra>")
        show(style(fig, "Spread rata-rata harian (%)", rp_axis=False))

    st.divider()
    st.subheader("📉 Efisiensi Harga per Gram")
    options = sorted(current_all["Kategori"].unique())
    chosen = st.multiselect("Merek", options, default=options[:5], key="efficiency_categories")
    eff = current_all[current_all["Kategori"].isin(chosen)].sort_values(["Kategori", "Berat (gr)"])
    if eff.empty:
        st.info("Pilih minimal satu merek.")
        return
    fig = px.line(eff, x="Berat (gr)", y="Harga Jual (Per Gram)", color="Kategori", markers=True)
    fig.update_traces(hovertemplate="%{fullData.name} · %{x:g} gr: Rp %{y:,.0f}/gr<extra></extra>")
    fig.update_layout(hovermode="closest")
    show(style(fig, "Harga jual per gram vs berat"))
    st.caption("Harga per gram lebih rendah = unit economics lebih murah untuk denominasi tersebut, bukan sinyal return.")


# ============================================================================
# 10. TAB: SIMULASI
# ============================================================================
def _weights_of(current: pd.DataFrame, category: str) -> list[float]:
    return sorted(current.loc[current["Kategori"] == category, "Berat (gr)"].dropna().unique())


def render_break_even(current: pd.DataFrame) -> None:
    st.subheader("🧮 Kalkulator Break-even")
    cat = st.selectbox("Merek", sorted(current["Kategori"].unique()), key="be_category")
    weight = st.selectbox("Berat (gram)", _weights_of(current, cat), key="be_weight")
    qty = st.number_input("Jumlah keping", min_value=1, value=1, step=1, key="be_quantity")
    quote = get_quote(current, cat, weight)
    if quote is None:
        st.warning("Quote untuk kombinasi tersebut tidak tersedia.")
        return
    sale, buyback = float(quote["Harga Jual (Per Batang)"]), float(quote["Harga Buyback"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Modal beli", rupiah(sale * qty))
    c2.metric("Nilai buyback saat ini", rupiah(buyback * qty))
    c3.metric("Selisih jika langsung dijual", rupiah((buyback - sale) * qty))
    if buyback > 0:
        st.info(f"Buyback perlu naik ±**{(sale / buyback - 1) * 100:.2f}%** "
                f"({rupiah(buyback)} → {rupiah(sale)}) agar impas, dengan asumsi harga jual saat ini tetap.")
    else:
        st.error("Harga buyback tidak valid.")


def render_valuation(current: pd.DataFrame) -> None:
    st.subheader("💰 Valuasi Aset")
    cat = st.selectbox("Merek aset", sorted(current["Kategori"].unique()), key="val_category")
    weight = st.selectbox("Berat (gram)", _weights_of(current, cat), key="val_weight")
    qty = st.number_input("Jumlah keping", min_value=1, value=1, step=1, key="val_quantity")
    buy_price = st.number_input("Harga beli historis (Rp/keping)", min_value=0, value=1_000_000,
                                step=50_000, key="val_purchase_price")
    quote = get_quote(current, cat, weight)
    if quote is None:
        st.warning("Quote untuk kombinasi tersebut tidak tersedia.")
        return
    initial = buy_price * qty
    if initial <= 0:
        st.warning("Harga beli historis harus lebih besar dari 0.")
        return
    value_now = float(quote["Harga Buyback"]) * qty
    pnl = value_now - initial
    c1, c2, c3 = st.columns(3)
    c1.metric("Modal historis", rupiah(initial))
    c2.metric("Nilai buyback", rupiah(value_now))
    c3.metric("P/L", rupiah(pnl), delta=f"{pnl / initial * 100:+.2f}%")
    st.caption("Memakai buyback saat ini; belum memasukkan pajak atau biaya lain di luar data sumber.")


def render_upgrade(current: pd.DataFrame) -> None:
    st.subheader("🔄 Simulasi Tukar Tambah")
    st.caption("Emas lama dijual di harga buyback, emas baru dibeli di harga jual (snapshot saat ini).")
    cats = sorted(current["Kategori"].unique())
    left, right = st.columns(2)
    with left:
        st.markdown("**Emas lama — dijual**")
        old_cat = st.selectbox("Merek lama", cats, key="trade_old_category")
        old_w = st.selectbox("Berat lama (gram)", _weights_of(current, old_cat), key="trade_old_weight")
        old_q = st.number_input("Jumlah lama", min_value=1, value=1, step=1, key="trade_old_qty")
    with right:
        st.markdown("**Emas baru — dibeli**")
        new_cat = st.selectbox("Merek baru", cats, key="trade_new_category")
        new_w = st.selectbox("Berat baru (gram)", _weights_of(current, new_cat), key="trade_new_weight")
        new_q = st.number_input("Jumlah baru", min_value=1, value=1, step=1, key="trade_new_qty")
    old_quote, new_quote = get_quote(current, old_cat, old_w), get_quote(current, new_cat, new_w)
    if old_quote is None or new_quote is None:
        st.warning("Quote salah satu kombinasi belum tersedia.")
        return
    proceeds = float(old_quote["Harga Buyback"]) * old_q
    cost = float(new_quote["Harga Jual (Per Batang)"]) * new_q
    diff = cost - proceeds
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Hasil jual emas lama", rupiah(proceeds))
    m2.metric("Biaya emas baru", rupiah(cost))
    m3.metric("Top-up" if diff > 0 else "Kembalian", rupiah(abs(diff)))


@st.fragment
def render_simulators(current: pd.DataFrame) -> None:
    render_break_even(current)
    st.divider()
    render_valuation(current)
    st.divider()
    render_upgrade(current)


# ============================================================================
# 11. MAIN
# ============================================================================
def load_with_fallback():
    store = app_store()
    try:
        data = load_prepared()
        store["data"], store["loaded_at"] = data, now_wib()
        return data, None
    except DataError as exc:
        logger.error("Data error: %s", exc)
        message = str(exc)
    except Exception:  # noqa: BLE001 - jangan bocorkan detail ke pengunjung
        logger.exception("Gagal memuat data dari Google Sheets")
        message = "Gagal mengambil data dari Google Sheets."
    if store["data"] is not None:
        return store["data"], (
            f"{message} Menampilkan snapshot terakhir yang valid "
            f"(dimuat {store['loaded_at']:%d %b %Y %H:%M} WIB)."
        )
    st.error(f"{message} Coba lagi beberapa saat.")
    st.stop()


def main() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Harga ritel emas batangan Galeri24 (bukan spot XAU/USD). Data diobservasi tiap jam "
        "08:00–19:00 WIB; sesi malam tidak terobservasi. Bukan saran investasi."
    )

    refresh_col, status_col = st.columns([1, 5])
    with refresh_col:
        if st.button("🔄 Refresh", width="stretch"):
            store = app_store()
            if time.time() - store["last_refresh"] < REFRESH_COOLDOWN_S:
                st.toast(f"Tunggu {REFRESH_COOLDOWN_S}s antar refresh.")
            else:
                store["last_refresh"] = time.time()
                load_prepared.clear()
                st.rerun()

    (raw, current, daily, quality), fallback_notice = load_with_fallback()

    with status_col:
        age_h = (now_wib() - raw["timestamp"].max()).total_seconds() / 3600
        if fallback_notice:
            st.warning(fallback_notice)
        elif age_h > STALE_AFTER_HOURS:
            st.error(f"⚠️ Data terakhir {latest_update_text(current, raw)} ({age_h:.0f} jam lalu) — "
                     "scraper kemungkinan berhenti.")
        else:
            st.success(f"Data terakhir: {latest_update_text(current, raw)}")

    render_data_health(raw, current, daily, quality)

    # ---------------- Filter global (halaman utama, bukan sidebar)
    weights = sorted(daily["berat_gram"].unique())
    default_idx = weights.index(1) if 1 in weights else 0
    f1, f2, f3 = st.columns([1, 2, 2])
    with f1:
        weight = st.selectbox("⚙️ Berat (gram)", weights, index=default_idx)

    raw_w = raw[np.isclose(raw["berat_gram"], weight)]
    current_w = current[np.isclose(current["Berat (gr)"], weight)]
    daily_w = daily[np.isclose(daily["berat_gram"], weight)]
    if daily_w.empty:
        st.warning(f"Tidak ada data historis untuk {weight:g} gram.")
        st.stop()

    min_date, max_date = daily_w["tanggal"].min().date(), daily_w["tanggal"].max().date()
    default_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=90)).date())
    with f2:
        selected = st.date_input("📅 Rentang historis", value=(default_start, max_date),
                                 min_value=min_date, max_value=max_date)
    brand_options = sorted(daily_w["kategori"].unique())
    with f3:
        brands = st.multiselect("🏷️ Merek di grafik", brand_options, default=brand_options)

    if not (isinstance(selected, tuple) and len(selected) == 2):
        st.info("Pilih tanggal akhir untuk melengkapi rentang.")
        st.stop()
    if not brands:
        st.info("Pilih minimal satu merek.")
        st.stop()
    start_date, end_date = (pd.Timestamp(d) for d in selected)

    daily_v = daily_w[daily_w["kategori"].isin(brands)
                      & daily_w["tanggal"].between(start_date, end_date)]
    raw_v = raw_w[raw_w["kategori"].isin(brands)
                  & raw_w["tanggal"].between(start_date, end_date)]

    tabs = st.tabs(["📊 Ringkasan", "📈 Tren", "⚡ Volatilitas & Sesi",
                    "💸 Spread", "🧮 Simulasi", "🧭 Metodologi"])
    with tabs[0]:
        render_summary(current_w, daily_w, raw, weight)
    with tabs[1]:
        render_trend(daily_v, weight)
    with tabs[2]:
        render_volatility(raw_v, weight)
    with tabs[3]:
        render_spread(current_w, current, raw_v, weight)
    with tabs[4]:
        render_simulators(current)
    with tabs[5]:
        render_methodology(raw)

    st.divider()
    st.caption(
        f"Sumber: galeri24.co.id/harga-emas · Observasi 08:00–19:00 WIB · "
        f"Data sejak {DATA_START_LABEL} · Berat aktif: {weight:g} gram"
    )


if __name__ == "__main__":
    main()
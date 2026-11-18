"""Funciones sencillas para el análisis de reservas marítimas."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from kmedoids import KMedoids

from sklearn.metrics import silhouette_score, calinski_harabasz_score

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler


def resumen_calidad(df):

    resumen = pd.DataFrame({
        "tipo": df.dtypes,
        "nulos": df.isna().sum(),
        "unicos": df.nunique()
    })

    resumen["ceros"] = [
        (df[col] == 0).sum()
        if pd.api.types.is_numeric_dtype(df[col])
        else "—"
        for col in df.columns
    ]

    resumen["negativos"] = [
        (df[col] < 0).sum()
        if pd.api.types.is_numeric_dtype(df[col])
        else "—"
        for col in df.columns
    ]

    return resumen


def normalizar_espacios(df: pd.DataFrame, columnas: Iterable[str]) -> pd.DataFrame:
    """Copia el DataFrame y normaliza espacios sin borrar observaciones."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame de pandas.")
    columnas = list(columnas)
    faltantes = [col for col in columnas if col not in df]
    if faltantes:
        raise KeyError(f"Columnas no encontradas: {faltantes}")
    out = df.copy()
    for col in columnas:
        out[col] = out[col].astype("string").str.strip().str.replace(
            r"\s+", " ", regex=True
        )
    return out


def normalizar_categorias(
    df: pd.DataFrame,
    columnas: Iterable[str],
    *,
    minusculas: bool = True,
    quitar_acentos: bool = False,
) -> pd.DataFrame:
    """Normaliza categorías textuales conservando filas y valores faltantes."""
    out = normalizar_espacios(df, columnas)
    for col in columnas:
        valores = out[col]
        if minusculas:
            valores = valores.str.lower()
        if quitar_acentos:
            valores = valores.map(
                lambda value: (
                    unicodedata.normalize("NFKD", value)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                    if pd.notna(value)
                    else value
                )
            )
        out[col] = valores
    return out


# Actualización por INPC
def ajustar_inpc(
    df: pd.DataFrame,
    columnas: str | list[str],
    fecha_col: str,
    fecha_objetivo: str | int,
    inpc: pd.Series,
    frecuencia: str = "anual",
    criterio_anual: str = "diciembre",
) -> pd.DataFrame:
    """Actualiza importes sin sobrescribir las columnas originales."""
    if not isinstance(df, pd.DataFrame) or not isinstance(inpc, pd.Series):
        raise TypeError("df debe ser DataFrame e inpc debe ser Series de pandas.")
    cols = [columnas] if isinstance(columnas, str) else list(columnas)
    if not cols or len(set(cols)) != len(cols):
        raise ValueError("Indica columnas únicas y no vacías.")
    missing = [col for col in [fecha_col, *cols] if col not in df]
    if missing:
        raise KeyError(f"Columnas no encontradas: {missing}")
    if frecuencia not in {"anual", "mensual", "diaria"}:
        raise ValueError("frecuencia debe ser anual, mensual o diaria.")
    if criterio_anual not in {"diciembre", "promedio"}:
        raise ValueError("criterio_anual debe ser diciembre o promedio.")

    salidas = [f"{col}_ajustada" for col in cols] + ["factor_inpc"]
    if any(col in df for col in salidas):
        raise ValueError("Algunas columnas de salida ya existen en el DataFrame.")

    objetivo = str(fecha_objetivo).strip()
    if re.fullmatch(r"\d{4}", objetivo):
        objetivo += "-12"
    objetivo = pd.to_datetime(objetivo, errors="coerce")
    if pd.isna(objetivo):
        raise ValueError("fecha_objetivo inválida.")
    objetivo = objetivo.to_period("M")

    if isinstance(inpc.index, pd.PeriodIndex):
        indice = inpc.index.asfreq("M")
    else:
        if pd.api.types.is_numeric_dtype(inpc.index.dtype):
            raise ValueError("Los índices del INPC deben tener fechas, no números.")
        fechas = pd.to_datetime(inpc.index, errors="coerce")
        if pd.isna(fechas).any():
            raise ValueError("Índice temporal del INPC inválido.")
        indice = fechas.to_period("M")
    if indice.has_duplicates:
        raise ValueError("Hay meses repetidos en la serie INPC.")
    valores = pd.to_numeric(inpc.to_numpy(), errors="coerce")
    if pd.isna(valores).any() or (valores <= 0).any():
        raise ValueError("Los valores INPC deben ser positivos y válidos.")
    mensual = pd.Series(valores, index=indice).sort_index()
    if objetivo not in mensual:
        raise ValueError(f"No existe INPC para el destino {objetivo}.")

    if frecuencia == "anual":
        raw = df[fecha_col].astype("string").str.strip()
        years = pd.to_numeric(raw, errors="coerce")
        valid = raw.str.fullmatch(r"\d{4}(?:\.0+)?").fillna(False)
        if not valid.all() or years.isna().any() or not years.between(1000, 9999).all():
            raise ValueError(
                "Para frecuencia anual, fecha_col debe contener años de cuatro cifras."
            )
        years = years.astype(int)
        if criterio_anual == "diciembre":
            periodos = pd.PeriodIndex([f"{year}-12" for year in years], freq="M")
            origen = pd.Series(mensual.reindex(periodos).to_numpy(), index=df.index)
        else:
            cantidad = mensual.groupby(mensual.index.year).size()
            faltan = sorted({year for year in years if cantidad.get(year, 0) != 12})
            if faltan:
                raise ValueError(
                    f"Faltan INPC mensuales para calcular promedio anual: {faltan}"
                )
            anual = mensual.groupby(mensual.index.year).mean()
            origen = pd.Series(years.map(anual).to_numpy(), index=df.index)
    else:
        raw = df[fecha_col].astype("string").str.strip()
        fechas = pd.to_datetime(raw, errors="coerce", dayfirst=True, format="mixed")
        if fechas.isna().any():
            raise ValueError("Hay fechas de origen inválidas.")
        periodos = pd.PeriodIndex(fechas.dt.to_period("M"), freq="M")
        origen = pd.Series(mensual.reindex(periodos).to_numpy(), index=df.index)

    if origen.isna().any():
        raise ValueError("La serie INPC no cubre todos los periodos de origen.")
    result = df.copy()
    result["factor_inpc"] = mensual.loc[objetivo] / origen
    for col in cols:
        numero = pd.to_numeric(df[col], errors="coerce")
        if (numero.isna() & df[col].notna()).any():
            raise ValueError(f"La columna {col} contiene importes no numéricos.")
        result[f"{col}_ajustada"] = numero * result["factor_inpc"]
    return result


# Snapshots y reproducibilidad
def _hash_archivo(ruta: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def guardar_snapshot(
    df: pd.DataFrame,
    ruta: str | Path,
    *,
    periodo_valuacion: str,
    fuente_inpc: str | None = None,
    criterio_actualizacion: str | None = None,
    version: str = "1",
    archivos_origen: Iterable[str | Path] = (),
) -> Path:
    """Guarda un snapshot Parquet y un JSON mínimo de reproducibilidad."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame de pandas.")
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    metadata = {
        "periodo_valuacion": periodo_valuacion,
        "fuente_inpc": fuente_inpc,
        "criterio_actualizacion": criterio_actualizacion,
        "version_procesamiento": version,
        "archivos_origen": {
            str(Path(origen)): _hash_archivo(origen)
            for origen in archivos_origen
            if Path(origen).exists()
        },
    }
    ruta.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ruta


def snapshot_es_valido(
    ruta: str | Path,
    *,
    periodo_valuacion: str,
    version: str = "1",
    archivos_origen: Iterable[str | Path] = (),
) -> bool:
    """Comprueba existencia, metadatos y hashes de un snapshot."""
    ruta = Path(ruta)
    metadata_ruta = ruta.with_suffix(".json")
    if not ruta.exists() or not metadata_ruta.exists():
        return False
    try:
        metadata = json.loads(metadata_ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("periodo_valuacion") != periodo_valuacion:
        return False
    if metadata.get("version_procesamiento") != version:
        return False
    for origen in archivos_origen:
        origen = Path(origen)
        hashes = metadata.get("archivos_origen", {})
        if not origen.exists() or hashes.get(str(origen)) != _hash_archivo(origen):
            return False
    return True


def leer_snapshot(ruta: str | Path) -> pd.DataFrame:
    """Lee un snapshot existente o informa exactamente qué falta."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el snapshot: {ruta}")
    return pd.read_parquet(ruta)


# Resúmenes y exportación
def resumen_emision_anual(df: pd.DataFrame, anio: str = "año") -> pd.DataFrame:
    """Suma importes y calcula crecimiento, retención y comisiones."""
    columnas = [
        "prima_emitida_ajustada",
        "prima_retenida_ajustada",
        "comision_directa_ajustada",
    ]
    faltantes = [col for col in [anio, *columnas] if col not in df]
    if faltantes:
        raise KeyError(f"Columnas no encontradas: {faltantes}")
    agregado = df.groupby(anio, dropna=False)[columnas].sum().sort_index()
    for col, etiqueta in [
        (columnas[0], "crecimiento_emitida_pct"),
        (columnas[1], "crecimiento_retenida_pct"),
    ]:
        anterior = agregado[col].shift()
        agregado[etiqueta] = np.where(
            anterior > 0, (agregado[col] / anterior - 1) * 100, np.nan
        )
    base = agregado[columnas[0]].where(agregado[columnas[0]] > 0)
    agregado["retencion_pct"] = agregado[columnas[1]] / base * 100
    agregado["comisiones_pct"] = agregado[columnas[2]] / base * 100
    return agregado.reset_index()


def participacion_por_categoria(
    df: pd.DataFrame, categoria: str = "tipo_embarcacion"
) -> pd.DataFrame:
    """Calcula participaciones netas y conserva categorías con importes negativos."""
    requeridas = ["año", categoria, "prima_emitida_ajustada"]
    faltantes = [col for col in requeridas if col not in df]
    if faltantes:
        raise KeyError(f"Columnas no encontradas: {faltantes}")
    agregado = (
        df.groupby(["año", categoria], dropna=False)["prima_emitida_ajustada"]
        .sum()
        .unstack(fill_value=0)
    )
    total = agregado.sum(axis=1)
    if (total <= 0).any():
        raise ValueError("Hay años cuya emisión neta total no es positiva.")
    return agregado.div(total, axis=0).mul(100)


def evaluar_modelo(X, metodo, k_max):
    """
    Evalúa un método de clustering para k = 2 hasta k_max.

    Retorna un DataFrame con:
    - Silhouette
    - Calinski-Harabasz
    - WCSS
    - Tamaños de los grupos
    - Tamaño del grupo más pequeño

    Parameters
    ----------
    metodo : str
        Método de clustering a evaluar. Puede ser "Ward", "Liga completa", "K-means" o "K-medoids".
    """

    X = np.asarray(X)
    resultados = []

    for k in range(2, k_max + 1):

        # Seleccionar modelo
        modelos = {
            "Ward": AgglomerativeClustering(
                n_clusters=k, linkage="ward"
            ),

            "Liga completa": AgglomerativeClustering(
                n_clusters=k, linkage="complete"
            ),

            "K-means": KMeans(
                n_clusters=k, random_state=42, n_init=20
            ),

            "K-medoids": KMedoids(
                n_clusters=k, metric="euclidean",
                method="pam", random_state=42
            )
        }

        if metodo not in modelos:
            raise ValueError(f"Método no válido: {metodo}")

        # Ajustar modelo para este k
        modelo = modelos[metodo]
        etiquetas = modelo.fit_predict(X)

        # Métricas
        silhouette = silhouette_score(X, etiquetas)
        calinski = calinski_harabasz_score(X, etiquetas)

        # WCSS: suma de cuadrados respecto a los centroides
        wcss = 0

        for grupo in np.unique(etiquetas):
            puntos = X[etiquetas == grupo]
            centroide = puntos.mean(axis=0)

            wcss += ((puntos - centroide) ** 2).sum()

        # Tamaño de los grupos
        tamanos = np.bincount(etiquetas)
        tamanos = np.sort(tamanos)[::-1]

        # Guardar resultados
        resultados.append({
            "Número de clusters": k,
            "Silhouette": silhouette,
            "Calinski-Harabasz": calinski,
            "WCSS": wcss,
            "Tamaño de grupos": tamanos.tolist(),
            "Grupo mínimo": tamanos.min()
        })

    return pd.DataFrame(resultados).round(3)


# Siniestralidad y conglomerados
def agregar_siniestro_total(
    siniestros: pd.DataFrame,
    columna_salida: str = "siniestro_total_ajustada",
) -> pd.DataFrame:
    """Suma los cuatro desarrollos ajustados de cada registro."""
    columnas = [f"{desarrollo}_monto pagado_ajustada" for desarrollo in range(4)]
    faltantes = [col for col in columnas if col not in siniestros]
    if faltantes:
        raise KeyError(f"Columnas no encontradas: {faltantes}")
    out = siniestros.copy()
    out[columna_salida] = out[columnas].sum(axis=1, min_count=1)
    return out


def factor_siniestralidad_mercado(
    emision: pd.DataFrame,
    siniestros: pd.DataFrame,
    anios: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Calcula prima, monto pagado y factor anual del mercado."""
    columnas = [
        "año",
        "prima_emitida_ajustada",
    ]
    faltantes = [col for col in columnas if col not in emision]
    if faltantes:
        raise KeyError(f"Columnas no encontradas en emisión: {faltantes}")

    siniestros = agregar_siniestro_total(siniestros)
    prima = emision.groupby("año", dropna=False)["prima_emitida_ajustada"].sum()
    monto = siniestros.groupby("año", dropna=False)["siniestro_total_ajustada"].sum()

    if anios is not None:
        anios = list(anios)
        prima = prima.reindex(anios)
        monto = monto.reindex(anios)

    anual = pd.DataFrame({
        "prima_emitida": prima,
        "monto_siniestro": monto,
    }).fillna(0)
    denominador = anual["prima_emitida"].mask(anual["prima_emitida"] == 0, 1)
    anual["factor_siniestralidad"] = anual["monto_siniestro"] / denominador
    anual["variacion_anual"] = anual["factor_siniestralidad"].diff()
    anual["prima_cero"] = anual["prima_emitida"] == 0
    return anual.reset_index().rename(columns={"año": "Año"})


def factor_siniestralidad_tipo(
    emision: pd.DataFrame,
    siniestros: pd.DataFrame,
    anios: Iterable[int],
    categoria: str = "tipo_embarcacion",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Construye factores por año y tipo, además de activos y primas cero."""
    anios = list(anios)
    requeridas_emision = ["año", categoria, "prima_emitida_ajustada"]
    faltantes = [col for col in requeridas_emision if col not in emision]
    if faltantes:
        raise KeyError(f"Columnas no encontradas en emisión: {faltantes}")

    siniestros = agregar_siniestro_total(siniestros)
    requeridas_siniestros = ["año", categoria]
    faltantes = [col for col in requeridas_siniestros if col not in siniestros]
    if faltantes:
        raise KeyError(f"Columnas no encontradas en siniestros: {faltantes}")

    emision_filtrada = emision[emision["año"].isin(anios)]
    siniestros_filtrados = siniestros[siniestros["año"].isin(anios)]

    primas = (
        emision_filtrada
        .groupby(["año", categoria], dropna=False)["prima_emitida_ajustada"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=anios)
    )
    siniestros_tipo = (
        siniestros_filtrados
        .groupby(["año", categoria], dropna=False)["siniestro_total_ajustada"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=anios)
    )
    siniestros_tipo = siniestros_tipo.reindex(columns=primas.columns, fill_value=0)

    denominador = primas.mask(primas == 0, 1)
    factores = siniestros_tipo / denominador
    factores = factores.replace([np.inf, -np.inf], np.nan)

    activos = (primas > 0).sum(axis=1).rename("Tipos activos")
    prima_cero = primas == 0
    return factores, activos, prima_cero, primas


def estadisticas_factor_tipo(
    factores: pd.DataFrame,
    prima_cero: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume factores por tipo e identifica atípicos por IQR."""
    filas = []
    for tipo in factores.columns:
        serie = factores[tipo].dropna()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        mascara = (serie < limite_inferior) | (serie > limite_superior)

        primacero_anios = [
            int(anio)
            for anio in serie.index[mascara]
            if prima_cero.loc[anio, tipo]
        ] if tipo in prima_cero.columns else []

        filas.append({
            "Tipo": tipo,
            "Mínimo": serie.min(),
            "Máximo": serie.max(),
            "Media": serie.mean(),
            "Mediana": serie.median(),
            "IQR": iqr,
            "Atípicos": int(mascara.sum()),
            "Años atípicos": ", ".join(str(int(anio)) for anio in serie.index[mascara]),
            "Atípicos por prima cero": ", ".join(map(str, primacero_anios)) or "Ninguno",
        })

    estadisticas = pd.DataFrame(filas).set_index("Tipo")
    cambios = factores.diff(axis=0)
    return estadisticas, cambios, pd.DataFrame()


def graficar_factor_mercado(factor_mercado: pd.DataFrame):
    """Crea la gráfica lineal estándar del factor anual del mercado."""
    import plotly.graph_objects as go

    promedio = factor_mercado["factor_siniestralidad"].mean()
    fig = go.Figure()
    fig.add_scatter(
        x=factor_mercado["Año"],
        y=factor_mercado["factor_siniestralidad"],
        mode="lines+markers+text",
        text=factor_mercado["factor_siniestralidad"].round(2),
        textposition="top center",
        name="Factor de siniestralidad",
        line=dict(color="#5B8FF9", width=3),
    )
    fig.add_hline(
        y=promedio,
        line_dash="dot",
        line_color="#9C9C9C",
        annotation_text=f"Promedio {promedio:.2f}",
    )
    fig.update_layout(
        title="Factor de siniestralidad del mercado",
        template="reservas",
        height=500,
        xaxis=dict(title="Año", dtick=1),
        yaxis=dict(title="Siniestros / Prima emitida"),
    )
    return fig


def graficar_factores_tipo(
    factores: pd.DataFrame,
    prima_cero: pd.DataFrame,
):
    """Crea boxplot y heatmap de factores por tipo de embarcación."""
    import plotly.graph_objects as go

    # Las celdas con prima cero tienen factor por definición (pago / 1),
    # pero no son observaciones comparables y se excluyen de las visuales.
    factores_visuales = factores.mask(prima_cero)
    factores_largos = (
        factores_visuales.reset_index()
        .rename(columns={
            factores_visuales.index.name or "index": "Año",
        })
        .melt(id_vars="Año", var_name="Tipo", value_name="Factor")
        .dropna(subset=["Factor"])
    )

    box = go.Figure()
    factores_largos = factores_largos.dropna(subset=["Factor"])
    valores_positivos = factores_largos.loc[
        factores_largos["Factor"] > 0,
        "Factor",
    ]
    escala_log = (
        len(valores_positivos) > 0
        and valores_positivos.max() / max(valores_positivos.quantile(0.50), 1e-9) > 20
    )

    for tipo in factores.columns:
        serie = factores_largos[factores_largos["Tipo"] == tipo]
        serie = serie[serie["Factor"] > 0]
        if serie.empty:
            continue
        box.add_box(
            x=[tipo] * len(serie),
            y=serie["Factor"],
            name=tipo,
            boxpoints="all",
            jitter=0.25,
            pointpos=0,
            marker=dict(size=5, opacity=0.65, color="#F58518"),
            line=dict(color="#2A4163", width=1),
            showlegend=False,
        )
    box.update_layout(
        title="Factores comparables por tipo (sin años con prima cero)",
        template="reservas",
        height=650,
        xaxis=dict(title="Tipo de embarcación", tickangle=-45),
        yaxis=dict(
            title="Factor de siniestralidad",
            type="log" if escala_log else "linear",
            tickformat=".2f",
        ),
    )

    heatmap_valores = factores_visuales.clip(lower=0)
    vmax = float(heatmap_valores.quantile(0.95).quantile(0.95)) if not heatmap_valores.empty else 1
    vmax = max(vmax, 1.0)
    heat = go.Figure(go.Heatmap(
        z=heatmap_valores.T,
        x=heatmap_valores.index,
        y=heatmap_valores.columns,
        colorscale=[
            [0, "#FFFFFF"],
            [0.5, "#9ECAE1"],
            [1, "#08519C"],
        ],
        zmin=0,
        zmax=vmax,
        colorbar=dict(title="Factor"),
        hovertemplate="Tipo %{y}<br>Año %{x}<br>Factor %{z:.2f}<extra></extra>",
    ))
    heat.update_layout(
        title="Factor de siniestralidad por tipo y año",
        template="reservas",
        height=max(650, 18 * len(factores.columns)),
        xaxis=dict(title="Año", dtick=1),
        yaxis=dict(title="Tipo de embarcación"),
    )
    return box, heat


def preparar_perfil_conglomerados(
    factores: pd.DataFrame,
    prima_cero: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Winsoriza extremos, transforma y estandariza perfiles históricos."""
    factores = factores.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Winsorización por columna para limitar factores extremos derivados de
    # siniestros sin prima comparable, sin eliminar observaciones.
    robusto = factores.copy()
    for anio in robusto.index:
        serie = robusto.loc[anio]
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 3 * iqr
        limite_superior = q3 + 3 * iqr
        robusto.loc[anio] = serie.clip(limite_inferior, limite_superior)

    transformado = np.log1p(robusto.clip(lower=0)).T
    transformado.index.name = "tipo_embarcacion"
    perfiles = pd.DataFrame({
        "Nivel": transformado.mean(axis=1),
        "Variabilidad": transformado.std(axis=1),
    })

    escalado = StandardScaler().fit_transform(perfiles)
    perfiles_escalados = pd.DataFrame(
        escalado,
        index=perfiles.index,
        columns=["Nivel estandarizado", "Variabilidad estandarizada"],
    )
    return perfiles_escalados, perfiles


def evaluar_conglomerados(
    perfiles_escalados: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 5,
) -> pd.DataFrame:
    """Evalúa Ward por silueta, Calinski-Harabasz y WCSS."""
    X = perfiles_escalados.to_numpy()
    filas = []
    for k in range(k_min, k_max + 1):
        modelo = AgglomerativeClustering(n_clusters=k, linkage="ward")
        etiquetas = modelo.fit_predict(X)
        centroides = [
            X[etiquetas == grupo].mean(axis=0)
            for grupo in np.unique(etiquetas)
        ]
        wcss = sum(
            ((X[etiquetas == grupo] - centroide) ** 2).sum()
            for grupo, centroide in zip(np.unique(etiquetas), centroides)
        )
        filas.append({
            "k": k,
            "Silueta": silhouette_score(X, etiquetas),
            "Calinski-Harabasz": calinski_harabasz_score(X, etiquetas),
            "WCSS": wcss,
        })
    return pd.DataFrame(filas)


def asignar_conglomerados(
    perfiles_escalados: pd.DataFrame,
    perfiles: pd.DataFrame,
    k: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Asigna Ward, ordena grupos y genera perfil histórico."""
    modelo = AgglomerativeClustering(n_clusters=k, linkage="ward")
    etiquetas = modelo.fit_predict(perfiles_escalados)

    crudo = pd.Series(etiquetas, index=perfiles_escalados.index, name="Conglomerado")
    orden = (
        perfiles["Nivel"].groupby(crudo).median().sort_values().index
    )
    nombres = {grupo: f"G{posicion}" for posicion, grupo in enumerate(orden, 1)}
    asignacion = crudo.map(nombres).rename("Conglomerado")

    perfiles_finales = (
        perfiles.join(asignacion)
        .groupby("Conglomerado")
        .agg(
            Tipos=("Nivel", "size"),
            Nivel_promedio=("Nivel", "mean"),
            Nivel_mediano=("Nivel", "median"),
            Variabilidad_promedio=("Variabilidad", "mean"),
            Variabilidad_mediana=("Variabilidad", "median"),
        )
        .sort_index()
    )
    return asignacion, perfiles_finales




def factor_siniestralidad_conglomerado(
    factores: pd.DataFrame,
    primas: pd.DataFrame,
    asignacion: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega prima y siniestro por conglomerado y calcula su factor anual."""
    tipos = asignacion.index.intersection(factores.columns)
    factores = factores.reindex(columns=tipos)
    primas = primas.reindex(index=factores.index, columns=tipos)

    siniestros = factores * primas
    siniestro_agregado = pd.DataFrame(index=factores.index)
    prima_agregada = pd.DataFrame(index=factores.index)

    for grupo in sorted(asignacion.unique()):
        tipos_grupo = asignacion[asignacion == grupo].index
        tipos_grupo = tipos_grupo.intersection(factores.columns)
        prima_grupo = primas[tipos_grupo].sum(axis=1)
        siniestro_grupo = siniestros[tipos_grupo].sum(axis=1)
        prima_agregada[grupo] = prima_grupo
        siniestro_agregado[grupo] = siniestro_grupo

    factores_agregados = siniestro_agregado / prima_agregada.replace(0, 1)
    resumen = pd.DataFrame({
        "Prima agregada": prima_agregada.sum(axis=0),
        "Siniestro agregado": siniestro_agregado.sum(axis=0),
        "Factor agregado": (
            siniestro_agregado.sum(axis=0)
            / prima_agregada.replace(0, 1).sum(axis=0)
        ),
        "Factor anual medio": factores_agregados.mean(axis=0),
        "Factor anual mediano": factores_agregados.median(axis=0),
        "Volatilidad (DE)": factores_agregados.std(axis=0),
    })
    resumen.index.name = "Conglomerado"
    return factores_agregados, resumen


def graficar_factor_conglomerado(
    factores_agregados: pd.DataFrame,
):
    """Grafica la evolución anual y el promedio de cada conglomerado."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colores = ["#5B8FF9", "#F58518", "#54A24B", "#B279A2"]
    for i, grupo in enumerate(factores_agregados.columns):
        serie = factores_agregados[grupo]
        fig.add_scatter(
            x=serie.index,
            y=serie,
            mode="lines+markers+text",
            name=grupo,
            text=serie.round(2),
            textposition="top center",
            line=dict(width=3, color=colores[i % len(colores)]),
        )
    fig.update_layout(
        title="Factor anual de siniestralidad por conglomerado",
        template="reservas",
        height=520,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis=dict(title="Año", dtick=1),
        yaxis=dict(title="Siniestros agregados / Prima agregada"),
    )
    return fig


def graficar_evaluacion_conglomerados(
    evaluacion: pd.DataFrame,
):
    """Crea el gráfico del método del codo y silueta."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_bar(
        x=evaluacion["k"],
        y=evaluacion["Silueta"],
        name="Silueta",
        marker_color="#5B8FF9",
    )
    fig.update_layout(
        title="Selección de conglomerados",
        template="reservas",
        height=450,
        xaxis=dict(title="Número de conglomerados", dtick=1),
        yaxis=dict(title="Coeficiente de silueta"),
    )
    return fig


def graficar_dendrograma(
    perfiles_escalados: pd.DataFrame,
):
    """Crea un dendrograma Ward con etiquetas de tipo."""
    import plotly.figure_factory as ff

    z = linkage(perfiles_escalados.to_numpy(), method="ward")
    fig = ff.create_dendrogram(
        perfiles_escalados.to_numpy(),
        orientation="left",
        labels=perfiles_escalados.index.tolist(),
        linkagefun=lambda _: z,
        color_threshold=None,
    )
    fig.update_layout(
        title="Dendrograma de tipos de embarcación",
        template="reservas",
        height=max(520, 16 * len(perfiles_escalados)),
        xaxis=dict(title="Distancia Ward"),
        yaxis=dict(title="Tipo"),
    )
    return fig


def graficar_perfiles_conglomerados(
    perfiles: pd.DataFrame,
    asignacion: pd.Series,
):
    """Compara nivel y volatilidad histórica de los conglomerados."""
    import plotly.express as px

    datos = perfiles.join(asignacion).reset_index()
    fig = px.scatter(
        datos,
        x="Nivel",
        y="Variabilidad",
        color="Conglomerado",
        hover_name="tipo_embarcacion",
        title="Perfiles de siniestralidad por conglomerado",
        labels={
            "Nivel": "Nivel histórico",
            "Variabilidad": "Variabilidad histórica",
        },
        color_discrete_sequence=["#5B8FF9", "#F58518", "#54A24B", "#B279A2"],
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(template="reservas", height=520)
    return fig
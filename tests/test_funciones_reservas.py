import json

import pandas as pd
import pytest

from funciones_reservas import (
    calcular_rrc_exacto,
    calcular_rrc_veinticuatroavos,
    calcular_sonr_chain_ladder,
    construir_triangulos_desarrollo,
    guardar_snapshot,
    leer_snapshot,
    snapshot_es_valido,
)


def test_snapshot_se_guarda_se_lee_y_conserva_metadatos(tmp_path):
    origen = tmp_path / "origen.xlsx"
    origen.write_bytes(b"base sintetica")
    ruta = tmp_path / "procesado.parquet"
    esperado = pd.DataFrame({"año": [2025], "importe": [-10.0]})

    guardar_snapshot(
        esperado,
        ruta,
        periodo_valuacion="2025-12",
        fuente_inpc="serie sintetica",
        criterio_actualizacion="diciembre",
        archivos_origen=[origen],
    )

    pd.testing.assert_frame_equal(leer_snapshot(ruta), esperado)
    metadata = json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["periodo_valuacion"] == "2025-12"
    assert snapshot_es_valido(
        ruta,
        periodo_valuacion="2025-12",
        archivos_origen=[origen],
    )


def test_snapshot_invalido_si_cambia_la_fuente(tmp_path):
    origen = tmp_path / "origen.xlsx"
    origen.write_bytes(b"version 1")
    ruta = tmp_path / "procesado.parquet"
    guardar_snapshot(
        pd.DataFrame({"importe": [1]}),
        ruta,
        periodo_valuacion="2025-12",
        archivos_origen=[origen],
    )
    origen.write_bytes(b"version 2")
    assert not snapshot_es_valido(
        ruta,
        periodo_valuacion="2025-12",
        archivos_origen=[origen],
    )


def test_lectura_de_snapshot_inexistente_es_explicita(tmp_path):
    with pytest.raises(FileNotFoundError, match="No existe el snapshot"):
        leer_snapshot(tmp_path / "faltante.parquet")


def test_rrc_exacto_y_veinticuatroavos():
    df = pd.DataFrame({
        "inicio_vigencia": ["2025-07-02", "2025-01-01"],
        "fin_vigencia": ["2026-07-02", "2025-12-01"],
        "prima_tarifa": [365.0, 100.0],
    })
    res_ex = calcular_rrc_exacto(df, col_prima="prima_tarifa", fecha_valuacion="2025-12-31")
    # 2026-07-02 minus 2025-12-31 is 183 days out of 365 -> RRC = 183.0; second policy expired -> 0.0
    assert res_ex["rrc_exacto"].tolist() == pytest.approx([183.0, 0.0])

    res_24 = calcular_rrc_veinticuatroavos(df, col_prima="prima_tarifa")
    # July (m=7) -> 13/24; Jan (m=1) -> 1/24
    assert res_24["rrc_24avos"].tolist() == pytest.approx([365.0 * 13 / 24, 100.0 * 1 / 24])


def test_chain_ladder_triangulo_sintetico():
    df_sin = pd.DataFrame({
        "año": [2023, 2024, 2025],
        "0_monto pagado_ajustada": [100.0, 200.0, 300.0],
        "1_monto pagado_ajustada": [50.0, 100.0, float("nan")],
        "2_monto pagado_ajustada": [30.0, float("nan"), float("nan")],
    })
    tri_inc, tri_cum = construir_triangulos_desarrollo(
        df_sin,
        columnas_desarrollo=[f"{i}_monto pagado_ajustada" for i in range(3)],
    )
    res = calcular_sonr_chain_ladder(tri_cum, metodo_factores="ponderado")
    # f_0->1 = (150+300)/(100+200) = 1.5; f_1->2 = 180/150 = 1.2
    assert res["factores"] == pytest.approx([1.5, 1.2])
    # 2023: ult=180, pag=180 -> sonr=0
    # 2024: ult=300*1.2=360, pag=300 -> sonr=60
    # 2025: ult=300*1.5*1.2=540, pag=300 -> sonr=240
    assert res["sonr_total"] == pytest.approx(300.0)


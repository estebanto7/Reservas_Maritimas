import pandas as pd

from funciones_reservas import (
    estadisticas_factor_tipo,
    factor_siniestralidad_mercado,
    factor_siniestralidad_tipo,
)


def test_factor_mercado_sustituye_prima_cero_por_uno():
    emision = pd.DataFrame({
        "año": [2024, 2025],
        "prima_emitida_ajustada": [100.0, 0.0],
    })
    siniestros = pd.DataFrame({
        "año": [2024, 2025],
        "0_monto pagado_ajustada": [20.0, 50.0],
        "1_monto pagado_ajustada": [0.0, 0.0],
        "2_monto pagado_ajustada": [0.0, 0.0],
        "3_monto pagado_ajustada": [0.0, 0.0],
    })

    result = factor_siniestralidad_mercado(
        emision, siniestros, anios=[2024, 2025]
    )

    assert result.loc[0, "factor_siniestralidad"] == 0.2
    assert result.loc[1, "factor_siniestralidad"] == 50.0
    assert result.loc[1, "prima_cero"]


def test_factor_por_tipo_e_identifica_prima_cero():
    emision = pd.DataFrame({
        "año": [2024, 2024, 2025, 2025],
        "tipo_embarcacion": ["Lancha", "Yate", "Lancha", "Yate"],
        "prima_emitida_ajustada": [100.0, 50.0, 0.0, 100.0],
    })
    siniestros = pd.DataFrame({
        "año": [2024, 2024, 2025, 2025],
        "tipo_embarcacion": ["Lancha", "Yate", "Lancha", "Yate"],
        "0_monto pagado_ajustada": [10.0, 100.0, 30.0, 20.0],
        "1_monto pagado_ajustada": [0.0, 0.0, 0.0, 0.0],
        "2_monto pagado_ajustada": [0.0, 0.0, 0.0, 0.0],
        "3_monto pagado_ajustada": [0.0, 0.0, 0.0, 0.0],
    })

    factores, activos, prima_cero, _ = factor_siniestralidad_tipo(
        emision, siniestros, anios=[2024, 2025]
    )
    estadisticas, cambios, _ = estadisticas_factor_tipo(factores, prima_cero)

    assert factores.loc[2025, "Lancha"] == 30.0
    assert activos.loc[2025] == 1
    assert prima_cero.loc[2025, "Lancha"]
    assert "Lancha" in estadisticas.index



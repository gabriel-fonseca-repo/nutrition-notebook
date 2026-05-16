import math
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linprog  # pyright: ignore[reportUnknownVariableType]

from constants import (
    ATIVIDADE_ALIASES,
    DRI,
    IDADE_ADOLESCENTE_14,
    IDADE_ADULTO,
    IDADE_ADULTO_30,
    IDADE_ADULTO_50,
    IDADE_ADULTO_70,
    IDADE_AMINOACIDOS_ADOLESCENTE,
    IDADE_AMINOACIDOS_CRIANCA,
    IDADE_CRIANCA_4,
    IDADE_CRIANCA_9,
    IDADE_GESTANTE_JOVEM_MAX,
    IDADE_MINIMA_ESTIMADOR,
    IMC_GESTACIONAL_ALIASES,
    MESES_LACTACAO_1_ANO,
    MESES_LACTACAO_1_SEMESTRE,
    SEMANA_GESTACAO_ADICIONAL,
    SEXO_ALIASES,
    Atividade,
    Sexo,
)

type FaixaAmdr = tuple[float, float]
type LinhaNecessidade = dict[str, str | int | float | None]
type ArrayFloat = NDArray[np.float64]
type ResumoLp = dict[str, str | int | float]


class MetaLp(TypedDict):
    nutriente: str
    colunas_taco_usadas: str
    minimo_exigido: float | None
    maximo_permitido: float | None
    unidade: str
    vetor: ArrayFloat


COLUNAS_IDENTIFICACAO = [
    "Número do Alimento",
    "Categoria do Alimento",
    "Descrição dos Alimentos",
]
DIAS_POR_SEMANA = 7
DIAS_POR_MES = 30
LIMIAR_ARREDONDAR_5G = 100
LIMIAR_ARREDONDAR_1G = 20
LIMIAR_ARREDONDAR_05G = 5
LIMIAR_FORMATO_DIARIO = 80
LIMIAR_FORMATO_SEMANAL = 10
LIMIAR_PORCAO_DIARIA_ALTA = 300
LIMIAR_MICROQUANTIDADE = 10


def limpar_rotulo(coluna: str) -> str:
    mapa_rotulos = {
        "Categoria do alimento": "Categoria do Alimento",
        "Descrição dos alimentos": "Descrição dos Alimentos",
    }
    if coluna in mapa_rotulos:
        return mapa_rotulos[coluna]

    unidade = None
    match_unidade = re.search(r"\.\.([A-Za-z]+)\.$", coluna)
    if match_unidade:
        unidade = match_unidade.group(1)
        base = coluna[: match_unidade.start()]
    else:
        base = re.sub(r"\.+$", "", coluna)

    if base.startswith("X"):
        codigo = base[1:]
        match_codigo = re.match(r"^(\d+)\.(\d+)(.*)$", codigo)
        if match_codigo:
            resto = match_codigo.group(3).replace(".n.", " n-").replace(".", " ")
            base = f"{match_codigo.group(1)}:{match_codigo.group(2)}{resto}"
        else:
            base = codigo.replace(".", " ")
    else:
        base = base.replace(".", " ")

    base = " ".join(base.split())
    return f"{base} ({unidade})" if unidade else base


def _is_missing_scalar(valor: object) -> bool:
    if valor is None or type(valor).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(valor, Decimal):
        return valor.is_nan()
    if isinstance(valor, Real):
        return math.isnan(float(valor))
    return False


def formatar_numero_brasileiro(valor: object) -> object:
    if _is_missing_scalar(valor):
        return ""
    try:
        numero = Decimal(str(valor))
    except InvalidOperation, ValueError:
        return valor

    texto = format(numero, "f")
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def normalizar_sexo(sexo: object) -> Sexo:
    chave = str(sexo).strip().lower()
    if chave not in SEXO_ALIASES:
        raise ValueError("sexo deve ser masculino/feminino")
    return SEXO_ALIASES[chave]


def normalizar_atividade(atividade: object) -> Atividade:
    chave = str(atividade).strip().lower()
    if chave not in ATIVIDADE_ALIASES:
        raise ValueError("atividade deve ser sedentario, leve, moderado ou intenso")
    return ATIVIDADE_ALIASES[chave]


def estagio_vida(  # noqa: PLR0911, PLR0912
    sexo: Sexo,
    idade_anos: float,
    gestante: bool = False,
    meses_lactacao: float | None = None,
) -> str:
    if gestante:
        if sexo != "female":
            raise ValueError("gestante=True só é válido para sexo feminino")
        if idade_anos <= IDADE_GESTANTE_JOVEM_MAX:
            return "pregnancy_u18"
        if idade_anos <= IDADE_ADULTO_30:
            return "pregnancy_19_30"
        return "pregnancy_31_50"

    if meses_lactacao is not None:
        if sexo != "female":
            raise ValueError("meses_lactacao só é válido para sexo feminino")
        if idade_anos <= IDADE_GESTANTE_JOVEM_MAX:
            return "lactation_u18"
        if idade_anos <= IDADE_ADULTO_30:
            return "lactation_19_30"
        return "lactation_31_50"

    if idade_anos < IDADE_MINIMA_ESTIMADOR:
        raise ValueError("este estimador começa em 1 ano de idade")
    if idade_anos < IDADE_CRIANCA_4:
        return "child_1_3"
    if idade_anos < IDADE_CRIANCA_9:
        return "child_4_8"

    prefixo = "male" if sexo == "male" else "female"
    if idade_anos < IDADE_ADOLESCENTE_14:
        return f"{prefixo}_9_13"
    if idade_anos < IDADE_ADULTO:
        return f"{prefixo}_14_18"
    if idade_anos <= IDADE_ADULTO_30:
        return f"{prefixo}_19_30"
    if idade_anos <= IDADE_ADULTO_50:
        return f"{prefixo}_31_50"
    if idade_anos <= IDADE_ADULTO_70:
        return f"{prefixo}_51_70"
    return f"{prefixo}_70_plus"


def calcular_eer(  # noqa: PLR0911, PLR0912, PLR0913
    sexo: Sexo,
    idade_anos: float,
    altura_m: float,
    peso_kg: float,
    atividade: object,
    gestante: bool = False,
    semanas_gestacao: float | None = None,
    imc_pre_gestacional: object = "normal",
    meses_lactacao: float | None = None,
) -> float:
    """Estimated Energy Requirement, em kcal/dia."""
    altura_cm = altura_m * 100
    atividade = normalizar_atividade(atividade)

    if gestante:
        if sexo != "female":
            raise ValueError("gestante=True só é válido para sexo feminino")
        if semanas_gestacao is None:
            raise ValueError("informe semanas_gestacao para gestantes")
        if semanas_gestacao < SEMANA_GESTACAO_ADICIONAL:
            return calcular_eer(sexo, idade_anos, altura_m, peso_kg, atividade)

        imc_chave = IMC_GESTACIONAL_ALIASES[str(imc_pre_gestacional).strip().lower()]
        ajuste = {"baixo": 300, "normal": 200, "sobrepeso": 150, "obesidade": -50}[
            imc_chave
        ]
        formulas = {
            "sedentario": 1131.20
            - 2.04 * idade_anos
            + 0.34 * altura_cm
            + 12.15 * peso_kg
            + 9.16 * semanas_gestacao
            + ajuste,
            "leve": 693.35
            - 2.04 * idade_anos
            + 5.73 * altura_cm
            + 10.20 * peso_kg
            + 9.16 * semanas_gestacao
            + ajuste,
            "moderado": -223.84
            - 2.04 * idade_anos
            + 13.23 * altura_cm
            + 8.15 * peso_kg
            + 9.16 * semanas_gestacao
            + ajuste,
            "intenso": -779.72
            - 2.04 * idade_anos
            + 18.45 * altura_cm
            + 8.73 * peso_kg
            + 9.16 * semanas_gestacao
            + ajuste,
        }
        return formulas[atividade]

    if meses_lactacao is not None:
        base = calcular_eer(sexo, idade_anos, altura_m, peso_kg, atividade)
        if meses_lactacao <= MESES_LACTACAO_1_SEMESTRE:
            return base + 400
        if meses_lactacao <= MESES_LACTACAO_1_ANO:
            return base + 380
        return base

    if idade_anos >= IDADE_ADULTO:
        if sexo == "male":
            formulas = {
                "sedentario": 753.07
                - 10.83 * idade_anos
                + 6.50 * altura_cm
                + 14.10 * peso_kg,
                "leve": 581.47
                - 10.83 * idade_anos
                + 8.30 * altura_cm
                + 14.94 * peso_kg,
                "moderado": 1004.82
                - 10.83 * idade_anos
                + 6.52 * altura_cm
                + 15.91 * peso_kg,
                "intenso": -517.88
                - 10.83 * idade_anos
                + 15.61 * altura_cm
                + 19.11 * peso_kg,
            }
        else:
            formulas = {
                "sedentario": 584.90
                - 7.01 * idade_anos
                + 5.72 * altura_cm
                + 11.71 * peso_kg,
                "leve": 575.77 - 7.01 * idade_anos + 6.60 * altura_cm + 12.14 * peso_kg,
                "moderado": 710.25
                - 7.01 * idade_anos
                + 6.54 * altura_cm
                + 12.34 * peso_kg,
                "intenso": 511.83
                - 7.01 * idade_anos
                + 9.07 * altura_cm
                + 12.56 * peso_kg,
            }
        return formulas[atividade]

    if idade_anos >= IDADE_AMINOACIDOS_CRIANCA:
        if idade_anos < IDADE_CRIANCA_4:
            deposito = 20 if sexo == "male" else 15
        elif idade_anos < IDADE_CRIANCA_9:
            deposito = 15
        elif idade_anos < IDADE_ADOLESCENTE_14:
            deposito = 25 if sexo == "male" else 30
        else:
            deposito = 20

        if sexo == "male":
            formulas = {
                "sedentario": -447.51
                + 3.68 * idade_anos
                + 13.01 * altura_cm
                + 13.15 * peso_kg
                + deposito,
                "leve": 19.12
                + 3.68 * idade_anos
                + 8.62 * altura_cm
                + 20.28 * peso_kg
                + deposito,
                "moderado": -388.19
                + 3.68 * idade_anos
                + 12.66 * altura_cm
                + 20.46 * peso_kg
                + deposito,
                "intenso": -671.75
                + 3.68 * idade_anos
                + 15.38 * altura_cm
                + 23.25 * peso_kg
                + deposito,
            }
        else:
            formulas = {
                "sedentario": 55.59
                - 22.25 * idade_anos
                + 8.43 * altura_cm
                + 17.07 * peso_kg
                + deposito,
                "leve": -297.54
                - 22.25 * idade_anos
                + 12.77 * altura_cm
                + 14.73 * peso_kg
                + deposito,
                "moderado": -189.55
                - 22.25 * idade_anos
                + 11.74 * altura_cm
                + 18.34 * peso_kg
                + deposito,
                "intenso": -709.59
                - 22.25 * idade_anos
                + 18.22 * altura_cm
                + 14.25 * peso_kg
                + deposito,
            }
        return formulas[atividade]

    if sexo == "male":
        return -716.45 - idade_anos + 17.82 * altura_cm + 15.06 * peso_kg + 20
    return -69.15 + 80 * idade_anos + 2.65 * altura_cm + 54.15 * peso_kg + 15


def amdr(idade_anos: float, eer_kcal: float) -> dict[str, FaixaAmdr]:
    if idade_anos < IDADE_CRIANCA_4:
        carb_pct, protein_pct, fat_pct = (0.45, 0.65), (0.05, 0.20), (0.30, 0.40)
    elif idade_anos < IDADE_ADULTO:
        carb_pct, protein_pct, fat_pct = (0.45, 0.65), (0.10, 0.30), (0.25, 0.35)
    else:
        carb_pct, protein_pct, fat_pct = (0.45, 0.65), (0.10, 0.35), (0.20, 0.35)
    return {
        "carb_g": (eer_kcal * carb_pct[0] / 4, eer_kcal * carb_pct[1] / 4),
        "protein_g": (eer_kcal * protein_pct[0] / 4, eer_kcal * protein_pct[1] / 4),
        "fat_g": (eer_kcal * fat_pct[0] / 9, eer_kcal * fat_pct[1] / 9),
        "linoleic_g": (eer_kcal * 0.05 / 9, eer_kcal * 0.10 / 9),
        "ala_g": (eer_kcal * 0.006 / 9, eer_kcal * 0.012 / 9),
    }


def aminoacidos_mg_por_kg(idade_anos: float) -> dict[str, float]:
    if idade_anos < IDADE_AMINOACIDOS_CRIANCA:
        return {
            "Histidina": 15,
            "Isoleucina": 27,
            "Leucina": 54,
            "Lisina": 44,
            "Metionina + Cistina": 22,
            "Fenilalanina + Tirosina": 40,
            "Treonina": 24,
            "Triptofano": 6,
            "Valina": 36,
        }
    if idade_anos < IDADE_AMINOACIDOS_ADOLESCENTE:
        return {
            "Histidina": 12,
            "Isoleucina": 22,
            "Leucina": 44,
            "Lisina": 35,
            "Metionina + Cistina": 17,
            "Fenilalanina + Tirosina": 30,
            "Treonina": 18,
            "Triptofano": 4.8,
            "Valina": 29,
        }
    if idade_anos < IDADE_ADULTO:
        return {
            "Histidina": 11,
            "Isoleucina": 21,
            "Leucina": 42,
            "Lisina": 33,
            "Metionina + Cistina": 16,
            "Fenilalanina + Tirosina": 28,
            "Treonina": 17,
            "Triptofano": 4.4,
            "Valina": 28,
        }
    return {
        "Histidina": 10,
        "Isoleucina": 20,
        "Leucina": 39,
        "Lisina": 30,
        "Metionina + Cistina": 15,
        "Fenilalanina + Tirosina": 25,
        "Treonina": 15,
        "Triptofano": 4,
        "Valina": 26,
    }


def formatar_numero_exportacao(valor: object) -> object:
    if valor is None or _is_missing_scalar(valor):
        return ""
    try:
        numero = Decimal(str(valor))
    except InvalidOperation, ValueError:
        return valor
    texto = format(numero, "f")
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def calcular_necessidades(  # noqa: PLR0913
    sexo: object,
    idade_anos: int,
    altura_m: float,
    peso_kg: float,
    atividade: object = "moderado",
    fumante: bool = False,
    gestante: bool = False,
    semanas_gestacao: float | None = None,
    imc_pre_gestacional: object = "normal",
    meses_lactacao: float | None = None,
    colunas_taco: Sequence[str] | None = None,
) -> pd.DataFrame:
    sexo = normalizar_sexo(sexo)
    atividade = normalizar_atividade(atividade)
    estagio = estagio_vida(sexo, idade_anos, gestante, meses_lactacao)
    dri = DRI[estagio]
    eer = calcular_eer(
        sexo,
        idade_anos,
        altura_m,
        peso_kg,
        atividade,
        gestante=gestante,
        semanas_gestacao=semanas_gestacao,
        imc_pre_gestacional=imc_pre_gestacional,
        meses_lactacao=meses_lactacao,
    )
    faixas = amdr(idade_anos, eer)
    colunas_nao_nutrientes = {
        "Número do Alimento",
        "Categoria do Alimento",
        "Descrição dos Alimentos",
    }
    colunas_taco = list(colunas_taco or [])
    colunas_nutrientes = [c for c in colunas_taco if c not in colunas_nao_nutrientes]

    linhas: list[LinhaNecessidade] = []
    cobertas: set[str] = set()

    def add(  # noqa: PLR0913
        nutriente: str,
        unidade: str,
        alvo: str | int | float | None = None,
        minimo: str | int | float | None = None,
        maximo: str | int | float | None = None,
        tipo: str = "",
        base: str = "",
        observacoes: str = "",
        colunas_taco: Sequence[str] | None = None,
    ) -> None:
        colunas_usadas = list(colunas_taco or [])
        cobertas.update(colunas_usadas)
        linhas.append(
            {
                "Nutriente": nutriente,
                "Colunas TACO usadas": ", ".join(colunas_usadas),
                "Tipo": tipo,
                "Alvo": alvo,
                "Mínimo": minimo,
                "Máximo": maximo,
                "Unidade": unidade,
                "Base científica": base,
                "Observações": observacoes,
            }
        )

    add(
        "Energia",
        "kcal/dia",
        round(eer),
        tipo="EER",
        base="Equação NASEM 2023 por sexo, idade, altura, peso e atividade",
        colunas_taco=["Energia (kcal)"],
    )
    add(
        "Carboidrato",
        "g/dia",
        dri["carb"],
        round(faixas["carb_g"][0], 1),
        round(faixas["carb_g"][1], 1),
        "RDA + AMDR",
        "DRI: RDA e 45-65% da energia",
        colunas_taco=["Carboidrato (g)"],
    )
    proteina_alvo = max(dri["protein_ref"], dri["protein_kg"] * peso_kg)
    add(
        "Proteína",
        "g/dia",
        round(proteina_alvo, 1),
        round(faixas["protein_g"][0], 1),
        round(faixas["protein_g"][1], 1),
        "RDA por kg + AMDR",
        (
            f"{dri['protein_kg']} g/kg/dia, com mínimo de referência "
            f"{dri['protein_ref']} g/dia"
        ),
        colunas_taco=["Proteína (g)"],
    )
    add(
        "Lipídeos totais",
        "g/dia",
        None,
        round(faixas["fat_g"][0], 1),
        round(faixas["fat_g"][1], 1),
        "AMDR",
        "Percentual de energia vindo de gorduras totais",
        colunas_taco=["Lipídeos (g)"],
    )
    add(
        "Fibra Alimentar",
        "g/dia",
        round(eer * 14 / 1000, 1),
        tipo="AI estimada por energia",
        base="14 g/1000 kcal; tabela DRI também informa AI por estágio",
        observacoes=f"AI do estágio de vida na tabela: {dri['fiber']} g/dia",
        colunas_taco=["Fibra Alimentar (g)"],
    )
    add(
        "Ácidos graxos saturados",
        "g/dia",
        None,
        None,
        round(eer * 0.10 / 9, 1),
        "limite",
        "Diretrizes alimentares: <10% da energia; DRI: tão baixo quanto possível",
        colunas_taco=["Saturados (g)"],
    )
    add(
        "Ácidos graxos trans",
        "g/dia",
        0,
        0,
        0,
        "limite",
        "DRI: tão baixo quanto possível em dieta nutricionalmente adequada",
        colunas_taco=["18:1t (g)", "18:2t (g)"],
    )
    add(
        "Colesterol",
        "mg/dia",
        None,
        None,
        None,
        "sem RDA/AI",
        "DRI: tão baixo quanto possível; sem meta numérica individual",
        colunas_taco=["Colesterol (mg)"],
    )
    add(
        "Ácido linoleico n-6",
        "g/dia",
        dri["linoleic"],
        round(faixas["linoleic_g"][0], 1),
        round(faixas["linoleic_g"][1], 1),
        "AI + AMDR",
        "DRI para ácido graxo essencial n-6",
        colunas_taco=["18:2 n-6 (g)"],
    )
    add(
        "Ácido alfa-linolênico n-3",
        "g/dia",
        dri["ala"],
        round(faixas["ala_g"][0], 1),
        round(faixas["ala_g"][1], 1),
        "AI + AMDR",
        "DRI para ácido graxo essencial n-3",
        colunas_taco=["18:3 n-3 (g)"],
    )

    vitamina_c = dri["c"] + (35 if fumante else 0)
    add(
        "Vitamina A",
        "mcg RAE/dia",
        dri["a"],
        tipo="RDA/AI",
        base="DRI em Retinol Activity Equivalents",
        observacoes=(
            "Compare preferencialmente com RAE; Retinol e RE são "
            "formas/equivalências da tabela."
        ),
        colunas_taco=["RAE (mcg)", "RE (mcg)", "Retinol (mcg)"],
    )
    add(
        "Tiamina",
        "mg/dia",
        dri["b1"],
        tipo="RDA/AI",
        base="DRI",
        colunas_taco=["Tiamina (mg)"],
    )
    add(
        "Riboflavina",
        "mg/dia",
        dri["b2"],
        tipo="RDA/AI",
        base="DRI",
        colunas_taco=["Riboflavina (mg)"],
    )
    add(
        "Piridoxina / Vitamina B6",
        "mg/dia",
        dri["b6"],
        tipo="RDA/AI",
        base="DRI",
        colunas_taco=["Piridoxina (mg)"],
    )
    add(
        "Niacina",
        "mg NE/dia",
        dri["b3"],
        tipo="RDA/AI",
        base="DRI em equivalentes de niacina",
        colunas_taco=["Niacina (mg)"],
    )
    add(
        "Vitamina C",
        "mg/dia",
        vitamina_c,
        tipo="RDA/AI",
        base="DRI",
        observacoes="+35 mg/dia se fumante." if fumante else "",
        colunas_taco=["Vitamina C (mg)"],
    )

    add(
        "Cálcio",
        "mg/dia",
        dri["calcium"],
        tipo="RDA/AI",
        base="DRI",
        colunas_taco=["Cálcio (mg)"],
    )
    add(
        "Magnésio",
        "mg/dia",
        dri["magnesium"],
        tipo="RDA/AI",
        base="DRI",
        colunas_taco=["Magnésio (mg)"],
    )
    add(
        "Manganês",
        "mg/dia",
        dri["manganese"],
        tipo="AI",
        base="DRI",
        colunas_taco=["Manganês (mg)"],
    )
    add(
        "Fósforo",
        "mg/dia",
        dri["phosphorus"],
        tipo="RDA/AI",
        base="DRI",
        colunas_taco=["Fósforo (mg)"],
    )
    add(
        "Ferro",
        "mg/dia",
        dri["iron"],
        tipo="RDA",
        base="DRI",
        observacoes="Necessidade pode ser 1,8x maior em dieta vegetariana estrita.",
        colunas_taco=["Ferro (mg)"],
    )
    add(
        "Sódio",
        "mg/dia",
        dri["sodium"],
        None,
        dri["sodium_cdrr"],
        "AI + CDRR",
        "NASEM 2019: AI e limite de redução de risco crônico",
        colunas_taco=["Sódio (mg)"],
    )
    add(
        "Potássio",
        "mg/dia",
        dri["potassium"],
        tipo="AI",
        base="NASEM 2019",
        colunas_taco=["Potássio (mg)"],
    )
    add(
        "Cobre",
        "mg/dia",
        dri["copper"],
        tipo="RDA/AI",
        base="DRI; convertido de mcg para mg",
        colunas_taco=["Cobre (mg)"],
    )
    add(
        "Zinco",
        "mg/dia",
        dri["zinc"],
        tipo="RDA",
        base="DRI",
        observacoes=(
            "Necessidade pode ser até 50% maior em dietas vegetarianas "
            "estritas ricas em fitato."
        ),
        colunas_taco=["Zinco (mg)"],
    )

    amino_cols = {
        "Histidina": ["Histidina (g)"],
        "Isoleucina": ["Isoleucina (g)"],
        "Leucina": ["Leucina (g)"],
        "Lisina": ["Lisina (g)"],
        "Metionina + Cistina": ["Metionina (g)", "Cistina (g)"],
        "Fenilalanina + Tirosina": ["Fenilalanina (g)", "Tirosina (g)"],
        "Treonina": ["Treonina (g)"],
        "Triptofano": ["Triptofano (g)"],
        "Valina": ["Valina (g)"],
    }
    for amino, mg_kg in aminoacidos_mg_por_kg(idade_anos).items():
        add(
            amino,
            "g/dia",
            round(mg_kg * peso_kg / 1000, 3),
            tipo="requisito por kg",
            base=f"{mg_kg} mg/kg/dia",
            observacoes="Use a soma das colunas TACO quando o alvo for combinado."
            if "+" in amino
            else "",
            colunas_taco=amino_cols[amino],
        )

    for coluna in colunas_nutrientes:
        if coluna not in cobertas:
            add(
                coluna,
                "",
                None,
                None,
                None,
                "sem DRI individual",
                "Sem RDA/AI individual estabelecida para esta coluna isolada",
                observacoes=(
                    "Use como dado de composição alimentar, não como meta "
                    "diária isolada."
                ),
                colunas_taco=[coluna],
            )

    resultado = pd.DataFrame(linhas)
    resultado.insert(0, "Estágio de vida", estagio)
    resultado.insert(1, "EER usado (kcal/dia)", round(eer))
    return resultado


def converter_coluna_numerica_lp(serie: pd.Series) -> pd.Series:
    serie_any = cast(Any, serie)
    texto = cast(pd.Series, serie_any.astype(str).str.strip())
    texto_any = cast(Any, texto)
    texto = cast(
        pd.Series,
        texto_any.replace({"": pd.NA, "NA": pd.NA, "nan": pd.NA, "None": pd.NA}),
    )
    texto_any = cast(Any, texto)
    tem_decimal_brasileiro = cast(
        bool, texto_any.str.contains(",", regex=False, na=False).any()
    )
    if tem_decimal_brasileiro:
        texto = cast(
            pd.Series,
            texto_any.str.replace(".", "", regex=False).str.replace(
                ",", ".", regex=False
            ),
        )
    to_numeric: Any = pd.to_numeric  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    numeros = cast(pd.Series, to_numeric(texto, errors="coerce"))
    return cast(pd.Series, cast(Any, numeros).fillna(0))


def numero_lp(valor: object) -> float | None:
    if _is_missing_scalar(valor):
        return None
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return None
        texto = texto.replace(".", "").replace(",", ".") if "," in texto else texto
        return float(texto)
    try:
        return float(cast(Any, valor))
    except TypeError, ValueError:
        return None


def preparar_taco_para_lp(tabela_base: pd.DataFrame) -> pd.DataFrame:
    taco = tabela_base.copy()
    colunas = [
        limpar_rotulo(str(coluna))
        for coluna in cast(Sequence[object], tabela_base.columns)
    ]
    taco.columns = colunas

    for coluna in colunas:
        if coluna not in COLUNAS_IDENTIFICACAO:
            taco[coluna] = converter_coluna_numerica_lp(
                cast(pd.Series, cast(Any, taco)[coluna])
            )

    return taco


def filtrar_candidatos_lp(
    taco: pd.DataFrame,
    categorias_permitidas: Sequence[str] | None = None,
    termos_excluidos: Sequence[str] | list[str] | None = None,
) -> pd.DataFrame:
    candidatos = taco.copy()
    mascara = cast(pd.Series, candidatos["Energia (kcal)"] > 0)

    if categorias_permitidas:
        categorias = cast(Any, candidatos["Categoria do Alimento"])
        mascara = cast(
            pd.Series,
            cast(Any, mascara) & categorias.isin(categorias_permitidas),
        )

    descricoes_coluna = cast(Any, candidatos["Descrição dos Alimentos"])
    descricoes = cast(
        pd.Series,
        descricoes_coluna.fillna("").str.lower(),
    )
    descricoes_any = cast(Any, descricoes)
    for termo in termos_excluidos or []:
        mascara = cast(
            pd.Series,
            cast(Any, mascara)
            & ~descricoes_any.str.contains(termo.lower(), regex=False),
        )

    return cast(pd.DataFrame, cast(Any, candidatos).loc[mascara].reset_index(drop=True))


def vetor_nutriente_lp(
    candidatos: pd.DataFrame,
    nutriente: str,
    colunas: Sequence[str],
) -> tuple[ArrayFloat | None, list[str]]:
    # RAE, RE and Retinol are alternative vitamin-A measures, not additive targets.
    if nutriente == "Vitamina A":
        for coluna_preferida in ["RAE (mcg)", "RE (mcg)", "Retinol (mcg)"]:
            if coluna_preferida not in candidatos.columns:
                continue
            serie_preferida = cast(pd.Series, cast(Any, candidatos)[coluna_preferida])
            serie_preferida_any = cast(Any, serie_preferida)
            maior_valor = numero_lp(serie_preferida_any.max()) or 0
            if maior_valor > 0:
                vetor = cast(ArrayFloat, serie_preferida_any.to_numpy(dtype=float))
                return vetor, [coluna_preferida]

    colunas_existentes = [coluna for coluna in colunas if coluna in candidatos.columns]
    if not colunas_existentes:
        return None, []

    soma = cast(pd.Series, cast(Any, candidatos[colunas_existentes]).sum(axis=1))
    vetor = cast(ArrayFloat, cast(Any, soma).to_numpy(dtype=float))
    return vetor, colunas_existentes


def montar_modelo_lp(
    candidatos: pd.DataFrame,
    necessidades: pd.DataFrame,
    tolerancia_energia_acima: float = 0.05,
) -> tuple[ArrayFloat, ArrayFloat, list[MetaLp]]:
    a_ub: list[ArrayFloat] = []
    b_ub: list[float] = []
    metas: list[MetaLp] = []

    linhas = cast(list[dict[str, object]], cast(Any, necessidades).to_dict("records"))
    for linha in linhas:
        tipo = str(linha.get("Tipo", ""))
        maximo = numero_lp(linha.get("Máximo"))

        if tipo in {"sem DRI individual", "sem RDA/AI"} and maximo is None:
            continue

        colunas = [
            coluna.strip()
            for coluna in str(linha.get("Colunas TACO usadas", "")).split(",")
            if coluna.strip()
        ]
        nutriente = str(linha["Nutriente"])
        vetor, colunas_usadas = vetor_nutriente_lp(candidatos, nutriente, colunas)
        if vetor is None or not colunas_usadas:
            continue

        alvo = numero_lp(linha.get("Alvo"))
        minimo = numero_lp(linha.get("Mínimo"))
        lower_candidates = [valor for valor in [alvo, minimo] if valor is not None]
        minimo_exigido = max(lower_candidates) if lower_candidates else None
        maximo_permitido = maximo

        if nutriente == "Energia":
            if alvo is None:
                continue
            minimo_exigido = alvo
            maximo_permitido = alvo * (1 + tolerancia_energia_acima)

        if nutriente == "Colesterol":
            minimo_exigido = None

        if minimo_exigido is None and maximo_permitido is None:
            continue

        if minimo_exigido is not None and minimo_exigido > 0:
            a_ub.append(-vetor)
            b_ub.append(-minimo_exigido)

        if maximo_permitido is not None and maximo_permitido >= 0:
            a_ub.append(vetor)
            b_ub.append(maximo_permitido)

        metas.append(
            {
                "nutriente": nutriente,
                "colunas_taco_usadas": ", ".join(colunas_usadas),
                "minimo_exigido": minimo_exigido,
                "maximo_permitido": maximo_permitido,
                "unidade": str(linha.get("Unidade", "")),
                "vetor": vetor,
            }
        )

    return np.array(a_ub, dtype=float), np.array(b_ub, dtype=float), metas


def avaliar_cobertura_lp(
    metas: Sequence[MetaLp],
    porcoes_100g: ArrayFloat,
) -> pd.DataFrame:
    linhas: list[LinhaNecessidade | dict[str, bool | float | str | None]] = []
    for meta in metas:
        consumo = float(np.dot(meta["vetor"], porcoes_100g))
        minimo = meta["minimo_exigido"]
        maximo = meta["maximo_permitido"]
        atende_minimo = minimo is None or consumo + 1e-7 >= minimo
        atende_maximo = maximo is None or consumo <= maximo + 1e-7
        linhas.append(
            {
                "Nutriente": meta["nutriente"],
                "Colunas TACO usadas": meta["colunas_taco_usadas"],
                "Consumo estimado": consumo,
                "Mínimo exigido": minimo,
                "Máximo permitido": maximo,
                "Unidade": meta["unidade"],
                "Atendeu": atende_minimo and atende_maximo,
            }
        )
    return pd.DataFrame(linhas)


def otimizar_dieta_lp(  # noqa: PLR0913
    tabela_base: pd.DataFrame,
    necessidades: pd.DataFrame,
    max_gramas_por_alimento: float | None = 500,
    tolerancia_energia_acima: float = 0.05,
    categorias_permitidas: Sequence[str] | None = None,
    termos_excluidos: Sequence[str] | list[str] | None = None,
    min_gramas_para_exibir: float = 0.1,
) -> tuple[ResumoLp, pd.DataFrame, pd.DataFrame]:
    taco_lp = preparar_taco_para_lp(tabela_base)
    candidatos = filtrar_candidatos_lp(
        taco_lp,
        categorias_permitidas=categorias_permitidas,
        termos_excluidos=termos_excluidos,
    )

    a_ub, b_ub, metas = montar_modelo_lp(
        candidatos,
        necessidades,
        tolerancia_energia_acima=tolerancia_energia_acima,
    )

    custo = np.ones(len(candidatos), dtype=float) * 100
    limite_superior = (
        None if max_gramas_por_alimento is None else max_gramas_por_alimento / 100
    )
    limites = [(0, limite_superior) for _ in range(len(candidatos))]

    resultado: Any = linprog(
        c=custo,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=limites,
        method="highs",
    )

    if not resultado.success:
        restricoes = pd.DataFrame(
            [
                {
                    "Nutriente": meta["nutriente"],
                    "Colunas TACO usadas": meta["colunas_taco_usadas"],
                    "Mínimo exigido": meta["minimo_exigido"],
                    "Máximo permitido": meta["maximo_permitido"],
                    "Unidade": meta["unidade"],
                }
                for meta in metas
            ]
        )
        raise RuntimeError(
            "O problema de programação linear não encontrou solução: "
            f"{resultado.message}\n"
            "Tente aumentar MAX_GRAMAS_POR_ALIMENTO, relaxar "
            "TOLERANCIA_ENERGIA_ACIMA ou remover algumas restrições/categorias.\n"
            f"Restrições montadas: {len(restricoes)}"
        )

    porcoes = cast(ArrayFloat, resultado.x)
    gramas = porcoes * 100
    selecionados = gramas >= min_gramas_para_exibir

    dieta = cast(
        pd.DataFrame,
        cast(Any, candidatos).loc[selecionados, COLUNAS_IDENTIFICACAO].copy(),
    )
    dieta.insert(3, "Quantidade (g)", gramas[selecionados])
    dieta.insert(4, "Porções de 100 g", porcoes[selecionados])

    for coluna in [
        "Energia (kcal)",
        "Proteína (g)",
        "Carboidrato (g)",
        "Lipídeos (g)",
        "Fibra Alimentar (g)",
        "Sódio (mg)",
    ]:
        if coluna in candidatos.columns:
            valores = cast(
                ArrayFloat,
                cast(Any, candidatos).loc[selecionados, coluna].to_numpy(dtype=float),
            )
            dieta[f"{coluna} no plano"] = valores * porcoes[selecionados]

    dieta = dieta.sort_values("Quantidade (g)", ascending=False).reset_index(drop=True)
    cobertura = avaliar_cobertura_lp(metas, porcoes)

    resumo: ResumoLp = {
        "status": str(resultado.message),
        "total_gramas": float(gramas.sum()),
        "alimentos_usados": int(selecionados.sum()),
        "candidatos_avaliados": len(candidatos),
        "restricoes_ativas": len(metas),
    }

    return resumo, dieta, cobertura


def numero_float_legivel(valor: object) -> float:
    if _is_missing_scalar(valor):
        return 0.0
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return 0.0
        texto = texto.replace(".", "").replace(",", ".") if "," in texto else texto
        return float(texto)
    try:
        return float(cast(Any, valor))
    except TypeError, ValueError:
        return 0.0


def numero_float_ou_nan(valor: object) -> float:
    if _is_missing_scalar(valor):
        return float("nan")
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return float("nan")
        texto = texto.replace(".", "").replace(",", ".") if "," in texto else texto
        return float(texto)
    try:
        return float(cast(Any, valor))
    except TypeError, ValueError:
        return float("nan")


def arredondar_pratico(valor: object) -> float:
    valor = float(cast(Any, valor))
    if valor >= LIMIAR_ARREDONDAR_5G:
        base = 5
    elif valor >= LIMIAR_ARREDONDAR_1G:
        base = 1
    elif valor >= LIMIAR_ARREDONDAR_05G:
        base = 0.5
    else:
        base = 0.1
    return round(valor / base) * base


def formatar_quantidade(valor: object, unidade: str = "g") -> str:
    valor = arredondar_pratico(valor)
    texto = formatar_numero_exportacao(valor)
    return f"{texto} {unidade}"


def formato_sugerido(gramas_dia: object) -> str:
    gramas_dia = float(cast(Any, gramas_dia))
    gramas_semana = gramas_dia * DIAS_POR_SEMANA
    gramas_mes = gramas_dia * DIAS_POR_MES

    if gramas_dia >= LIMIAR_FORMATO_DIARIO:
        return f"{formatar_quantidade(gramas_dia)} por dia"
    if gramas_dia >= LIMIAR_FORMATO_SEMANAL:
        return (
            f"{formatar_quantidade(gramas_semana)} por semana "
            f"(~{formatar_quantidade(gramas_dia)}/dia)"
        )
    if gramas_semana >= LIMIAR_FORMATO_SEMANAL:
        return f"{formatar_quantidade(gramas_semana)} por semana"
    return f"{formatar_quantidade(gramas_mes)} por mês"


def observacao_pratica(descricao: object, gramas_dia: float) -> str:
    descricao_normalizada = str(descricao).lower()
    alertas: list[str] = []
    if gramas_dia >= LIMIAR_PORCAO_DIARIA_ALTA:
        alertas.append("porção diária alta")
    if gramas_dia < LIMIAR_MICROQUANTIDADE:
        alertas.append("microquantidade; mais fácil planejar por semana")
    if any(termo in descricao_normalizada for termo in ["cru", "pó", "semente"]):
        alertas.append("peso da TACO pode mudar após preparo")
    if any(
        termo in descricao_normalizada for termo in ["doce", "marmelada", "gelatina"]
    ):
        alertas.append(
            "item denso em açúcar; revisar se quiser um cardápio mais realista"
        )
    return "; ".join(alertas)


def preparar_plano_humano(dieta: pd.DataFrame) -> pd.DataFrame:
    plano = dieta.copy()
    plano_any = cast(Any, plano)
    quantidade_diaria_exata = cast(
        pd.Series, plano_any["Quantidade (g)"].map(numero_float_legivel)
    )
    quantidade_diaria_any = cast(Any, quantidade_diaria_exata)
    plano_any["Quantidade diária (g)"] = quantidade_diaria_any.map(arredondar_pratico)
    plano_any["Quantidade semanal (g)"] = (
        quantidade_diaria_exata * DIAS_POR_SEMANA
    ).map(arredondar_pratico)
    plano_any["Quantidade mensal (g)"] = (quantidade_diaria_exata * DIAS_POR_MES).map(
        arredondar_pratico
    )
    plano_any["Formato sugerido"] = quantidade_diaria_any.map(formato_sugerido)

    def observacao_linha(linha: Any) -> str:
        return observacao_pratica(
            linha["Descrição dos Alimentos"],
            numero_float_legivel(linha["Quantidade diária (g)"]),
        )

    plano_any["Observação prática"] = plano_any.apply(observacao_linha, axis=1)

    arredondamentos = {
        "Energia (kcal) no plano": 0,
        "Proteína (g) no plano": 1,
        "Carboidrato (g) no plano": 1,
        "Lipídeos (g) no plano": 1,
        "Fibra Alimentar (g) no plano": 1,
        "Sódio (mg) no plano": 0,
    }
    for coluna, casas_decimais in arredondamentos.items():
        if coluna in plano_any.columns:
            plano_any[coluna] = (
                plano_any[coluna].map(numero_float_legivel).round(casas_decimais)
            )

    colunas_resultado = [
        "Descrição dos Alimentos",
        "Categoria do Alimento",
        "Formato sugerido",
        "Quantidade diária (g)",
        "Quantidade semanal (g)",
        "Quantidade mensal (g)",
        "Energia (kcal) no plano",
        "Proteína (g) no plano",
        "Carboidrato (g) no plano",
        "Lipídeos (g) no plano",
        "Fibra Alimentar (g) no plano",
        "Sódio (mg) no plano",
        "Observação prática",
    ]
    colunas_resultado = [
        coluna for coluna in colunas_resultado if coluna in plano_any.columns
    ]
    return cast(
        pd.DataFrame,
        plano_any[colunas_resultado]
        .sort_values("Quantidade diária (g)", ascending=False)
        .reset_index(drop=True),
    )


def preparar_resumo_categorias(plano: pd.DataFrame) -> pd.DataFrame:
    plano_any = cast(Any, plano)
    resumo = (
        plano_any.groupby("Categoria do Alimento", as_index=False)
        .agg(
            Alimentos=("Descrição dos Alimentos", "count"),
            **{
                "Total diário (g)": ("Quantidade diária (g)", "sum"),
                "Total semanal (g)": ("Quantidade semanal (g)", "sum"),
                "Energia diária (kcal)": ("Energia (kcal) no plano", "sum"),
            },
        )
        .sort_values("Total diário (g)", ascending=False)
        .reset_index(drop=True)
    )
    return cast(pd.DataFrame, resumo)


def preparar_cobertura_humana(cobertura: pd.DataFrame) -> pd.DataFrame:
    cobertura_humana = cobertura.copy()
    cobertura_any = cast(Any, cobertura_humana)
    for coluna in ["Consumo estimado", "Mínimo exigido", "Máximo permitido"]:
        cobertura_any[coluna] = cobertura_any[coluna].map(numero_float_ou_nan)

    minimo = cobertura_any["Mínimo exigido"].replace(0, pd.NA)
    maximo = cobertura_any["Máximo permitido"].replace(0, pd.NA)
    cobertura_any["% do mínimo"] = cobertura_any["Consumo estimado"] / minimo * 100
    cobertura_any["% do teto"] = cobertura_any["Consumo estimado"] / maximo * 100
    for coluna in [
        "Consumo estimado",
        "Mínimo exigido",
        "Máximo permitido",
        "% do mínimo",
        "% do teto",
    ]:
        to_numeric: Any = pd.to_numeric  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        cobertura_any[coluna] = to_numeric(
            cobertura_any[coluna], errors="coerce"
        ).round(1)
    cobertura_any["Status"] = cobertura_any["Atendeu"].map(
        {True: "OK", False: "Revisar"}
    )

    return cast(
        pd.DataFrame,
        cobertura_any[
            [
                "Nutriente",
                "Consumo estimado",
                "Mínimo exigido",
                "Máximo permitido",
                "% do mínimo",
                "% do teto",
                "Unidade",
                "Status",
            ]
        ],
    )


def formatar_tabela_exportacao(tabela: pd.DataFrame) -> pd.DataFrame:
    exportacao = tabela.copy()
    exportacao_any = cast(Any, exportacao)
    for coluna in exportacao_any.select_dtypes(include="number").columns:
        exportacao_any[coluna] = exportacao_any[coluna].map(formatar_numero_exportacao)
    return exportacao


def faixas_amdr_por_calorias(
    idade_anos: float,
    calorias: float,
) -> dict[str, FaixaAmdr]:
    if idade_anos < IDADE_CRIANCA_4:
        carb_pct, protein_pct, fat_pct = (0.45, 0.65), (0.05, 0.20), (0.30, 0.40)
    elif idade_anos < IDADE_ADULTO:
        carb_pct, protein_pct, fat_pct = (0.45, 0.65), (0.10, 0.30), (0.25, 0.35)
    else:
        carb_pct, protein_pct, fat_pct = (0.45, 0.65), (0.10, 0.35), (0.20, 0.35)

    return {
        "carboidrato": (calorias * carb_pct[0] / 4, calorias * carb_pct[1] / 4),
        "proteina": (calorias * protein_pct[0] / 4, calorias * protein_pct[1] / 4),
        "lipideos": (calorias * fat_pct[0] / 9, calorias * fat_pct[1] / 9),
        "linoleico": (calorias * 0.05 / 9, calorias * 0.10 / 9),
        "ala": (calorias * 0.006 / 9, calorias * 0.012 / 9),
    }


def definir_valores_nutriente(
    tabela: pd.DataFrame,
    nutriente: str,
    alvo: float | None = None,
    minimo: float | None = None,
    maximo: float | None = None,
) -> None:
    tabela_any = cast(Any, tabela)
    mascara = tabela_any["Nutriente"].eq(nutriente)
    if alvo is not None:
        tabela_any.loc[mascara, "Alvo"] = alvo
    if minimo is not None:
        tabela_any.loc[mascara, "Mínimo"] = minimo
    if maximo is not None:
        tabela_any.loc[mascara, "Máximo"] = maximo


def aplicar_meta_calorica_para_lp(
    necessidades: pd.DataFrame,
    idade_anos: float,
    deficit_kcal: float = 500,
    meta_calorica_kcal: float | None = None,
    tolerancia_abaixo: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    ajustadas = necessidades.copy()
    ajustadas_any = cast(Any, ajustadas)
    eer = float(ajustadas_any["EER usado (kcal/dia)"].iloc[0])
    meta = meta_calorica_kcal if meta_calorica_kcal is not None else eer - deficit_kcal
    if meta <= 0:
        raise ValueError("A meta calórica precisa ser maior que zero.")

    calorias_minimas = meta * (1 - tolerancia_abaixo)
    tolerancia_acima_para_lp = (meta / calorias_minimas) - 1
    faixas = faixas_amdr_por_calorias(idade_anos, meta)

    ajustadas_any["Meta calórica para LP (kcal/dia)"] = meta
    ajustadas_any["Déficit aplicado (kcal/dia)"] = eer - meta

    # The current LP builder uses the Energia row's Alvo as the lower bound and
    # Alvo * (1 + tolerancia_energia_acima) as the upper bound.
    definir_valores_nutriente(
        ajustadas,
        "Energia",
        alvo=calorias_minimas,
        minimo=calorias_minimas,
        maximo=meta,
    )
    definir_valores_nutriente(
        ajustadas,
        "Carboidrato",
        minimo=faixas["carboidrato"][0],
        maximo=faixas["carboidrato"][1],
    )
    definir_valores_nutriente(
        ajustadas,
        "Proteína",
        minimo=faixas["proteina"][0],
        maximo=faixas["proteina"][1],
    )
    definir_valores_nutriente(
        ajustadas,
        "Lipídeos totais",
        minimo=faixas["lipideos"][0],
        maximo=faixas["lipideos"][1],
    )
    definir_valores_nutriente(
        ajustadas,
        "Fibra Alimentar",
        alvo=meta * 14 / 1000,
        minimo=meta * 14 / 1000,
    )
    definir_valores_nutriente(
        ajustadas,
        "Ácidos graxos saturados",
        maximo=meta * 0.10 / 9,
    )
    definir_valores_nutriente(
        ajustadas,
        "Ácido linoleico n-6",
        minimo=faixas["linoleico"][0],
        maximo=faixas["linoleico"][1],
    )
    definir_valores_nutriente(
        ajustadas,
        "Ácido alfa-linolênico n-3",
        minimo=faixas["ala"][0],
        maximo=faixas["ala"][1],
    )

    resumo = {
        "EER original (kcal/dia)": eer,
        "Meta calórica (kcal/dia)": meta,
        "Déficit (kcal/dia)": eer - meta,
        "Mínimo energético LP (kcal/dia)": calorias_minimas,
        "Máximo energético LP (kcal/dia)": meta,
        "Tolerância superior enviada ao LP": tolerancia_acima_para_lp,
    }
    return ajustadas, resumo, tolerancia_acima_para_lp

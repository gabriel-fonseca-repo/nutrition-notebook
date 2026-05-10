import math
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from numbers import Real

import pandas as pd

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
    idade_anos: float,
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

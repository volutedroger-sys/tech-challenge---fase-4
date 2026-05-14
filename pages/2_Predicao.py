"""
Sistema Preditivo — Visão de Negócio / Suporte à Decisão Clínica.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Predicao | Obesidade", page_icon="🔬", layout="wide")

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "modelo_obesidade.pkl"

# ── Mapeamentos de negócio ─────────────────────────────────────────────────────
RISCO = {
    "Insufficient_Weight": ("Abaixo do Peso",     "info",    "#4FC3F7", "Baixo"),
    "Normal_Weight":        ("Peso Normal",         "success", "#81C784", "Baixo"),
    "Overweight_Level_I":   ("Sobrepeso Grau I",    "warning", "#FFD54F", "Moderado"),
    "Overweight_Level_II":  ("Sobrepeso Grau II",   "warning", "#FFB74D", "Moderado"),
    "Obesity_Type_I":       ("Obesidade Grau I",    "error",   "#FF8A65", "Alto"),
    "Obesity_Type_II":      ("Obesidade Grau II",   "error",   "#E57373", "Alto"),
    "Obesity_Type_III":     ("Obesidade Grau III",  "error",   "#B71C1C", "Critico"),
}

CONDUTA = {
    "Insufficient_Weight": {
        "urgencia": "Eletiva",
        "encaminhamento": "Nutricao",
        "acoes": [
            "Avaliacao nutricional para ganho de peso saudavel",
            "Investigar causas subjacentes (disturbios alimentares, patologias)",
            "Monitoramento trimestral de peso e composicao corporal",
        ],
    },
    "Normal_Weight": {
        "urgencia": "Preventiva",
        "encaminhamento": "Clinico Geral",
        "acoes": [
            "Manter habitos alimentares e atividade fisica atuais",
            "Check-up anual de rotina",
            "Reforcar educacao em saude preventiva",
        ],
    },
    "Overweight_Level_I": {
        "urgencia": "Atencao",
        "encaminhamento": "Nutricao + Educacao Fisica",
        "acoes": [
            "Orientacao nutricional para reducao calorica moderada",
            "Prescricao de atividade fisica (150 min/semana)",
            "Reavaliacao em 3 meses",
        ],
    },
    "Overweight_Level_II": {
        "urgencia": "Atencao Prioritaria",
        "encaminhamento": "Nutricao + Endocrinologia",
        "acoes": [
            "Plano alimentar individualizado com nutricionista",
            "Avaliacao metabolica (glicemia, colesterol, pressao arterial)",
            "Programa estruturado de exercicios",
            "Reavaliacao em 60 dias",
        ],
    },
    "Obesity_Type_I": {
        "urgencia": "Alta",
        "encaminhamento": "Equipe Multidisciplinar",
        "acoes": [
            "Encaminhamento imediato: nutricao, endocrinologia e psicologia",
            "Exames laboratoriais completos (sindrome metabolica)",
            "Avaliacao de comorbidades (hipertensao, diabetes, apneia)",
            "Acompanhamento mensal",
        ],
    },
    "Obesity_Type_II": {
        "urgencia": "Muito Alta",
        "encaminhamento": "Equipe Multidisciplinar + Cirurgia Bariatrica",
        "acoes": [
            "Avaliacao para cirurgia bariatrica conforme criterios do CFM",
            "Tratamento intensivo de comorbidades",
            "Suporte psicologico obrigatorio",
            "Monitoramento quinzenal",
        ],
    },
    "Obesity_Type_III": {
        "urgencia": "Critica — Acao Imediata",
        "encaminhamento": "Internacao / Cirurgia Bariatrica",
        "acoes": [
            "Avaliacao urgente para internacao ou cirurgia bariatrica",
            "Controle intensivo de comorbidades graves",
            "Suporte psiquiatrico e nutricional hospitalar",
            "Monitoramento semanal ou internacao conforme quadro clinico",
        ],
    },
}

URGENCIA_COLOR = {
    "Eletiva": "#4FC3F7",
    "Preventiva": "#81C784",
    "Atencao": "#FFD54F",
    "Atencao Prioritaria": "#FFB74D",
    "Alta": "#FF8A65",
    "Muito Alta": "#E57373",
    "Critica — Acao Imediata": "#B71C1C",
}

# ── Carregar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    artifact = load_model()
    model = artifact["model"]
    le = artifact["label_encoder"]
    feature_cols = artifact["feature_cols"]
    model_ok = True
except Exception as e:
    st.error(f"Modelo nao encontrado. Execute pipeline.py primeiro. Erro: {e}")
    model_ok = False
    st.stop()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Sistema de Suporte a Decisao Clinica — Risco de Obesidade")
st.markdown(
    "Ferramenta de apoio ao diagnostico baseada em **fatores comportamentais e historico clinico**. "
    "Insira os dados do paciente para obter a classificacao de risco e a conduta recomendada."
)
st.info(
    "**Por que nao pedimos altura e peso?** O modelo foi treinado sem essas variaveis para avaliar "
    "os **determinantes comportamentais** da obesidade — o que possibilita intervencao precoce "
    "mesmo antes de o IMC indicar risco."
)
st.divider()

# ── Formulário ─────────────────────────────────────────────────────────────────
with st.form("form_clinico"):
    col_esq, col_dir = st.columns([1, 1], gap="large")

    with col_esq:
        st.markdown("#### Perfil do Paciente")
        gender = st.selectbox("Sexo biologico", ["Female", "Male"],
                              format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
        age = st.slider("Idade (anos)", 14, 80, 28)
        family_history = st.selectbox(
            "Historico familiar de excesso de peso",
            ["yes", "no"],
            format_func=lambda x: "Sim" if x == "yes" else "Nao",
            help="Pais ou irmaos com historico de sobrepeso ou obesidade"
        )

        st.markdown("#### Habitos Alimentares")
        favc = st.selectbox(
            "Consumo frequente de alimentos hipercaloricos",
            ["yes", "no"],
            format_func=lambda x: "Sim" if x == "yes" else "Nao",
        )
        fcvc = st.select_slider(
            "Frequencia de consumo de vegetais",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Raramente", 2: "As vezes", 3: "Diariamente"}[x],
        )
        ncp = st.select_slider(
            "Refeicoes principais por dia",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1: "1 refeicao", 2: "2 refeicoes", 3: "3 refeicoes", 4: "4 ou mais"}[x],
        )
        caec = st.selectbox(
            "Frequencia de lanches entre refeicoes",
            ["no", "Sometimes", "Frequently", "Always"],
            format_func=lambda x: {"no": "Nao lanche", "Sometimes": "As vezes",
                                    "Frequently": "Frequentemente", "Always": "Sempre"}[x],
        )

    with col_dir:
        st.markdown("#### Consumo e Hidratacao")
        ch2o = st.select_slider(
            "Consumo diario de agua",
            options=[1, 2, 3],
            value=2,
            format_func=lambda x: {1: "Menos de 1 litro", 2: "1 a 2 litros", 3: "Mais de 2 litros"}[x],
        )
        calc = st.selectbox(
            "Frequencia de consumo de alcool",
            ["no", "Sometimes", "Frequently", "Always"],
            format_func=lambda x: {"no": "Nao consome", "Sometimes": "Ocasionalmente",
                                    "Frequently": "Frequentemente", "Always": "Diariamente"}[x],
        )
        smoke = st.selectbox(
            "Tabagismo",
            ["no", "yes"],
            format_func=lambda x: "Nao fumante" if x == "no" else "Fumante",
        )

        st.markdown("#### Estilo de Vida")
        faf = st.select_slider(
            "Atividade fisica semanal",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "Sedentario", 1: "1-2x por semana",
                                    2: "3-4x por semana", 3: "5x ou mais"}[x],
        )
        tue = st.select_slider(
            "Tempo diario em frente a telas (TV, celular, computador)",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Ate 2 horas", 1: "3 a 5 horas", 2: "Mais de 5 horas"}[x],
        )
        scc = st.selectbox(
            "Monitoramento calorico",
            ["no", "yes"],
            format_func=lambda x: "Nao monitora" if x == "no" else "Monitora calorias",
        )
        mtrans = st.selectbox(
            "Principal meio de deslocamento",
            ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
            format_func=lambda x: {
                "Automobile": "Automovel", "Motorbike": "Motocicleta",
                "Bike": "Bicicleta", "Public_Transportation": "Transporte Publico",
                "Walking": "Caminhada",
            }[x],
        )

    st.markdown("")
    submitted = st.form_submit_button(
        "Classificar Risco e Gerar Conduta Clinica",
        use_container_width=True,
        type="primary",
    )

# ── Predição e resultado ───────────────────────────────────────────────────────
if submitted:
    freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    binary_map = {"Female": 0, "Male": 1, "yes": 1, "no": 0}

    caec_ord = freq_map[caec]
    calc_ord = freq_map[calc]
    healthy_score = fcvc + faf + ch2o - caec_ord - calc_ord - tue
    risk_score = caec_ord + calc_ord + tue + binary_map[favc] + binary_map[smoke]

    input_dict = {
        "Gender": binary_map[gender],
        "Age": age,
        "family_history": binary_map[family_history],
        "FAVC": binary_map[favc],
        "FCVC": fcvc,
        "NCP": ncp,
        "SMOKE": binary_map[smoke],
        "CH2O": ch2o,
        "SCC": binary_map[scc],
        "FAF": faf,
        "TUE": tue,
        "CAEC_ord": caec_ord,
        "CALC_ord": calc_ord,
        "healthy_score": healthy_score,
        "risk_score": risk_score,
        "MTRANS_Bike": int(mtrans == "Bike"),
        "MTRANS_Motorbike": int(mtrans == "Motorbike"),
        "MTRANS_Public_Transportation": int(mtrans == "Public_Transportation"),
        "MTRANS_Walking": int(mtrans == "Walking"),
    }

    X_input = pd.DataFrame([input_dict])[feature_cols]
    pred_enc = model.predict(X_input)[0]
    pred_class = le.inverse_transform([pred_enc])[0]
    proba = model.predict_proba(X_input)[0]
    confianca = max(proba) * 100

    label, _, color, nivel_risco = RISCO[pred_class]
    conduta = CONDUTA[pred_class]
    urgencia = conduta["urgencia"]
    urg_color = URGENCIA_COLOR[urgencia]

    st.divider()

    # Cabeçalho do resultado
    st.subheader("Resultado da Avaliacao Clinica")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.markdown(
        f"<div style='background:{color}22;border-left:5px solid {color};"
        f"padding:0.8rem;border-radius:6px'>"
        f"<small>Classificacao</small><br>"
        f"<strong style='font-size:1.1rem;color:{color}'>{label}</strong></div>",
        unsafe_allow_html=True,
    )
    res_col2.markdown(
        f"<div style='background:{urg_color}22;border-left:5px solid {urg_color};"
        f"padding:0.8rem;border-radius:6px'>"
        f"<small>Urgencia de Atendimento</small><br>"
        f"<strong style='font-size:1.1rem;color:{urg_color}'>{urgencia}</strong></div>",
        unsafe_allow_html=True,
    )
    res_col3.markdown(
        f"<div style='background:#37474F22;border-left:5px solid #607D8B;"
        f"padding:0.8rem;border-radius:6px'>"
        f"<small>Encaminhamento Indicado</small><br>"
        f"<strong style='font-size:1rem'>{conduta['encaminhamento']}</strong></div>",
        unsafe_allow_html=True,
    )
    res_col4.markdown(
        f"<div style='background:#1565C022;border-left:5px solid #1565C0;"
        f"padding:0.8rem;border-radius:6px'>"
        f"<small>Confianca do Modelo</small><br>"
        f"<strong style='font-size:1.1rem;color:#1565C0'>{confianca:.1f}%</strong></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Conduta e probabilidades
    cond_col, prob_col = st.columns([1, 1], gap="large")

    with cond_col:
        st.markdown("#### Conduta Recomendada")
        for i, acao in enumerate(conduta["acoes"], 1):
            st.markdown(f"**{i}.** {acao}")

    with prob_col:
        st.markdown("#### Distribuicao de Probabilidade por Classe")
        prob_df = pd.DataFrame({
            "Classe": [RISCO[c][0] for c in le.classes_],
            "Probabilidade": proba,
            "raw": le.classes_,
            "Nivel": [RISCO[c][3] for c in le.classes_],
        }).sort_values("Probabilidade", ascending=True)

        colors_bar = [RISCO[c][2] for c in prob_df["raw"]]
        fig_prob = px.bar(
            prob_df, x="Probabilidade", y="Classe",
            orientation="h",
            text=prob_df["Probabilidade"].apply(lambda x: f"{x*100:.1f}%"),
            color="Classe",
            color_discrete_sequence=colors_bar,
        )
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(
            showlegend=False, xaxis_tickformat=".0%",
            height=320, margin=dict(t=5, b=5),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # Perfil comportamental do paciente
    st.subheader("Perfil Comportamental do Paciente")
    p1, p2, p3, p4, p5 = st.columns(5)

    def indicador(col, titulo, valor, bom, ruim, fmt=str):
        cor = "#81C784" if bom(valor) else ("#E57373" if ruim(valor) else "#FFD54F")
        col.markdown(
            f"<div style='background:{cor}22;border:1px solid {cor};"
            f"padding:0.6rem;border-radius:6px;text-align:center'>"
            f"<small>{titulo}</small><br>"
            f"<strong style='font-size:1.1rem;color:{cor}'>{fmt(valor)}</strong></div>",
            unsafe_allow_html=True,
        )

    ativ_label = {0: "Sedentario", 1: "Pouco ativo", 2: "Moderado", 3: "Ativo"}
    indicador(p1, "Atividade Fisica", faf,
              bom=lambda v: v >= 2, ruim=lambda v: v == 0,
              fmt=lambda v: ativ_label[v])

    fcvc_label = {1: "Raramente", 2: "As vezes", 3: "Diariamente"}
    indicador(p2, "Consumo de Vegetais", fcvc,
              bom=lambda v: v == 3, ruim=lambda v: v == 1,
              fmt=lambda v: fcvc_label[v])

    h2o_label = {1: "<1L/dia", 2: "1-2L/dia", 3: ">2L/dia"}
    indicador(p3, "Hidratacao", ch2o,
              bom=lambda v: v == 3, ruim=lambda v: v == 1,
              fmt=lambda v: h2o_label[v])

    indicador(p4, "Score Saudavel", healthy_score,
              bom=lambda v: v >= 3, ruim=lambda v: v <= 0,
              fmt=lambda v: f"{v:+d}")

    indicador(p5, "Score de Risco", risk_score,
              bom=lambda v: v == 0, ruim=lambda v: v >= 4,
              fmt=lambda v: f"{v}/7")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Classificacao gerada por modelo Random Forest treinado com 1.688 pacientes | "
        "Acuracia de validacao: 80.6% | "
        "Este sistema e um suporte a decisao — o diagnostico final e responsabilidade do profissional de saude."
    )

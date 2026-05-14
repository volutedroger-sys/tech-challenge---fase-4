"""
Dashboard — Visão de Negócio para a Equipe Médica.
Foco em segmentação de risco, fatores acionáveis e recomendações clínicas.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Dashboard | Obesidade", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "Obesity.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[col] = df[col].round().astype(int)

    # Segmentos de risco clínico
    def segment(cls):
        if cls in ["Obesity_Type_II", "Obesity_Type_III"]:
            return "Alto Risco"
        elif cls in ["Obesity_Type_I", "Overweight_Level_II"]:
            return "Risco Moderado"
        elif cls == "Overweight_Level_I":
            return "Atencao"
        elif cls == "Normal_Weight":
            return "Saudavel"
        else:
            return "Abaixo do Peso"

    df["Segmento"] = df["Obesity"].apply(segment)
    df["Requer Intervencao"] = df["Segmento"].isin(["Alto Risco", "Risco Moderado"])

    freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["CAEC_ord"] = df["CAEC"].map(freq_map)
    df["CALC_ord"] = df["CALC"].map(freq_map)
    df["healthy_score"] = df["FCVC"] + df["FAF"] + df["CH2O"] - df["CAEC_ord"] - df["CALC_ord"] - df["TUE"]

    bins = [0, 17, 25, 35, 50, 100]
    labels = ["<18", "18-25", "26-35", "36-50", "50+"]
    df["Faixa Etaria"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    return df

df = load_data()

SEG_COLOR = {
    "Alto Risco": "#B71C1C",
    "Risco Moderado": "#FF8A65",
    "Atencao": "#FFD54F",
    "Saudavel": "#81C784",
    "Abaixo do Peso": "#4FC3F7",
}
SEG_ORDER = ["Alto Risco", "Risco Moderado", "Atencao", "Saudavel", "Abaixo do Peso"]

total = len(df)
alto_risco = (df["Segmento"] == "Alto Risco").sum()
moderado = (df["Segmento"] == "Risco Moderado").sum()
intervencao = df["Requer Intervencao"].sum()
sem_exercicio_risco = df[df["Requer Intervencao"] & (df["FAF"] == 0)].shape[0]
hist_fam_risco = df[df["Requer Intervencao"] & (df["family_history"] == "yes")].shape[0]

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.title("Painel de Saude Populacional — Obesidade")
st.markdown(
    "**Visao executiva para a equipe medica.** "
    "Identifique pacientes em risco, compreenda os fatores determinantes "
    "e priorize intervencoes com base em evidencias."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════
# KPIs ESTRATÉGICOS
# ══════════════════════════════════════════════════════════════════════
st.subheader("Panorama da Populacao Analisada")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Pacientes Analisados", f"{total:,}")
k2.metric(
    "Alto Risco (Obesidade II/III)",
    f"{alto_risco:,}",
    f"{alto_risco/total*100:.1f}% da populacao",
    delta_color="inverse",
)
k3.metric(
    "Risco Moderado (Obesidade I / Sobrepeso II)",
    f"{moderado:,}",
    f"{moderado/total*100:.1f}% da populacao",
    delta_color="inverse",
)
k4.metric(
    "Requerem Intervencao Medica",
    f"{intervencao:,}",
    f"{intervencao/total*100:.1f}% da populacao",
    delta_color="inverse",
)
k5.metric(
    "Sem Exercicio + Alto/Mod. Risco",
    f"{sem_exercicio_risco:,}",
    f"{sem_exercicio_risco/intervencao*100:.1f}% dos casos criticos",
    delta_color="inverse",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════
# PIRÂMIDE DE RISCO + DISTRIBUIÇÃO POR SEGMENTO
# ══════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns([2, 3])

with col_a:
    st.subheader("Piramide de Risco Clinico")
    seg_counts = df["Segmento"].value_counts().reindex(SEG_ORDER).reset_index()
    seg_counts.columns = ["Segmento", "Pacientes"]
    seg_counts["Pct"] = (seg_counts["Pacientes"] / total * 100).round(1)

    fig_pir = go.Figure()
    for _, row in seg_counts.iterrows():
        fig_pir.add_trace(go.Bar(
            x=[row["Pacientes"]],
            y=[row["Segmento"]],
            orientation="h",
            marker_color=SEG_COLOR[row["Segmento"]],
            text=f"{row['Pacientes']:,} pacientes ({row['Pct']}%)",
            textposition="inside",
            name=row["Segmento"],
        ))
    fig_pir.update_layout(
        showlegend=False, height=320,
        xaxis_title="Pacientes",
        margin=dict(l=0, r=10, t=10, b=30),
    )
    st.plotly_chart(fig_pir, use_container_width=True)
    st.caption(
        "**Alto Risco** = Obesidade Grau II/III | "
        "**Risco Moderado** = Obesidade I + Sobrepeso II | "
        "**Atencao** = Sobrepeso I"
    )

with col_b:
    st.subheader("Distribuicao por Faixa Etaria e Segmento de Risco")
    age_seg = (
        df.groupby(["Faixa Etaria", "Segmento"])
        .size()
        .reset_index(name="Pacientes")
    )
    fig_age = px.bar(
        age_seg, x="Faixa Etaria", y="Pacientes", color="Segmento",
        barmode="stack",
        color_discrete_map=SEG_COLOR,
        category_orders={"Segmento": SEG_ORDER, "Faixa Etaria": ["<18", "18-25", "26-35", "36-50", "50+"]},
        labels={"Faixa Etaria": "Faixa Etaria", "Pacientes": "Pacientes"},
    )
    fig_age.update_layout(height=320, margin=dict(t=10))
    st.plotly_chart(fig_age, use_container_width=True)
    st.caption("Faixas etarias mais jovens concentram grande volume de casos de risco moderado e alto.")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# FATORES DE RISCO MODIFICÁVEIS
# ══════════════════════════════════════════════════════════════════════
st.subheader("Fatores de Risco Modificaveis — Onde Intervir?")
st.markdown(
    "Os graficos abaixo mostram a prevalencia de habitos de risco **dentro do grupo que ja requer intervencao**. "
    "Sao os alvos prioritarios para programas de mudanca de estilo de vida."
)

df_risco = df[df["Requer Intervencao"]].copy()

col_c, col_d, col_e = st.columns(3)

with col_c:
    st.markdown("**Atividade Fisica**")
    faf_map = {0: "Sedentario", 1: "Pouco ativo", 2: "Moderado", 3: "Ativo"}
    faf_counts = df_risco["FAF"].map(faf_map).value_counts().reset_index()
    faf_counts.columns = ["Nivel", "Pacientes"]
    order_faf = ["Sedentario", "Pouco ativo", "Moderado", "Ativo"]
    faf_counts["Nivel"] = pd.Categorical(faf_counts["Nivel"], categories=order_faf, ordered=True)
    faf_counts = faf_counts.sort_values("Nivel")
    fig_faf = px.pie(faf_counts, values="Pacientes", names="Nivel",
                     color="Nivel",
                     color_discrete_map={
                         "Sedentario": "#B71C1C", "Pouco ativo": "#FF8A65",
                         "Moderado": "#FFD54F", "Ativo": "#81C784",
                     },
                     hole=0.45)
    fig_faf.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig_faf, use_container_width=True)
    sed_pct = (df_risco["FAF"] == 0).mean() * 100
    st.info(f"**{sed_pct:.0f}%** dos pacientes de risco sao sedentarios ou pouco ativos.")

with col_d:
    st.markdown("**Consumo de Alimentos Calóricos**")
    favc_counts = df_risco["FAVC"].value_counts().reset_index()
    favc_counts.columns = ["FAVC", "Pacientes"]
    favc_counts["Rotulo"] = favc_counts["FAVC"].map({"yes": "Consome frequentemente", "no": "Nao consome"})
    fig_favc = px.pie(favc_counts, values="Pacientes", names="Rotulo",
                      color="Rotulo",
                      color_discrete_map={"Consome frequentemente": "#E57373", "Nao consome": "#81C784"},
                      hole=0.45)
    fig_favc.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig_favc, use_container_width=True)
    favc_pct = (df_risco["FAVC"] == "yes").mean() * 100
    st.info(f"**{favc_pct:.0f}%** dos pacientes de risco consomem alimentos calóricos com frequencia.")

with col_e:
    st.markdown("**Historico Familiar**")
    fam_counts = df_risco["family_history"].value_counts().reset_index()
    fam_counts.columns = ["family_history", "Pacientes"]
    fam_counts["Rotulo"] = fam_counts["family_history"].map({"yes": "Com historico familiar", "no": "Sem historico"})
    fig_fam = px.pie(fam_counts, values="Pacientes", names="Rotulo",
                     color="Rotulo",
                     color_discrete_map={"Com historico familiar": "#FF8A65", "Sem historico": "#90CAF9"},
                     hole=0.45)
    fig_fam.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig_fam, use_container_width=True)
    fam_pct = (df_risco["family_history"] == "yes").mean() * 100
    st.info(f"**{fam_pct:.0f}%** dos pacientes de risco possuem historico familiar de obesidade.")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# COMPORTAMENTOS PROTETORES vs FATORES DE RISCO
# ══════════════════════════════════════════════════════════════════════
col_f, col_g = st.columns(2)

with col_f:
    st.subheader("Score de Habitos Saudaveis por Segmento")
    st.markdown("Pacientes saudaveis apresentam score significativamente mais alto — confirma o poder preditivo dos habitos comportamentais.")
    box_df = df.copy()
    box_df["Segmento"] = pd.Categorical(box_df["Segmento"], categories=SEG_ORDER, ordered=True)
    box_df = box_df.sort_values("Segmento")
    fig_box = px.box(
        box_df, x="Segmento", y="healthy_score",
        color="Segmento",
        color_discrete_map=SEG_COLOR,
        labels={"healthy_score": "Score de Habitos Saudaveis", "Segmento": ""},
    )
    fig_box.update_layout(showlegend=False, height=340, xaxis_tickangle=-15)
    st.plotly_chart(fig_box, use_container_width=True)

with col_g:
    st.subheader("Meio de Transporte e Nivel de Risco")
    st.markdown("Pacientes que usam automovel concentram maior prevalencia de risco — sinal de sedentarismo no deslocamento.")
    trans_seg = (
        df.groupby(["MTRANS", "Segmento"])
        .size()
        .reset_index(name="Pacientes")
    )
    trans_labels = {
        "Automobile": "Automovel", "Motorbike": "Moto",
        "Bike": "Bicicleta", "Public_Transportation": "Transporte Publico", "Walking": "A Pe",
    }
    trans_seg["Transporte"] = trans_seg["MTRANS"].map(trans_labels)
    fig_trans = px.bar(
        trans_seg, x="Transporte", y="Pacientes", color="Segmento",
        barmode="stack",
        color_discrete_map=SEG_COLOR,
        category_orders={"Segmento": SEG_ORDER},
        labels={"Transporte": "", "Pacientes": "Pacientes"},
    )
    fig_trans.update_layout(height=340, xaxis_tickangle=-15)
    st.plotly_chart(fig_trans, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════
# RECOMENDAÇÕES ESTRATÉGICAS
# ══════════════════════════════════════════════════════════════════════
st.subheader("Recomendacoes Estrategicas para a Equipe Medica")

r1, r2, r3 = st.columns(3)
with r1:
    st.error(
        "**Intervencao Prioritaria**\n\n"
        f"{alto_risco:,} pacientes ({alto_risco/total*100:.1f}%) estao em Obesidade Grau II ou III. "
        "Encaminhamento imediato para equipe multidisciplinar (nutricao, endocrinologia, psicologia)."
    )
with r2:
    st.warning(
        "**Programa de Atividade Fisica**\n\n"
        f"{sem_exercicio_risco:,} pacientes de risco sao sedentarios. "
        "Prescricao de atividade fisica supervisionada e acompanhamento periodico sao acoes de alto impacto."
    )
with r3:
    st.info(
        "**Educacao Alimentar**\n\n"
        f"{favc_pct:.0f}% dos casos de risco consomem alimentos calóricos com frequencia. "
        "Programas de reeducacao alimentar e monitoramento calorico podem reduzir este indicador."
    )

st.caption(
    "Fonte: Dataset Obesity — FIAP Tech Challenge Fase 4. "
    "Analise baseada em 2.111 pacientes. Height e Weight excluidos para evitar data leakage via IMC."
)

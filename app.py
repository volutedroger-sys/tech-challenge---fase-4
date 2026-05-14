"""
Tech Challenge Fase 4 — Predição de Obesidade
Entry point da aplicação Streamlit multi-página.
"""

import streamlit as st

st.set_page_config(
    page_title="Obesidade | Tech Challenge Fase 4",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🏥 Obesidade TC4")
st.sidebar.markdown("Navegue pelas páginas acima.")

st.title("Sistema de Análise e Predição de Obesidade")
st.markdown(
    """
    Bem-vindo ao sistema desenvolvido para o **Tech Challenge Fase 4 — POSTECH / FIAP**.

    Use o menu lateral para navegar entre as páginas:

    | Página | Descrição |
    |--------|-----------|
    | **Dashboard** | Insights analíticos sobre o dataset de obesidade |
    | **Predição** | Formulário preditivo para diagnóstico assistido |

    > **Nota:** Height e Weight foram excluídos do modelo para evitar *data leakage* via IMC.
    """
)

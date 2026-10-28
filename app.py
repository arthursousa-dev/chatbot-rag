"""Interface web do chatbot (Streamlit). Rode com: streamlit run app.py"""

import streamlit as st

from rag import Indice, carregar_trechos, responder

st.set_page_config(page_title="Chatbot RAG — Clínica", page_icon="🦷")
st.title("Assistente da Clínica")
st.caption("Respostas geradas a partir dos documentos da clínica, com a fonte citada.")


@st.cache_resource
def obter_indice() -> Indice:
    return Indice(carregar_trechos())


indice = obter_indice()

if "historico" not in st.session_state:
    st.session_state.historico = []

for papel, texto in st.session_state.historico:
    st.chat_message(papel).write(texto)

if pergunta := st.chat_input("Pergunte sobre horários, preços, convênios..."):
    st.chat_message("user").write(pergunta)
    r = responder(indice, pergunta)
    texto = r["resposta"] + (f"\n\n*Fontes: {', '.join(r['fontes'])}*" if r["fontes"] else "")
    st.chat_message("assistant").write(texto)
    st.session_state.historico += [("user", pergunta), ("assistant", texto)]

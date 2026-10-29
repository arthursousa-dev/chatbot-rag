# Chatbot RAG para clínicas odontológicas

Assistente que responde dúvidas de pacientes (horários, preços, convênios, cuidados pós-procedimento) **a partir dos documentos da própria clínica**, citando a fonte de cada resposta. Perguntas fora do escopo não são inventadas: o bot orienta a falar com a recepção.

## Como funciona

1. **Ingestão:** os arquivos `.md`, `.txt` e `.pdf` em `docs/` são divididos em trechos curtos (o texto do PDF é extraído com `pypdf`).
2. **Recuperação:** a pergunta é comparada com os trechos por similaridade (TF-IDF com n-gramas de caracteres + cosseno), e os 3 mais relevantes são selecionados. Abaixo de um limiar mínimo, nada é recuperado.
3. **Geração:** os trechos vão como contexto para a API da Claude, com instruções para responder só com base neles e citar a fonte.

## Rodando

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sua-chave
streamlit run app.py      # interface web
python rag.py             # modo terminal
pytest                    # testes da recuperação
```

Sem a chave da API, o bot roda em modo de teste e mostra o trecho recuperado.

## Próximos passos

- [ ] Trocar TF-IDF por embeddings semânticos com pgvector (PostgreSQL)
- [x] Ingestão de PDFs
- [ ] Integração com WhatsApp

"""Pipeline RAG: carrega documentos, divide em trechos, recupera os mais
relevantes para a pergunta e gera a resposta com a API da Claude."""

import os
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent / "docs"
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# Palavras muito comuns que não ajudam a achar o trecho certo.
STOPWORDS = set("""a o as os um uma uns umas de do da dos das em no na nos nas por para pra com sem
e ou que qual quais quanto quanta como quando onde se seu sua seus suas meu minha voce voces
eu ele ela nos eles elas isso isto esse essa este esta ao aos e ja nao sim mais muito pode posso
tem ter ser sao foi vai depois antes""".split())


@dataclass
class Trecho:
    fonte: str
    texto: str


def carregar_trechos(pasta: Path = DOCS_DIR, max_palavras: int = 40) -> list[Trecho]:
    """Lê os .md/.txt da pasta e divide cada um em trechos por parágrafo,
    agrupando parágrafos curtos até ~max_palavras."""
    trechos = []
    for arq in sorted(pasta.glob("*.*")):
        if arq.suffix not in {".md", ".txt"}:
            continue
        paragrafos = [p.strip() for p in arq.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        atual = []
        for p in paragrafos:
            atual.append(p)
            if sum(len(x.split()) for x in atual) >= max_palavras:
                trechos.append(Trecho(arq.name, "\n".join(atual)))
                atual = []
        if atual:
            trechos.append(Trecho(arq.name, "\n".join(atual)))
    return trechos


class Indice:
    """Índice de busca por similaridade (TF-IDF + cosseno)."""

    def __init__(self, trechos: list[Trecho]):
        if not trechos:
            raise ValueError("Nenhum documento encontrado em docs/")
        self.trechos = trechos
        # n-gramas de caracteres toleram variações como "sábado"/"sábados".
        self.vetorizador = TfidfVectorizer(
            strip_accents="unicode", lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
            sublinear_tf=True,
        )
        self.matriz = self.vetorizador.fit_transform([self._limpar(t.texto) for t in trechos])

    @staticmethod
    def _limpar(texto: str) -> str:
        import unicodedata

        sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
        palavras = "".join(c if c.isalnum() else " " for c in sem_acento).split()
        return " ".join(p for p in palavras if p not in STOPWORDS)

    def buscar(self, pergunta: str, k: int = 3, minimo: float = 0.12) -> list[tuple[Trecho, float]]:
        scores = cosine_similarity(self.vetorizador.transform([self._limpar(pergunta)]), self.matriz)[0]
        ordem = scores.argsort()[::-1][:k]
        return [(self.trechos[i], float(scores[i])) for i in ordem if scores[i] >= minimo]


PROMPT_SISTEMA = (
    "Você é o assistente virtual de uma clínica odontológica. Responda em português, "
    "de forma curta e cordial, usando SOMENTE as informações dos trechos fornecidos. "
    "Se a resposta não estiver nos trechos, diga que não tem essa informação e sugira "
    "falar com a recepção. Ao final, cite a fonte entre colchetes, ex.: [precos.md]."
)


def montar_contexto(resultados: list[tuple[Trecho, float]]) -> str:
    return "\n\n".join(f"<trecho fonte=\"{t.fonte}\">\n{t.texto}\n</trecho>" for t, _ in resultados)


def responder(indice: Indice, pergunta: str) -> dict:
    resultados = indice.buscar(pergunta)
    fontes = sorted({t.fonte for t, _ in resultados})
    if not resultados:
        return {"resposta": "Não encontrei essa informação. Fale com a recepção da clínica.", "fontes": []}

    if not os.getenv("ANTHROPIC_API_KEY"):
        # Modo sem chave: mostra o trecho recuperado, útil para testar a busca.
        melhor = resultados[0][0]
        return {"resposta": f"(modo sem API) Trecho mais relevante:\n\n{melhor.texto}", "fontes": fontes}

    import anthropic

    cliente = anthropic.Anthropic()
    msg = cliente.messages.create(
        model=MODEL,
        max_tokens=500,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": f"{montar_contexto(resultados)}\n\nPergunta: {pergunta}"}],
    )
    return {"resposta": msg.content[0].text, "fontes": fontes}


if __name__ == "__main__":
    idx = Indice(carregar_trechos())
    print(f"{len(idx.trechos)} trechos indexados. Digite uma pergunta (ou 'sair').")
    while (q := input("> ").strip()).lower() not in {"sair", "exit"}:
        r = responder(idx, q)
        print(r["resposta"], "\nFontes:", ", ".join(r["fontes"]) or "-")

from rag import Indice, carregar_trechos, responder

IDX = Indice(carregar_trechos())


def fontes(pergunta):
    return [t.fonte for t, _ in IDX.buscar(pergunta)]


def test_recupera_o_documento_certo_entre_os_trechos_enviados_ao_llm():
    assert "procedimentos-e-precos.md" in fontes("quanto custa o clareamento")
    assert "horarios-e-contato.md" in fontes("vocês atendem sábado?")
    assert "convenios.md" in fontes("aceitam convênio")
    assert "convenios.md" in fontes("posso parcelar?")
    assert "cuidados-pos-procedimento.md" in fontes("o que fazer depois de uma extração")


def test_pergunta_fora_do_escopo_nao_recupera_nada():
    assert responder(IDX, "qual a capital da França")["fontes"] == []


def test_falha_da_api_vira_mensagem_amigavel(monkeypatch):
    import anthropic

    class ClienteFalho:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise anthropic.APIError.__new__(anthropic.APIConnectionError)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "teste")
    monkeypatch.setattr(anthropic, "Anthropic", ClienteFalho)
    r = responder(IDX, "quanto custa o clareamento")
    assert r["fontes"] == [] and "recepção" in r["resposta"]

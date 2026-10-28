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

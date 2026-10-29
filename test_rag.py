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


def _pdf_minimo(texto: str) -> bytes:
    """PDF de uma página com uma linha de texto (Helvetica), montado à mão para o teste."""
    conteudo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(conteudo)).encode() + b" >>\nstream\n" + conteudo + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objetos, 1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objetos) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    pdf += f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return pdf


def test_pdf_na_pasta_de_documentos_e_indexado(tmp_path):
    (tmp_path / "plantao.pdf").write_bytes(_pdf_minimo("Plantao de urgencia aos domingos das 8h as 12h"))
    (tmp_path / "ignorado.docx").write_bytes(b"nao lido")
    trechos = carregar_trechos(tmp_path)
    assert [t.fonte for t in trechos] == ["plantao.pdf"]
    assert "Plantao de urgencia" in trechos[0].texto
    assert [t.fonte for t, _ in Indice(trechos).buscar("tem plantao no domingo?")] == ["plantao.pdf"]


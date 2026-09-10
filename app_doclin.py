"""
FRONTEND STREAMLIT PARA RAG LOCAL COM BANCO DOCLING

Fluxo:
1. O usuário envia uma pergunta.
2. A pergunta é transformada em embedding.
3. O Chroma consulta os chunks estruturados pelo Docling.
4. Os trechos são enviados ao modelo local do Ollama.
5. Os trechos são enviados ao Gemini.
6. O modelo responde somente com base nos documentos.

Não utiliza a API da OpenAI.
Usa GOOGLE_API_KEY armazenada nos Secrets do Streamlit Cloud.
"""

from pathlib import Path
import re

import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

DIRETORIO_PROJETO = Path(__file__).resolve().parent

# Banco produzido pelo arquivo criar_db_doclin.py.
CAMINHO_DB = DIRETORIO_PROJETO / "db_doclin"

NOME_COLECAO = "documentos_pdf"

MODELO_EMBEDDING = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Modelo Gemini usado na nuvem pelo Streamlit Cloud.
MODELO_GEMINI = "gemini-2.5-flash"

QUANTIDADE_PADRAO_DOCUMENTOS = 14

# Tamanho máximo solicitado para cada resposta.
LIMITE_PALAVRAS_RESPOSTA = 80


# ==========================================================
# CONFIGURAÇÃO DO STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="Agente Virtual Financeiro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# ESTILO
# ==========================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        .titulo-principal {
            font-size: 38px;
            font-weight: 700;
            color: #1565C0;
            margin-bottom: 5px;
        }

        .subtitulo {
            color: #666666;
            font-size: 17px;
            margin-bottom: 25px;
        }

        .status-ok {
            background-color: #DFF2E1;
            color: #185C25;
            border: 1px solid #A8D5AE;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
        }

        .status-aviso {
            background-color: #FFF3CD;
            color: #664D03;
            border: 1px solid #FFECB5;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
        }

        .fonte-documento {
            background-color: #F5F7FA;
            border-left: 4px solid #1565C0;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CARREGAR O BANCO CHROMA
# ==========================================================

@st.cache_resource(
    show_spinner="Carregando o banco de documentos..."
)
def carregar_banco_vetorial():
    """
    Carrega o modelo de embeddings e abre o banco Chroma.
    """

    caminho = CAMINHO_DB

    if not caminho.exists():
        raise FileNotFoundError(
            f"A pasta '{CAMINHO_DB}' não foi encontrada. "
            "Execute primeiro o arquivo criar_db_doclin.py."
        )

    if not caminho.is_dir():
        raise NotADirectoryError(
            f"'{CAMINHO_DB}' existe, mas não é uma pasta."
        )

    arquivo_chroma = caminho / "chroma.sqlite3"

    if not arquivo_chroma.exists():
        raise FileNotFoundError(
            f"O arquivo '{arquivo_chroma}' não foi encontrado. "
            "A criação do banco Docling pode não ter sido concluída."
        )

    funcao_embedding = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDING,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    banco = Chroma(
        persist_directory=str(CAMINHO_DB),
        embedding_function=funcao_embedding,
        collection_name=NOME_COLECAO
    )

    quantidade_registros = banco._collection.count()

    if quantidade_registros == 0:
        raise ValueError(
            f"A coleção '{NOME_COLECAO}' está vazia. "
            "Crie novamente o banco com os documentos PDF."
        )

    return banco, quantidade_registros


# ==========================================================
# CARREGAR O GEMINI
# ==========================================================

@st.cache_resource(
    show_spinner="Conectando ao Gemini..."
)
def carregar_modelo():
    """
    Cria a conexão com o modelo Gemini usando a chave armazenada
    nos Secrets do Streamlit Cloud.
    """

    try:
        chave_google = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        chave_google = None

    if not chave_google:
        raise RuntimeError(
            "A chave GOOGLE_API_KEY não foi configurada nos Secrets "
            "do Streamlit Cloud."
        )

    modelo = ChatGoogleGenerativeAI(
        model=MODELO_GEMINI,
        google_api_key=chave_google,
        temperature=0,
        max_output_tokens=250
    )

    return modelo


# ==========================================================
# PROMPT
# ==========================================================

PROMPT_TEMPLATE = """
Você é um assistente de consulta de documentos. Sua tarefa é localizar
no contexto a informação que melhor responde à pergunta, mesmo quando
a pergunta for curta, informal ou estiver incompleta.

Responda usando exclusivamente as informações presentes no
contexto recuperado do banco de documentos.

REGRAS OBRIGATÓRIAS:

1. Leia todo o contexto antes de responder.
2. Interprete expressões equivalentes. Por exemplo, "taxa de juros",
   "juros básicos" e "taxa Selic" podem se referir ao mesmo assunto.
3. Se a pergunta pedir a taxa de juros, procure no contexto o percentual
   da Selic e informe também se ele é ao ano.
4. Se a pergunta pedir uma data, procure datas e referências de calendário.
5. Interprete corretamente tabelas em Markdown, relacionando cabeçalhos,
   linhas, colunas, períodos e valores antes de responder.
6. Considere os títulos e subtítulos preservados pelo Docling como parte do
   contexto de cada trecho.
7. Não utilize conhecimentos externos nem invente informações.
8. Responda em português do Brasil, de maneira resumida, clara e direta.
9. A resposta deve ter no máximo {limite_palavras} palavras.
10. Comece diretamente pela resposta, sem explicar o processo de busca.
11. Sempre entregue uma resposta útil. Se o valor ou a data exata não estiverem
   explícitos, explique objetivamente o que o documento permite concluir e
   qual informação exata não foi informada.
12. Nunca responda somente "não sei" ou somente que não encontrou a resposta.
13. Não mencione que você é um modelo de linguagem.

PERGUNTA DO USUÁRIO:

{pergunta}

CONTEXTO RECUPERADO DOS DOCUMENTOS:

{base_conhecimento}

RESPOSTA:
"""

prompt_rag = ChatPromptTemplate.from_template(
    PROMPT_TEMPLATE
)


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def obter_nome_fonte(documento):
    """
    Obtém somente o nome do arquivo de origem.
    """

    fonte = documento.metadata.get(
        "source",
        "Fonte não identificada"
    )

    try:
        return Path(str(fonte)).name
    except Exception:
        return str(fonte)


def obter_pagina(documento):
    """
    Obtém o número da página do documento.
    """

    pagina = documento.metadata.get("page")

    if pagina is None:
        return None

    try:
        # O criar_db_doclin.py salva a página começando em zero.
        return int(pagina) + 1
    except (TypeError, ValueError):
        return pagina


def montar_contexto(documentos):
    """
    Junta os documentos recuperados para enviar ao modelo.
    """

    partes = []

    for numero, documento in enumerate(
        documentos,
        start=1
    ):
        fonte = obter_nome_fonte(documento)
        pagina = obter_pagina(documento)

        identificacao = [
            f"DOCUMENTO {numero}",
            f"Fonte: {fonte}"
        ]

        if pagina is not None:
            identificacao.append(f"Página: {pagina}")

        cabecalho = "\n".join(identificacao)

        partes.append(
            f"{cabecalho}\n\n{documento.page_content}"
        )

    return "\n\n============================\n\n".join(partes)


def pesquisar_documentos(
    pergunta,
    banco,
    quantidade_documentos
):
    """
    Pesquisa no Chroma os documentos mais semelhantes.
    """

    pergunta_expandida = ampliar_pergunta_para_busca(pergunta)

    documentos = banco.similarity_search(
        query=pergunta_expandida,
        k=quantidade_documentos
    )

    return documentos


def ampliar_pergunta_para_busca(pergunta):
    """
    Acrescenta termos equivalentes para melhorar perguntas muito curtas.
    A ampliação é usada somente na busca; o modelo recebe a pergunta original.
    """

    texto = pergunta.lower()
    complementos = []

    if any(
        termo in texto
        for termo in ["juro", "juros", "selic", "taxa"]
    ):
        complementos.append(
            "taxa Selic taxa básica de juros percentual ao ano "
            "decisão do Copom manutenção elevação redução"
        )

    if any(
        termo in texto
        for termo in ["reunião", "reuniao", "copom", "próxima", "proxima"]
    ):
        complementos.append(
            "data calendário próxima reunião do Copom "
            "Comitê de Política Monetária"
        )

    if not complementos:
        return pergunta

    return pergunta + " " + " ".join(complementos)


def gerar_resposta_extrativa(pergunta, documentos):
    """
    Fallback seguro: seleciona diretamente a frase mais relacionada do PDF
    quando o modelo local insiste em responder apenas "não sei".
    """

    stopwords = {
        "a", "o", "as", "os", "de", "da", "do", "das", "dos",
        "e", "em", "para", "por", "qual", "quais", "foi", "será",
        "sera", "nos", "nas", "um", "uma", "que", "pelo", "pela"
    }

    termos = {
        termo
        for termo in re.findall(r"[a-záàâãéêíóôõúç]+", pergunta.lower())
        if len(termo) > 2 and termo not in stopwords
    }

    candidatos = []

    for documento in documentos:
        texto = re.sub(r"\s+", " ", documento.page_content).strip()
        frases = re.split(r"(?<=[.!?])\s+", texto)

        for frase in frases:
            frase = frase.strip()

            if len(frase) < 30:
                continue

            frase_minuscula = frase.lower()
            pontuacao = sum(
                1 for termo in termos if termo in frase_minuscula
            )

            if any(t in pergunta.lower() for t in ["selic", "juro", "taxa"]):
                if "selic" in frase_minuscula:
                    pontuacao += 5
                if "%" in frase:
                    pontuacao += 4
                if "a.a" in frase_minuscula or "ao ano" in frase_minuscula:
                    pontuacao += 2

            if any(t in pergunta.lower() for t in ["data", "quando", "reunião", "reuniao"]):
                if "reuni" in frase_minuscula:
                    pontuacao += 3
                if re.search(r"\b\d{1,2}\s+e\s+\d{1,2}\s+de\s+[a-zç]+", frase_minuscula):
                    pontuacao += 5

            candidatos.append((pontuacao, frase))

    if not candidatos:
        return "Os documentos não possuem texto suficiente para responder."

    candidatos.sort(key=lambda item: item[0], reverse=True)
    melhor_frase = candidatos[0][1]
    palavras = melhor_frase.split()

    if len(palavras) > LIMITE_PALAVRAS_RESPOSTA:
        melhor_frase = " ".join(
            palavras[:LIMITE_PALAVRAS_RESPOSTA]
        ).rstrip(",;:") + "..."

    return melhor_frase


def gerar_resposta(
    pergunta,
    banco,
    modelo,
    quantidade_documentos
):
    """
    Executa todo o processo da RAG.
    """

    documentos = pesquisar_documentos(
        pergunta=pergunta,
        banco=banco,
        quantidade_documentos=quantidade_documentos
    )

    if not documentos:
        return (
            "Não sei a resposta com base nos documentos disponíveis.",
            []
        )

    base_conhecimento = montar_contexto(documentos)

    cadeia = (
        prompt_rag
        | modelo
        | StrOutputParser()
    )

    resposta = cadeia.invoke(
        {
            "pergunta": pergunta,
            "base_conhecimento": base_conhecimento,
            "limite_palavras": LIMITE_PALAVRAS_RESPOSTA
        }
    )

    resposta = resposta.strip()

    resposta_normalizada = resposta.lower().strip()

    if (
        not resposta
        or resposta_normalizada.startswith("não sei")
        or resposta_normalizada.startswith("nao sei")
        or "não encontrei" in resposta_normalizada
        or "não foi possível encontrar" in resposta_normalizada
        or "não há informação" in resposta_normalizada
        or "não há informações" in resposta_normalizada
        or "não consta" in resposta_normalizada
        or "contexto não fornece" in resposta_normalizada
    ):
        resposta = gerar_resposta_extrativa(
            pergunta=pergunta,
            documentos=documentos
        )

    return resposta, documentos


def mensagem_erro_gemini(erro):
    """
    Converte erros técnicos do Gemini em orientações.
    """

    texto_erro = str(erro).lower()

    if (
        "api key" in texto_erro
        or "api_key" in texto_erro
        or "authentication" in texto_erro
        or "unauthenticated" in texto_erro
        or "permission" in texto_erro
    ):
        return (
            "Não foi possível autenticar no Gemini. "
            "Verifique se o Secret `GOOGLE_API_KEY` está configurado "
            "corretamente no Streamlit Cloud."
        )

    if "quota" in texto_erro or "resource exhausted" in texto_erro:
        return (
            "O limite de uso da API do Gemini foi atingido. "
            "Verifique a cota e o faturamento da chave usada."
        )

    return f"Não foi possível gerar a resposta: {erro}"


# ==========================================================
# CABEÇALHO
# ==========================================================

st.markdown(
    '<div class="titulo-principal">'
    'Assistente Virtual IID - Anuário Estatístico IBGE_2022'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitulo">
        Faça perguntas sobre os documentos armazenados
        no banco vetorial.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# BARRA LATERAL
# ==========================================================

with st.sidebar:
    st.header("⚙️ Configurações")

    quantidade_documentos = st.slider(
        label="Quantidade de trechos consultados",
        min_value=1,
        max_value=14,
        value=QUANTIDADE_PADRAO_DOCUMENTOS,
        step=1,
        help=(
            "Os trechos são usados internamente para criar uma única "
            "resposta e não serão exibidos na tela."
        )
    )

    st.divider()

    st.subheader("🤖 Modelo de IA")

    st.write(f"**Modelo:** `{MODELO_GEMINI}`")
    st.write("**Banco:** `db_doclin` (processado com Docling)")
    st.write(
        f"**Resposta:** até {LIMITE_PALAVRAS_RESPOSTA} palavras"
    )
    st.write("**Servidor:** Google Gemini")
    st.write("**API:** Google Gemini")

    st.divider()

    if st.button(
        "🗑️ Limpar conversa",
        use_container_width=True
    ):
        st.session_state.mensagens = []
        st.rerun()


# ==========================================================
# INICIALIZAR O SISTEMA
# ==========================================================

try:
    banco, total_registros = carregar_banco_vetorial()
    modelo = carregar_modelo()

    st.markdown(
        f"""
        <div class="status-ok">
            ✅ Sistema carregado.
            O banco possui <strong>{total_registros}</strong>
            trechos e o modelo será executado na nuvem pelo Gemini.
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as erro:
    st.error(f"Erro ao iniciar o sistema: {erro}")

    st.markdown(
        """
        Verifique:

        1. Se a pasta `db_doclin` está junto deste arquivo;
        2. Se a coleção se chama `documentos_pdf`;
        3. Se existe `db_doclin/chroma.sqlite3`;
        4. Se as bibliotecas foram instaladas.
        """
    )

    st.stop()


# ==========================================================
# HISTÓRICO DA CONVERSA
# ==========================================================

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []


for mensagem in st.session_state.mensagens:
    papel = mensagem["papel"]
    conteudo = mensagem["conteudo"]

    with st.chat_message(papel):
        st.markdown(conteudo)


# ==========================================================
# CAMPO DA PERGUNTA
# ==========================================================

pergunta = st.chat_input(
    "Digite uma pergunta sobre os documentos..."
)


# ==========================================================
# PROCESSAR A PERGUNTA
# ==========================================================

if pergunta:
    pergunta = pergunta.strip()

    if pergunta:
        # Salvar pergunta no histórico
        st.session_state.mensagens.append(
            {
                "papel": "user",
                "conteudo": pergunta
            }
        )

        # Mostrar pergunta
        with st.chat_message("user"):
            st.markdown(pergunta)

        # Criar resposta
        with st.chat_message("assistant"):
            with st.spinner(
                "Pesquisando nos documentos e gerando a resposta..."
            ):
                try:
                    resposta, documentos = gerar_resposta(
                        pergunta=pergunta,
                        banco=banco,
                        modelo=modelo,
                        quantidade_documentos=quantidade_documentos
                    )

                    st.markdown(resposta)

                    # Salvar resposta no histórico
                    st.session_state.mensagens.append(
                        {
                            "papel": "assistant",
                            "conteudo": resposta,
                            "documentos": documentos
                        }
                    )

                except Exception as erro:
                    mensagem = mensagem_erro_gemini(erro)

                    st.error(mensagem)

                    st.session_state.mensagens.append(
                        {
                            "papel": "assistant",
                            "conteudo": mensagem,
                            "documentos": []
                        }
                    )

import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0.4)

# -------------------------------
# Catálogo fixo de links internos (Dra. Angélica Bauer)
# -------------------------------
LINKS_INTERNOS_ANGELICA = [
    {
        "titulo": "Home",
        "url": "https://www.angelicabauer.com.br/",
        "anchor_sugerida": "Dermatologia com foco em alopecia em Porto Alegre"
    },
    {
        "titulo": "Tratamento Dermatológico Clínico",
        "url": "https://www.angelicabauer.com.br/tratamento-dermatologico-clinico/",
        "anchor_sugerida": "tratamentos dermatológicos clínicos com a Dra. Angélica Bauer"
    },
    {
        "titulo": "Tricologia",
        "url": "https://www.angelicabauer.com.br/tricologia/",
        "anchor_sugerida": "cuidados e tratamentos em tricologia"
    },
    {
        "titulo": "Mapeamento Corporal",
        "url": "https://www.angelicabauer.com.br/mapeamento-corporal/",
        "anchor_sugerida": "mapeamento corporal para prevenção e diagnóstico precoce"
    },
    {
        "titulo": "Cirurgia Dermatológica",
        "url": "https://www.angelicabauer.com.br/cirurgia-dermatologica/",
        "anchor_sugerida": "procedimentos de cirurgia dermatológica"
    },
    {
        "titulo": "Procedimentos Estéticos e Tecnologias",
        "url": "https://www.angelicabauer.com.br/procedimentos-esteticos-e-tecnologias/",
        "anchor_sugerida": "procedimentos estéticos e tecnologias para saúde da pele"
    },
    {
        "titulo": "Blog",
        "url": "https://www.angelicabauer.com.br/blog/",
        "anchor_sugerida": "conteúdos sobre saúde da pele e cabelos"
    }
]

LINK_WHATSAPP_ANGELICA = "https://api.whatsapp.com/send?phone=5551999216941&text=Olá,%20estava%20navegando%20pelo%20site%20e%20gostaria%20de%20mais%20informações"

# -------------------------------
# SERP helper + whitelist para externos (autoridades médicas)
# -------------------------------
WHITELIST_EXTERNOS_ANGELICA = [
    ".gov", ".gov.br", ".edu", ".edu.br",
    "sbd.org.br", "aad.org", "who.int", "nhs.uk", "cdc.gov",
    "nih.gov", "ncbi.nlm.nih.gov", "medlineplus.gov",
    "cochranelibrary.com", "dermnetnz.org",
    "schema.org", "w3.org", "developers.google.com", "support.google.com"
]

def _usa_whitelist_angelica(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS_ANGELICA)

def buscar_concorrentes_serpapi_struct(palavra_chave: str) -> list[dict]:
    search = GoogleSearch({
        "q": palavra_chave,
        "hl": "pt-br",
        "gl": "br",
        "num": 10,
        "api_key": os.getenv("SERPAPI_API_KEY")
    })
    d = search.get_dict()
    return d.get("organic_results", []) or []

def selecionar_links_externos_autoritativos(resultados_serp: list[dict], max_links: int = 2) -> list[dict]:
    candidatos, vistos = [], set()
    for r in resultados_serp:
        url = r.get("link") or r.get("url") or ""
        titulo = (r.get("title") or "").strip()
        if not url or url in vistos:
            continue
        if _usa_whitelist_angelica(url):
            candidatos.append({
                "titulo": titulo[:90] or "Fonte externa",
                "url": url,
                "anchor_sugerida": (titulo[:70].lower() or "fonte oficial")
            })
            vistos.add(url)
        if len(candidatos) >= max_links:
            break
    return candidatos

def buscar_concorrentes_serpapi_texto(palavra_chave: str) -> str:
    """Versão textual só para inspiração (NÃO copiar)."""
    results = buscar_concorrentes_serpapi_struct(palavra_chave)
    output = []
    for res in results:
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "") or res.get("url", "")
        output.append(f"Título: {titulo}\nTrecho: {snippet}\nURL: {link}\n")
    return "\n".join(output)

# -------------------------------
# Função principal (Dra. Angélica Bauer)
# -------------------------------
def build_crew_angelica(tema: str, palavra_chave: str):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress, para a
    Dra. Angélica Bauer. 
    Foco EXCLUSIVO em alopecia.
    """
    llm_local = llm

    # Monta referências e links automaticamente
    dados_concorrencia_txt = buscar_concorrentes_serpapi_texto(palavra_chave)
    serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
    links_internos = LINKS_INTERNOS_ANGELICA[:]
    links_externos = selecionar_links_externos_autoritativos(serp_struct, max_links=2)

    # ==== Agentes ====
    agente_intro = Agent(
        role="Redator de Introdução (Alopecia)",
        goal="Escrever introdução clara e acolhedora (2 a 3 parágrafos) com foco exclusivo em alopecia.",
        backstory="Copywriter especializado em saúde capilar e tricologia.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_outline = Agent(
        role="Arquiteto de Estrutura",
        goal="Criar H2/H3 numerados, cobrindo todos os aspectos de alopecia relevantes ao tema.",
        backstory="Especialista em SEO para saúde capilar.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento",
        goal="Desenvolver cada seção com informação técnica, clara e empática sobre alopecia.",
        backstory="Produz conteúdo educativo sem promessas exageradas.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_conclusao = Agent(
        role="Redator de Conclusão",
        goal="Encerrar o texto com resumo claro e objetivo, sem CTA.",
        backstory="Foco em fechamento educativo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_unificador = Agent(
        role="Editor/Unificador de HTML",
        goal="Unir todas as partes em HTML único (body only), coerente e limpo.",
        backstory="Editor técnico de conteúdo para WordPress.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_linkagem = Agent(
        role="Especialista em Linkagem",
        goal="Adicionar links internos e externos de forma natural e distribuída.",
        backstory="Foco em EEAT e SEO médico.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_contato = Agent(
        role="Responsável por Assinatura",
        goal="Anexar assinatura padrão da Dra. Angélica Bauer ao final do HTML.",
        backstory="Mantém a identidade visual e institucional.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_revisor = Agent(
        role="Revisor Sênior PT-BR",
        goal="Listar melhorias objetivas no texto final.",
        backstory="Revisor especializado em textos de saúde.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )
    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as melhorias preservando estrutura e linkagem.",
        backstory="Editor técnico.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva INTRODUÇÃO para '{tema}' usando '{palavra_chave}' 1 vez.
Foco: alopecia. Parágrafos curtos. 1 link interno no 2º parágrafo.
Concorrência para inspiração (NÃO copiar):
{dados_concorrencia_txt}
""",
        expected_output="HTML com 2-3 <p> e 1 link interno.",
        agent=agente_intro
    )
    tarefa_outline = Task(
        description=f"""
Crie H2/H3 numerados para '{tema}'.
Foco: alopecia. 5-7 H2, incluir 'Erros comuns e armadilhas' e 'Exemplos práticos / aplicação'.
Pelo menos 1 heading com '{palavra_chave}'.
""",
        expected_output="Lista com <h2> e <h3> sem conteúdo.",
        agent=agente_outline
    )
    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva cada heading com foco exclusivo em alopecia.
Min 1200 palavras. Parágrafos curtos. Usar listas quando útil.
""",
        expected_output="HTML com <h2>/<h3>/<p>/<ul><li>.",
        agent=agente_desenvolvimento
    )
    tarefa_conclusao = Task(
        description="Conclusão com 1-2 <p>, sem CTA, possivelmente com 1 link interno.",
        expected_output="Conclusão <p>.",
        agent=agente_conclusao
    )
    tarefa_unificar = Task(
        description="Unir tudo em HTML único, sem <h1> e sem imagens.",
        expected_output="HTML final (body only).",
        agent=agente_unificador
    )

    links_internos_txt = "\n".join(
        f"- {li['titulo']}: {li['url']} | âncora sugerida: {li['anchor_sugerida']}"
        for li in links_internos
    )
    links_externos_txt = "\n".join(
        f"- {le['titulo']}: {le['url']} | âncora sugerida: {le['anchor_sugerida']}"
        for le in links_externos
    ) or "(nenhum externo autorizado encontrado)"

    tarefa_linkagem = Task(
        description=f"""
Adicionar links internos e externos.
Internos (usar >=3):
{links_internos_txt}

Externos (>=1 se possível):
{links_externos_txt}
""",
        expected_output="HTML com linkagem aplicada.",
        agent=agente_linkagem
    )
    tarefa_contato = Task(
        description=f"""
Anexar assinatura padrão:
<p>Agende sua consulta e aprenda como manter sua pele jovem, firme e luminosa após os 40!<br>
<a href="{LINK_WHATSAPP_ANGELICA}" target="_blank" rel="noopener noreferrer">{LINK_WHATSAPP_ANGELICA}</a></p>
<p><strong>Dra Angélica Bauer</strong> &nbsp;Dermatologista em Porto Alegre</p>
""",
        expected_output="HTML final com assinatura.",
        agent=agente_contato
    )
    tarefa_revisar = Task(
        description="Revisar ortografia, clareza, SEO e distribuição de links.",
        expected_output="Lista de melhorias.",
        agent=agente_revisor
    )
    tarefa_corrigir = Task(
        description="Aplicar melhorias mantendo estrutura e links.",
        expected_output="HTML revisado.",
        agent=agente_executor
    )

    crew_angelica = Crew(
        agents=[
            agente_intro, agente_outline, agente_desenvolvimento, agente_conclusao,
            agente_unificador, agente_linkagem, agente_contato,
            agente_revisor, agente_executor
        ],
        tasks=[
            tarefa_intro, tarefa_outline, tarefa_desenvolvimento, tarefa_conclusao,
            tarefa_unificar, tarefa_linkagem, tarefa_contato,
            tarefa_revisar, tarefa_corrigir
        ],
        verbose=True
    )
    return crew_angelica

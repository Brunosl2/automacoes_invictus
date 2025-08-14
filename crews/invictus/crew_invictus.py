import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0.4)

# -------------------------------
# Catálogo fixo de links internos (Invictus)
# -------------------------------
LINKS_INTERNOS_INVICTUS = [
    {"titulo": "SEO Técnico", "url": "https://invictus.com.br/seo-tecnico",
     "anchor_sugerida": "boas práticas de SEO técnico"},
    {"titulo": "Marketing de Conteúdo", "url": "https://invictus.com.br/marketing-de-conteudo",
     "anchor_sugerida": "estratégias de marketing de conteúdo"},
    {"titulo": "Gestão de Tráfego", "url": "https://invictus.com.br/gestao-trafego",
     "anchor_sugerida": "gestão de tráfego para aumentar conversões"},
    {"titulo": "Cases", "url": "https://invictus.com.br/cases",
     "anchor_sugerida": "nossos cases de resultados"},
    {"titulo": "Contato", "url": "https://invictus.com.br/contato",
     "anchor_sugerida": "fale com especialistas da Invictus"},
]

# -------------------------------
# SERP helper + whitelist para externos
# -------------------------------
WHITELIST_EXTERNOS = [
    ".gov", ".gov.br", ".edu", ".edu.br",
    "developers.google.com", "support.google.com", "search.google.com",
    "schema.org", "w3.org",
    "moz.com", "ahrefs.com", "semrush.com",
    "who.int", "oecd.org", "unesco.org", "iso.org", "data.gov"
]

def _usa_whitelist(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS)

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
        if _usa_whitelist(url):
            candidatos.append({
                "titulo": titulo[:90] or "Fonte externa",
                "url": url,
                "anchor_sugerida": (titulo[:70].lower() or "fonte oficial")
            })
            vistos.add(url)
        if len(candidatos) >= max_links:
            break
    return candidatos

def buscar_concorrentes_serpapi(palavra_chave: str) -> str:
    """Versão textual para inspiração de tópicos (NÃO copiar)."""
    results = buscar_concorrentes_serpapi_struct(palavra_chave)
    output = []
    for res in results:
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "") or res.get("url", "")
        output.append(f"Título: {titulo}\nTrecho: {snippet}\nURL: {link}\n")
    return "\n".join(output)

# -------------------------------
# Função principal
# -------------------------------
def build_crew_invictus_conteudo(
    tema: str,
    palavra_chave: str,
    links_internos: list[dict] | None = None,
    links_externos: list[dict] | None = None
):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress.

    Estilo de saída (modelo):
    - Introdução com 1–2 links naturais em <p>.
    - <h2> numerados: "1. Título", "2. Título", etc. (<h3> opcionais).
    - Parágrafos curtos (2–4 linhas) e listas <ul><li> quando fizer sentido.
    - Pelo menos UM heading contém a palavra‑chave.
    - Sem <h1>. Sem imagens (<img>, <figure>, <picture>).
    - Mínimo 1200 palavras no total.
    - Linkagem: >=3 internos distribuídos (intro/corpo/conclusão) e >=1 externo (se houver whitelist).
    - Anchors descritivas (nunca “clique aqui”).
    - Links externos: target="_blank" rel="noopener noreferrer".
    - Conclusão sem CTA comercial; CTA fica na assinatura institucional anexada ao final.
    """

    # Defaults de linkagem
    if not links_internos:
        links_internos = LINKS_INTERNOS_INVICTUS[:]

    if links_externos is None:
        serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
        links_externos = selecionar_links_externos_autoritativos(serp_struct, max_links=2)

    dados_concorrencia_txt = buscar_concorrentes_serpapi(palavra_chave)
    llm_local = llm

    # ==== Agentes ====
    agente_intro = Agent(
        role="Redator de Introdução",
        goal="Escrever introdução clara e persuasiva (2–3 parágrafos) no estilo do modelo fornecido, citando a palavra‑chave 1x.",
        backstory="Copywriter sênior B2B; evita clichês; parágrafos curtos; sem imagens.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura (H2/H3) com numeração",
        goal="Definir 5–7 H2 numerados (1., 2., 3., ...), com H3 opcionais; cobrir integralmente a intenção de busca e incluir a palavra‑chave em pelo menos um heading.",
        backstory="Especialista em outline SEO escaneável; nunca usa H1; mantém títulos específicos e não genéricos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento",
        goal="Preencher cada seção com <p> curtos e listas úteis, variar semântica da keyword sem stuffing e sem inserir imagens.",
        backstory="Criador de conteúdo útil, direto, com exemplos práticos e linguagem clara.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão (sem CTA)",
        goal="Encerrar resumindo aprendizados e próximos passos práticos sem convite comercial.",
        backstory="Especialista em fechamentos naturais e objetivos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Unificador de Conteúdo HTML",
        goal="Unir tudo em HTML único (apenas body), coerente, sem redundância, com numeração nos H2 e sem imagens.",
        backstory="Editor técnico focado em semântica e limpeza de HTML.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Planejador e Implementador de Linkagem",
        goal="Inserir links internos/externos de forma natural e distribuída, como no modelo, respeitando todas as regras.",
        backstory="Especialista em internal linking e EEAT.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_contato = Agent(
        role="Responsável por Contato e Assinatura",
        goal="Anexar assinatura institucional da Invictus ao final do HTML (CTA/WhatsApp), sem alterar o conteúdo anterior.",
        backstory="Garante padronização e identidade institucional.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior",
        goal="Listar melhorias objetivas (bullets) em clareza, gramática, estilo do modelo, distribuição de links e regras SEO.",
        backstory="Revisor PT‑BR; corta redundâncias; mantém utilidade e consistência.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as melhorias preservando estrutura semântica e linkagem.",
        backstory="Editor/Dev de HTML limpo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2–3 <p>) para '{tema}' usando a palavra‑chave '{palavra_chave}' apenas 1 vez.
Siga o estilo do MODELO (intro com link(s) natural(is) e menção a autoridade).
Regras:
- PT‑BR; parágrafos curtos (2–4 linhas).
- Sem clichês e sem promessas vazias.
- PROIBIDO: <h1> e qualquer imagem (<img>, <figure>, <picture>).
- Não usar headings na introdução; só <p>.
- Se houver âncora compatível em links_internos, inclua 1 link interno natural no 2º parágrafo (anchor descritiva).
Concorrência (inspiração de tópicos – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com 2–3 <p> (sem imagens) e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}' no estilo do MODELO:
- 5–7 <h2> numerados com prefixo '1. ', '2. ', '3. ' ...
- Até 2 <h3> por <h2> quando fizer sentido (sem numeração).
- Pelo menos UM heading (<h2> ou <h3>) deve conter a palavra‑chave '{palavra_chave}' de forma natural.
- Incluir um H2 equivalente a "Erros comuns e armadilhas" e outro a "Exemplos práticos / aplicação".
- Títulos específicos, claros e não genéricos.
- Nunca usar <h1>. Não incluir conteúdo; só <h2>/<h3>.
Baseie a cobertura na intenção de busca e em lacunas/oportunidades vistas nos concorrentes:
{dados_concorrencia_txt}
""".strip(),
        expected_output="Lista hierárquica com <h2> numerados e <h3> opcionais (sem conteúdo).",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos, mantendo a numeração dos H2:
- Mínimo de 1200 palavras no post completo (será validado no unificador).
- <p> curtos (2–4 linhas); usar <ul><li> quando listar.
- Explicar: o que é, por que importa, como fazer, exemplos reais.
- Variar semântica de '{palavra_chave}' sem stuffing.
- Sem autopromoção e sem CTA.
- PROIBIDO inserir QUALQUER imagem (<img>, <figure>, <picture>).
- Não inventar novos headings; usar apenas os fornecidos.
- Quando fizer sentido, inclua links internos naturais no corpo (anchors descritivas).
Concorrência (inspiração – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com <h2> numerados, <h3> opcionais, <p> e <ul><li> (sem imagens).",
        agent=agente_desenvolvimento
    )

    tarefa_conclusao = Task(
        description="""
Escreva a CONCLUSÃO:
- 1–2 <p> resumindo aprendizados e próximos passos práticos.
- Zero CTA (o CTA fica na assinatura institucional).
- Inclua 1 link interno natural se ainda não houver link na conclusão.
- Não inserir imagens.
""".strip(),
        expected_output="Conclusão em <p>, possivelmente com 1 link interno.",
        agent=agente_conclusao
    )

    tarefa_unificar = Task(
        description="""
Una introdução, corpo e conclusão em um único HTML (conteúdo do body, sem <body>).
Regras:
- Garantir coerência, zero repetição e manter a NUMERAÇÃO dos <h2> (1., 2., 3., ...).
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline, QUALQUER tag de imagem.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body, sem imagens).",
        agent=agente_unificador
    )

    tarefa_linkagem = Task(
        description=f"""
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão) seguindo o estilo do MODELO:
Contexto:
- links_internos: lista de dicionários no formato {{titulo, url, anchor_sugerida}}
- links_externos: lista opcional no formato {{titulo, url, anchor_sugerida}} (já filtrados por whitelist)
Regras:
- >=3 links internos distribuídos: 1 na intro, 1–2 no corpo, 1 na conclusão (se aplicável).
- >=1 link externo autoritativo (se fornecido), no corpo, com target="_blank" rel="noopener noreferrer".
- Âncoras naturais e descritivas; nunca usar "clique aqui".
- Não linkar em headings; apenas <p> e <li>.
- Não quebrar HTML semântico; sem inline style.
- Não adicionar imagens.
Saída: HTML com linkagem aplicada.
""".strip(),
        expected_output="HTML com links internos/externos aplicados (sem imagens).",
        agent=agente_linkagem,
        context=[{"links_internos": links_internos, "links_externos": links_externos}]
    )

    tarefa_contato = Task(
        description="""
Anexar ao FINAL do HTML a assinatura institucional (sem alterar o conteúdo anterior):
<p><strong>👉 Clique e agende uma reunião com nossos especialistas!</strong></p>
<p><a href="https://api.whatsapp.com/send?phone=5511947974924&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank" rel="noopener noreferrer">Fale conosco pelo WhatsApp</a></p>
<p><strong>Invictus Marketing</strong><br>Av. Casa Verde, 751 – São Paulo - SP</p>
""".strip(),
        expected_output="HTML final com assinatura adicionada.",
        agent=agente_contato
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT‑BR; clareza; tom Invictus (profissional, direto, útil).
- Estilo do MODELO: H2 numerados, parágrafos curtos, listas quando úteis, distribuição de links.
- Coerência e distribuição de links; âncoras descritivas; ausência de overstuffing de '{palavra_chave}'.
- Respeito às proibições de imagens e de <h1>.
Saída: lista de melhorias acionáveis em bullets JSON‑like:
- {{"campo":"trecho/resumo","problema":"...","acao":"..."}}
""".strip(),
        expected_output="Bullets com melhorias acionáveis.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="""
Aplique TODAS as melhorias propostas, preservando:
- Estrutura semântica (<h2> numerados/<h3>/<p>/<ul><li>/<a>).
- Linkagem já aplicada (ajuste âncora só se necessário).
- Ausência de imagens e de <h1>.
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only, sem imagens).",
        agent=agente_executor
    )

    # ==== Crew ====
    crew_invictus = Crew(
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
    return crew_invictus

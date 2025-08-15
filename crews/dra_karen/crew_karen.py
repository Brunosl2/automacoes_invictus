import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI


load_dotenv()
llm = ChatOpenAI(temperature=0.4)

def buscar_concorrentes_serpapi(palavra_chave):
    search = GoogleSearch({
        "q": palavra_chave,
        "hl": "pt-br",
        "gl": "br",
        "num": 5,
        "api_key": os.getenv("SERPAPI_API_KEY")
    })
    results = search.get_dict()
    output = []
    for res in results.get("organic_results", []):
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "")
        output.append(f"Título: {titulo}\nTrecho: {snippet}\nURL: {link}\n")
    return "\n".join(output)

# -------------------------------
# Catálogo fixo de links internos (Karen Voltan)
# -------------------------------
LINKS_INTERNOS_KAREN = [
    {
        "titulo": "Home — Dra. Karen Voltan",
        "url": "https://drakarenvoltan.com",
        "anchor_sugerida": "Ortopedia Oncológica em São Paulo"
    },
    {
        "titulo": "Lipossarcomas de Partes Moles",
        "url": "https://drakarenvoltan.com/lipossarcomas-de-partes-moles-na-ortopedia-oncologica",
        "anchor_sugerida": "lipossarcomas de partes moles"
    },
    {
        "titulo": "Tratamento do Mieloma Múltiplo",
        "url": "https://drakarenvoltan.com/tratamento-do-mieloma-multiplo",
        "anchor_sugerida": "tratamento do mieloma múltiplo"
    },
    {
        "titulo": "Câncer nos Ossos: sintomas iniciais e diagnóstico",
        "url": "https://drakarenvoltan.com/cancer-nos-ossos-sintomas-iniciais-diagnostico-e-cuidados-em-ortopedia-oncologica",
        "anchor_sugerida": "sintomas iniciais do câncer nos ossos"
    },
    {
        "titulo": "Ortopedia Oncológica: tratamento do câncer nos ossos",
        "url": "https://drakarenvoltan.com/ortopedia-oncologica-tratamento-do-cancer-nos-ossos",
        "anchor_sugerida": "tratamento do câncer nos ossos"
    },
    {
        "titulo": "Agendamento",
        "url": "https://drakarenvoltan.com/agendamento",
        "anchor_sugerida": "agendar avaliação com a Dra. Karen Voltan"
    },
]

# -------------------------------
# SERP helper + whitelist para externos (autoridades médicas) — Karen
# -------------------------------
WHITELIST_EXTERNOS_KAREN = [
    # TLDs de confiança
    ".gov", ".gov.br", ".edu", ".edu.br",
    # Autoridades de oncologia/ortopedia/saúde
    "who.int",               # OMS
    "nih.gov", "ncbi.nlm.nih.gov", "medlineplus.gov",  # NIH / PubMed / Medline
    "cancer.gov",            # NCI
    "nccn.org",              # NCCN (pautas clínicas)
    "aacr.org",              # American Association for Cancer Research
    "sbot.org.br",           # Sociedade Brasileira de Ortopedia e Traumatologia
    "inca.gov.br",           # Instituto Nacional de Câncer (INCA)
    "ms.gov.br",             # Ministério da Saúde
    "nice.org.uk",           # NICE (diretrizes)
    "bmj.com",               # BMJ
    "nejm.org",              # NEJM (quando houver página aberta)
    "nature.com",            # Nature (materiais abertos)
]

def _usa_whitelist_karen(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS_KAREN)

def selecionar_links_externos_autoritativos_karen(resultados_serp: list[dict], max_links: int = 2) -> list[dict]:
    candidatos, vistos = [], set()
    for r in resultados_serp:
        url = r.get("link") or r.get("url") or ""
        titulo = (r.get("title") or "").strip()
        if not url or url in vistos:
            continue
        if _usa_whitelist_karen(url):
            candidatos.append({
                "titulo": titulo[:90] or "Fonte externa",
                "url": url,
                "anchor_sugerida": (titulo[:70].lower() or "fonte oficial")
            })
            vistos.add(url)
        if len(candidatos) >= max_links:
            break
    return candidatos

# -------------------------------
# Função principal (Karen)
# -------------------------------
def build_crew_karen(tema: str, palavra_chave: str):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress,
    para a Dra. Karen Voltan (Ortopedia Oncológica).

    Estilo de saída:
    - Introdução com 1 a 2 links naturais em <p>.
    - <h2> numerados: "1. ...", "2. ..."; <h3> opcionais.
    - Parágrafos curtos (2 a 4 linhas); listas <ul><li> quando fizer sentido.
    - Pelo menos UM heading contém a palavra‑chave.
    - Sem <h1> e sem imagens.
    - Mínimo 1200 palavras.
    - Linkagem: >=3 internos (intro/corpo/conclusão) e >=1 externo (autoridade).
    - Âncoras descritivas; externos com target="_blank" rel="noopener noreferrer".
    - Conclusão sem CTA; CTA apenas na assinatura final da Dra. Karen.
    - Tom informativo, responsável e empático; sem promessas de cura.
    """
    llm_local = llm

    # SERP (reutiliza seus helpers existentes)
    dados_concorrencia_txt = buscar_concorrentes_serpapi_texto(palavra_chave)
    serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
    links_internos = LINKS_INTERNOS_KAREN[:]
    links_externos = selecionar_links_externos_autoritativos_karen(serp_struct, max_links=2)

    # ==== Agentes (voz ortopedia oncológica) ====
    agente_intro = Agent(
        role="Redator de Introdução (Ortopedia Oncológica)",
        goal="Escrever introdução acolhedora (2 a 3 parágrafos), citando a palavra‑chave 1x.",
        backstory="Copywriter sênior em saúde; linguagem acessível e responsável para pacientes oncológicos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura (H2/H3) numerada",
        goal="Definir 5 a 7 H2 numerados; cobrir intenção de busca do paciente; incluir a palavra‑chave em pelo menos um heading.",
        backstory="Especialista em outline SEO em saúde; títulos informativos, objetivos e éticos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento (Educação em Saúde Oncológica)",
        goal="Preencher cada seção com orientação prática, sem promessas; variar semântica da keyword sem stuffing.",
        backstory="Produz conteúdo claro, com exemplos, listas e linguagem empática; sem autopromoção.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão",
        goal="Encerrar resumindo aprendizados e próximos passos práticos, sem CTA comercial.",
        backstory="Fechamentos objetivos e humanos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Editor/Unificador de HTML",
        goal="Unir tudo em HTML único (apenas body), coerente, sem redundância, mantendo numeração.",
        backstory="Editor técnico focado em semântica limpa para WordPress.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Especialista em Linkagem (EEAT/Oncologia)",
        goal="Inserir links internos/externos de forma natural e distribuída, priorizando autoridade clínica.",
        backstory="Foco em experiência, expertise e confiabilidade (EEAT).",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_contato = Agent(
        role="Responsável por Assinatura (Dra. Karen Voltan)",
        goal="Anexar assinatura institucional ao final do HTML (CTA/Agendamento), sem alterar o conteúdo anterior.",
        backstory="Padronização e identidade da Dra. Karen.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior PT‑BR (Oncologia)",
        goal="Listar melhorias objetivas em clareza, gramática, estilo, linkagem e regras SEO.",
        backstory="Revisor de saúde; corta redundâncias e mantém consistência.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as melhorias preservando estrutura e linkagem.",
        backstory="Editor/Dev de HTML limpo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2 a 3 <p>) para '{tema}' usando a palavra‑chave '{palavra_chave}' apenas 1 vez.
Contexto: ortopedia oncológica; pacientes e familiares.
Regras:
- PT‑BR; parágrafos curtos (2 a 4 linhas); tom empático e responsável.
- Sem clichês e sem promessas.
- PROIBIDO: <h1> e qualquer imagem.
- Não usar headings; apenas <p>.
- Inclua 1 link interno natural no 2º parágrafo (anchor descritiva), se compatível.
Concorrência (inspiração a NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com 2 a 3 <p> e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}':
- 5 a 7 <h2> numerados ('1. ', '2. ', ...).
- Até 2 <h3> por <h2> quando fizer sentido.
- Pelo menos UM heading deve conter a palavra‑chave '{palavra_chave}' de forma natural.
- Incluir um H2 de "Erros comuns e armadilhas" e outro de "Exemplos práticos / aplicação".
- Nunca usar <h1>. Não incluir conteúdo; só <h2>/<h3>.
Baseie a cobertura na intenção de busca dos pacientes (ortopedia oncológica) e lacunas dos concorrentes:
{dados_concorrencia_txt}
""".strip(),
        expected_output="Lista hierárquica com <h2> numerados e <h3> opcionais (sem conteúdo).",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos, mantendo a numeração dos H2:
- Mínimo 1200 palavras no post completo.
- <p> curtos (2 a 4 linhas); usar <ul><li> quando listar.
- Explique: o que é, por que importa, como fazer, exemplos reais/rotina do paciente.
- Variar semântica de '{palavra_chave}' sem keyword stuffing.
- Sem autopromoção e sem CTA.
- PROIBIDO inserir imagens.
- Não inventar novos headings; usar apenas os fornecidos.
- Quando fizer sentido, inclua links internos naturais no corpo (anchors descritivas).
Concorrência (inspiração a NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com <h2> numerados, <h3> opcionais, <p> e <ul><li>.",
        agent=agente_desenvolvimento
    )

    tarefa_conclusao = Task(
        description="""
Escreva a CONCLUSÃO:
- 1 a 2 <p> resumindo aprendizados e próximos passos práticos (ex.: acompanhamento, sinais de alerta, adesão ao plano).
- Zero CTA (o CTA fica na assinatura).
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
- Garantir coerência, zero repetição e manter a NUMERAÇÃO dos <h2>.
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline, QUALQUER imagem.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body).",
        agent=agente_unificador
    )

    # Links (texto) para o agente de linkagem
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
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão) no estilo da Dra. Karen Voltan.

Links internos disponíveis (use pelo menos 3, distribuídos):
{links_internos_txt}

Links externos candidatos (use >=1, se listado; com target="_blank" rel="noopener noreferrer"):
{links_externos_txt}

Regras:
- Distribuição sugerida: 1 link interno na intro, 1 a 2 no corpo, 1 na conclusão (se aplicável).
- Âncoras naturais e descritivas; nunca usar "clique aqui".
- Não linkar em headings; apenas <p> e <li>.
- Não quebrar HTML semântico; sem inline style.
- Não adicionar imagens.
Saída: HTML com linkagem aplicada.
""".strip(),
        expected_output="HTML com links internos/externos aplicados.",
        agent=agente_linkagem
    )

    # Assinatura/CTA da Dra. Karen
    tarefa_contato = Task(
        description="""
Anexar ao FINAL do HTML a assinatura da Dra. Karen (sem alterar o conteúdo anterior):
<p><strong>👉 Agende sua consulta com a Dra. Karen Voltan</strong></p>
<p><a href="https://drakarenvoltan.com/agendamento" target="_blank" rel="noopener noreferrer">Agende sua avaliação online</a></p>
<p><strong>Dra. Karen Voltan — Ortopedista Oncológica</strong><br>Atendimento em São Paulo</p>
""".strip(),
        expected_output="HTML final com assinatura adicionada.",
        agent=agente_contato
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT‑BR; clareza; tom empático e responsável.
- Estilo: H2 numerados, parágrafos curtos, listas quando úteis, distribuição de links.
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
- Linkagem já aplicada (ajuste de âncora apenas se necessário).
- Ausência de imagens e de <h1>.
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only).",
        agent=agente_executor
    )

    crew_karen = Crew(
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
    return crew_karen

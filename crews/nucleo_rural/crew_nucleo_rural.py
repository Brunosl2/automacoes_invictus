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

def build_crew_nucleorural(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redator Agropecuário",
        goal="Criar uma introdução objetiva e direta sobre o problema enfrentado pelo produtor rural",
        backstory="Especialista em comunicação técnica para o setor agro, com foco em introduções práticas e engajadoras para quem vive o campo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Especialista em Conteúdo Agro",
        goal="Desenvolver o corpo do post com explicações claras e soluções práticas",
        backstory="Engenheiro agrônomo/redator experiente no campo, focado em suplementação, sanidade animal e produtividade no rebanho.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Fechamento com CTA Agro",
        goal="Finalizar com reforço nos benefícios práticos e chamada clara para contato",
        backstory="Consultor técnico em nutrição animal, com experiência em conversão de leitores em clientes por meio de CTAs diretos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador de HTML Técnico",
        goal="Organizar o post completo em HTML limpo e coerente para publicação no site da Núcleo Rural",
        backstory="Responsável pela padronização de conteúdo técnico para o agronegócio, garantindo escaneabilidade e estrutura.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor Agropecuário",
        goal="Revisar clareza técnica, coerência, gramática e impacto comercial",
        backstory="Revisor de conteúdo rural com foco em linguagem direta, precisão e aderência ao público produtor.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor Técnico de Revisão",
        goal="Aplicar as correções sugeridas mantendo a estrutura técnica intacta",
        backstory="Responsável por atualizar e finalizar textos técnicos para publicações em mídias rurais.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Rural",
        goal="Ajustar o texto com foco em palavras-chave de busca agropecuária e gerar a meta description",
        backstory="Profissional de SEO com foco em nutrição animal, produtividade e saúde do rebanho.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador para API Núcleo Rural",
        goal="Gerar JSON final com título, meta e conteúdo formatado",
        backstory="Responsável por transformar o conteúdo técnico em um pacote JSON pronto para publicação no WordPress.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva a introdução do post sobre '{tema}', com foco no problema enfrentado pelo produtor rural e mencionando a palavra-chave '{palavra_chave}'.",
            expected_output="Introdução em HTML com 2 parágrafos objetivos, claros e voltados ao pecuarista.",
            agent=agente_intro
        ),
        Task(
            description=f"Desenvolva o corpo do conteúdo com subtítulos <h2>, parágrafos <p> e listas <ul><li>. Use linguagem técnica e mostre como resolver o problema com soluções da Núcleo Rural. Baseie-se neste resumo:\n\n{dados_concorrencia}",
            expected_output="Texto com 800+ palavras explicando os benefícios e funcionamento dos produtos, com foco em resultado prático.",
            agent=agente_meio
        ),
        Task(
            description="Finalize o conteúdo com CTA direto e reforço dos ganhos para o produtor. Mantenha o tom objetivo.",
            expected_output="Parágrafo final em <p> com chamada para contato ou orçamento.",
            agent=agente_conclusao
        ),
        Task(
            description="Unifique as partes em HTML técnico e limpo, com formatação adequada para WordPress.",
            expected_output="HTML único com <h2>, <p>, <ul><li> e fluidez.",
            agent=agente_unificador
        ),
        Task(
            description="Revisar o conteúdo com foco técnico, gramatical e de clareza. Linguagem direta e rural.",
            expected_output="Lista de ajustes pontuais.",
            agent=agente_revisor
        ),
        Task(
            description="Aplicar as revisões mantendo fidelidade ao conteúdo e à estrutura HTML.",
            expected_output="HTML revisado final.",
            agent=agente_executor
        ),
        Task(
            description="Otimizar o post para SEO agropecuário e gerar meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Gerar JSON com campos: titulo, meta_description, html_body. Formatar conteúdo final para API.",
            expected_output="JSON final para publicação automática.",
            agent=agente_finalizador
        )
    ]

    crew = Crew(
        agents=[
            agente_intro, agente_meio, agente_conclusao, agente_unificador,
            agente_revisor, agente_executor, agente_seo, agente_finalizador
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew

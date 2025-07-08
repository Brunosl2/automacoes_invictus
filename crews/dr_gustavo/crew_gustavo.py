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

def build_crew_gustavo(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redator de Introdução Estética",
        goal="Criar uma introdução atrativa com foco em dermatologia moderna e autoestima",
        backstory="Especialista em copywriting para clínicas de estética avançada, com linguagem clara, acolhedora e profissional.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Redator Científico Estético",
        goal="Desenvolver o corpo do post com subtítulos, listas e explicações práticas sobre tratamentos dermatológicos",
        backstory="Jornalista médico especializado em estética e laser, com foco em clareza, autoridade e conteúdo útil.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador com CTA",
        goal="Concluir o post com reforço do valor da clínica e incentivo ao agendamento",
        backstory="Redator focado em gerar confiança, valorizando tecnologia, segurança e naturalidade nos tratamentos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador HTML da Clínica THÁ",
        goal="Unir as seções em um HTML limpo e fluido, pronto para WordPress",
        backstory="Especialista em formatação de conteúdo médico-estético, com foco em clareza visual e harmonia de seções.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor da Clínica THÁ Dermatologia",
        goal="Revisar o conteúdo para clareza, tom profissional e consistência com a marca da clínica",
        backstory="Responsável por manter a linguagem humanizada, confiante e coerente com o padrão dos especialistas Gustavo e Dayana Thá.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as revisões no HTML final",
        backstory="Desenvolvedor com experiência em conteúdo médico, garantindo fidelidade ao texto e estrutura limpa.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO para Dermatologia Estética",
        goal="Ajustar o conteúdo para ranqueamento local em Curitiba, gerar uma meta description convincente",
        backstory="Profissional de SEO com experiência em clínicas de estética, cuidando de palavras-chave, estrutura e conversão orgânica.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Empacotador para API",
        goal="Extrair título, descrição e <body> do HTML final em formato JSON",
        backstory="Responsável por transformar o conteúdo final em um pacote limpo, útil e direto para publicação automática.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Crie a introdução do artigo sobre '{tema}' com foco em tecnologia estética, autoestima e a palavra-chave '{palavra_chave}'. Use o contexto da concorrência:\n\n{dados_concorrencia}",
            expected_output="HTML com dois parágrafos <p> introdutórios, linguagem clara e acolhedora.",
            agent=agente_intro
        ),
        Task(
            description=f"Desenvolva o corpo do artigo com subtítulos <h2>, parágrafos <p> e listas <ul><li>, explicando tratamentos modernos, causas, dicas e resultados. Baseie-se na concorrência:\n\n{dados_concorrencia}",
            expected_output="HTML com 800+ palavras explicando tratamentos, benefícios e cuidados.",
            agent=agente_meio
        ),
        Task(
            description="Conclua o post reforçando a confiança na clínica THÁ e convidando para uma avaliação. Use CTA suave.",
            expected_output="HTML com parágrafo final e chamada para ação confiável.",
            agent=agente_conclusao
        ),
        Task(
            description="Una introdução, corpo e conclusão em um HTML limpo, coeso e com formatação adequada ao WordPress.",
            expected_output="HTML completo e formatado.",
            agent=agente_unificador
        ),
        Task(
            description="Revise o conteúdo HTML com foco em clareza, empatia, autoridade médica e linguagem da THÁ Dermatologia.",
            expected_output="Lista de sugestões de revisão.",
            agent=agente_revisor
        ),
        Task(
            description="Aplique as revisões ao HTML mantendo a integridade do conteúdo.",
            expected_output="HTML revisado final.",
            agent=agente_executor
        ),
        Task(
            description=f"Otimize o conteúdo para SEO local em Curitiba com foco em dermatologia estética e gere uma meta description com até 160 caracteres.",
            expected_output="HTML otimizado e meta description.",
            agent=agente_seo
        ),
        Task(
            description="Extraia título, meta description e conteúdo <body> do HTML final. Formate como JSON para API.",
            expected_output="JSON com: titulo, meta_description, html_body.",
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

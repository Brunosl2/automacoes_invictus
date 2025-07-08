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

def build_crew_francine(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redatora de Introdução",
        goal="Criar uma introdução acolhedora com foco em dermatologia e a palavra-chave",
        backstory="Especialista em comunicação empática em saúde dermatológica, introduzindo temas com clareza e acolhimento.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Redatora de Conteúdo",
        goal="Desenvolver o corpo do artigo com subtópicos claros, listas e orientações práticas",
        backstory="Redatora especializada em saúde da pele e estética, com foco em conteúdo educativo e humanizado.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Redatora de Conclusão",
        goal="Fechar o artigo com resumo, confiança e chamada para ação",
        backstory="Profissional com experiência em artigos médicos, responsável por conclusões suaves e CTA ético e engajador.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificadora de HTML",
        goal="Unir o artigo em HTML limpo e formatado para WordPress",
        backstory="Especialista em formatação de conteúdo médico para web, garantindo legibilidade e padrão visual.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisora da Dra. Francine Costa",
        goal="Revisar o post com foco em clareza, empatia e correção dermatológica",
        backstory="Redatora médica com experiência em revisar textos de dermatologia clínica e estética, sempre mantendo tom humanizado e técnico.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Aplicador de Revisões",
        goal="Aplicar as sugestões mantendo a estrutura HTML e o estilo da clínica",
        backstory="Desenvolvedora de conteúdo médico, especialista em ajustes técnicos e consistência de linguagem.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO para Dermatologia",
        goal="Otimizar o artigo com foco em busca orgânica para clínica dermatológica e gerar uma meta description eficaz",
        backstory="Especialista em SEO médico com foco em clínicas dermatológicas e estética segura, baseado em E-A-T.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Formatadora Final para Publicação",
        goal="Extrair título, meta description e <body> do HTML final para envio via API",
        backstory="Especialista em integração WordPress, entrega conteúdo pronto para automação de publicação.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefa_intro = Task(
        description=f"""Escreva a introdução do artigo sobre '{tema}' com a palavra-chave '{palavra_chave}', em PT-BR.
Use <p> e linguagem acolhedora, considerando o público de uma clínica dermatológica.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Introdução em HTML com dois parágrafos, linguagem acessível e a palavra-chave.",
        agent=agente_intro
    )

    tarefa_meio = Task(
        description=f"""Escreva o corpo do artigo com subtítulos <h2>, parágrafos <p> e listas <ul><li>.
Aborde orientações de cuidados, doenças, estética e protocolos comuns da dermatologia.
Use este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="HTML com ao menos 800 palavras, <h2>, <p>, <ul><li> e foco em clareza dermatológica.",
        agent=agente_meio
    )

    tarefa_conclusao = Task(
        description=f"""Escreva a conclusão do artigo, reforçando os cuidados com a pele e incentivando agendamento com a Dra. Francine Costa.""",
        expected_output="Parágrafos finais em <p> com CTA suave e empático.",
        agent=agente_conclusao
    )

    tarefa_unificar = Task(
        description="Una introdução, corpo e conclusão em um único HTML. Use tags válidas e mantenha fluidez e coerência.",
        expected_output="HTML completo formatado para WordPress.",
        agent=agente_unificador
    )

    tarefa_revisar = Task(
        description="Revise o HTML com foco em clareza, empatia e coerência com a linguagem da Dra. Francine Costa.",
        expected_output="Lista de sugestões claras e objetivas.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="Aplique as revisões mantendo estrutura HTML e tom humanizado.",
        expected_output="HTML revisado e finalizado.",
        agent=agente_executor
    )

    tarefa_seo = Task(
        description=f"""Otimize o HTML final para SEO, com foco em dermatologia clínica e estética.
Gere uma meta description persuasiva e natural. Baseie-se neste resumo:\n\n{dados_concorrencia}""",
        expected_output="HTML otimizado + meta description em PT-BR.",
        agent=agente_seo
    )

    tarefa_finalizar = Task(
        description="Extraia título, meta description e conteúdo <body>. Formate como JSON com os campos 'titulo', 'meta_description' e 'html_body'.",
        expected_output="JSON pronto para API WordPress.",
        agent=agente_finalizador
    )

    crew_francine = Crew(
        agents=[
            agente_intro, agente_meio, agente_conclusao, agente_unificador,
            agente_revisor, agente_executor, agente_seo, agente_finalizador
        ],
        tasks=[
            tarefa_intro, tarefa_meio, tarefa_conclusao,
            tarefa_unificar, tarefa_revisar, tarefa_corrigir,
            tarefa_seo, tarefa_finalizar
        ],
        verbose=True
    )

    return crew_francine

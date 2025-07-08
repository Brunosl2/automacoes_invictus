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

def build_crew_angelica(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redatora de Introdução Científica Acolhedora",
        goal="Criar uma introdução com base científica e linguagem acolhedora sobre o tema proposto",
        backstory="Especialista em introduções para conteúdos médicos com tom profissional e empático, focado em dermatologia.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Redatora Técnica em Dermatologia",
        goal="Desenvolver o conteúdo técnico com subtópicos claros e linguagem acessível, sem perder a precisão médica",
        backstory="Dermatologista-redatora com experiência em transformar ciência dermatológica em conteúdo educativo e confiável.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Conclusão com Confiança Médica",
        goal="Finalizar o artigo com resumo técnico e chamada para ação profissional e ética",
        backstory="Especialista em engajar pacientes com CTA suave, reforçando a credibilidade do atendimento dermatológico.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificadora HTML Médica",
        goal="Unir todas as partes em HTML limpo e adequado para publicação",
        backstory="Profissional de estruturação de conteúdo médico com foco em clareza e formatação WordPress.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisora da Dra. Angélica Bauer",
        goal="Revisar o conteúdo com foco em clareza, acolhimento e linguagem médica precisa",
        backstory="Revisora com experiência em conteúdos científicos e educativos voltados à dermatologia clínica e estética.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar correções mantendo fidelidade técnica e estilo profissional",
        backstory="Responsável por finalizar textos com consistência técnica e empatia para público leigo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Dermatológico",
        goal="Otimizar o conteúdo para SEO com foco em saúde da pele, cabelo e estética em Porto Alegre",
        backstory="Especialista em SEO médico com foco em clínicas dermatológicas e buscas locais de alta conversão.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Formatadora para API",
        goal="Gerar JSON final com título, descrição e corpo HTML formatado",
        backstory="Responsável por empacotar o conteúdo final de forma limpa e padronizada para publicação via API.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva a introdução para o artigo '{tema}' com base científica, linguagem acolhedora e incluindo a palavra-chave '{palavra_chave}'.",
            expected_output="Introdução com dois parágrafos <p> informativos e empáticos.",
            agent=agente_intro
        ),
        Task(
            description=f"Desenvolva o corpo do artigo com <h2>, <p> e listas <ul><li>, explicando com clareza temas como tricologia, dermatologia clínica ou procedimentos estéticos, conforme o tema. Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}",
            expected_output="Corpo técnico com 800+ palavras, linguagem acessível e embasada.",
            agent=agente_meio
        ),
        Task(
            description="Conclua o artigo com um resumo dos benefícios do tratamento e convide o leitor para agendar uma consulta com a Dra. Angélica Bauer.",
            expected_output="Conclusão em <p> com CTA profissional e discreto.",
            agent=agente_conclusao
        ),
        Task(
            description="Unifique o conteúdo em HTML limpo e padronizado para publicação em WordPress.",
            expected_output="HTML final estruturado.",
            agent=agente_unificador
        ),
        Task(
            description="Revisar conteúdo técnico e de linguagem, mantendo empatia e clareza.",
            expected_output="Sugestões de ajustes.",
            agent=agente_revisor
        ),
        Task(
            description="Aplicar as correções mantendo estrutura e tom da dermatologista.",
            expected_output="HTML revisado e final.",
            agent=agente_executor
        ),
        Task(
            description="Otimizar o HTML para SEO dermatológico em Porto Alegre e gerar uma meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Gerar JSON com: titulo, meta_description, html_body.",
            expected_output="JSON final para API.",
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

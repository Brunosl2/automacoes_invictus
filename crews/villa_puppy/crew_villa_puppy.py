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

def build_crew_villapuppy(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redator Pet Lover",
        goal="Criar uma introdução envolvente e carinhosa para tutores de pets",
        backstory="Especialista em escrever com empatia para quem ama animais, iniciando o texto com afeto e conexão com o leitor.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Especialista em Conteúdo Pet",
        goal="Explicar com clareza os serviços da Villa Puppy e como eles beneficiam o pet",
        backstory="Profissional apaixonado por pets e bem-estar animal, com habilidade de descrever serviços como banho, tosa, socialização e venda de filhotes de forma clara e envolvente.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Fechamento com Emoção",
        goal="Concluir com um convite carinhoso para conhecer a Villa Puppy",
        backstory="Especialista em criar chamadas para ação afetivas e empáticas, valorizando a conexão emocional entre pets e tutores.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador HTML para Pet Shop",
        goal="Formatar o conteúdo completo em HTML amigável para blogs de petshop",
        backstory="Profissional com experiência em estruturação de conteúdos afetivos para sites pet friendly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor da Villa Puppy",
        goal="Revisar a linguagem, empatia, clareza e consistência com a identidade da marca",
        backstory="Revisor com olhar cuidadoso para manter o tom acessível, carinhoso e profissional da Villa Puppy.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor Final de Texto",
        goal="Aplicar as correções mantendo o estilo leve e a formatação HTML intacta",
        backstory="Editor com experiência em conteúdos pet, que sabe alinhar clareza e afeto em cada frase.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Petshop",
        goal="Otimizar o conteúdo para buscas relacionadas a pet shop, banho e tosa e filhotes com pedigree",
        backstory="Consultor SEO especializado em e-commerce e pet care, com foco em conversão local para São Paulo e Shopping Villa Lobos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador para API",
        goal="Gerar JSON com título, meta description e HTML formatado para publicação",
        backstory="Responsável por empacotar o conteúdo para publicação automática no site da Villa Puppy.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva a introdução do post sobre '{tema}', usando a palavra-chave '{palavra_chave}'. A linguagem deve ser carinhosa, simpática e voltada a tutores de pets que valorizam cuidado e profissionalismo.",
            expected_output="Introdução em HTML com 2 parágrafos afetivos e envolventes.",
            agent=agente_intro
        ),
        Task(
            description=f"Escreva o corpo do artigo com <h2>, <p> e listas <ul><li>. Descreva serviços como banho, tosa, espaço de socialização, filhotes com pedigree e atendimento veterinário. Use este resumo da concorrência:\n\n{dados_concorrencia}",
            expected_output="Corpo com pelo menos 800 palavras, linguagem acessível e emocional.",
            agent=agente_meio
        ),
        Task(
            description="Conclua o post com CTA convidando o leitor a visitar a Villa Puppy, com tom empático e confiante.",
            expected_output="Parágrafo de encerramento com chamada para ação.",
            agent=agente_conclusao
        ),
        Task(
            description="Una introdução, corpo e conclusão em HTML limpo e bem estruturado para WordPress.",
            expected_output="HTML completo com boa fluidez.",
            agent=agente_unificador
        ),
        Task(
            description="Revise o HTML com foco em tom afetivo, empatia e consistência com a marca Villa Puppy.",
            expected_output="Lista de melhorias sugeridas.",
            agent=agente_revisor
        ),
        Task(
            description="Aplique as revisões sugeridas no HTML, mantendo fidelidade ao tom afetivo e à estrutura.",
            expected_output="HTML final revisado.",
            agent=agente_executor
        ),
        Task(
            description="Ajuste o conteúdo final para SEO local e relevante para termos como pet shop, banho e tosa, filhotes com pedigree. Gere uma meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Analise o HTML completo gerado. Crie um título chamativo e adequado para o artigo, uma meta description envolvente de até 160 caracteres e mantenha o conteúdo HTML original como 'html_body'. Gere um JSON com: titulo, meta_description, html_body.",
            expected_output="JSON pronto para API.",
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

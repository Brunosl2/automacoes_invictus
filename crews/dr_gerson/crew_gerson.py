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

def build_crew_gerson(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redator de Introdução para Saúde Feminina",
        goal="Criar uma introdução acolhedora, profissional e contextualizada sobre o tema",
        backstory="Especialista em introduções humanizadas para blogs médicos com foco em ginecologia e bem-estar feminino.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Redator Técnico de Conteúdo Feminino",
        goal="Explicar o funcionamento, indicações e benefícios de procedimentos ginecológicos ou nutrológicos",
        backstory="Redator com conhecimento em saúde da mulher, capaz de comunicar com clareza e empatia conteúdos técnicos como laser íntimo, nutrologia ou obstetrícia.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador com CTA acolhedor",
        goal="Concluir o artigo reforçando o cuidado com a mulher e convidando à consulta",
        backstory="Profissional com experiência em gerar confiança e direcionar pacientes com empatia e profissionalismo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador de HTML Feminino",
        goal="Organizar o conteúdo completo em HTML limpo e pronto para WordPress",
        backstory="Especialista em formatação de conteúdo médico feminino com foco em ginecologia regenerativa e estética íntima.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor do Dr. Gerson Righetto",
        goal="Revisar o post com foco em empatia, clareza e correção médica",
        backstory="Responsável por revisar artigos de saúde feminina, mantendo equilíbrio entre linguagem técnica e acolhedora.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor de Revisões Técnicas",
        goal="Aplicar as revisões mantendo fidelidade ao conteúdo e clareza visual",
        backstory="Desenvolvedor editorial com experiência em conteúdo ginecológico e nutrológico.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO para Ginecologia e Bem-Estar Feminino",
        goal="Ajustar o conteúdo para bom ranqueamento em Google com foco em saúde íntima, menopausa, laser CO₂ e nutrologia",
        backstory="Consultor de SEO com experiência em clínicas médicas femininas, focado em atração orgânica e relevância local (Curitiba).",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador para API",
        goal="Gerar o JSON final com título, meta description e HTML <body> formatado",
        backstory="Responsável por preparar o conteúdo para automação com WordPress de forma padronizada e limpa.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva uma introdução para o artigo '{tema}' usando a palavra-chave '{palavra_chave}', com linguagem acolhedora, profissional e voltada à saúde íntima e bem-estar feminino.",
            expected_output="2 parágrafos introdutórios em <p> com linguagem clara, empática e em PT-BR.",
            agent=agente_intro
        ),
        Task(
            description=f"Desenvolva o corpo do post com <h2>, <p> e listas <ul><li>. Explique de forma acessível o funcionamento de tecnologias como Laser Íntimo CO₂, nutrologia ou obstetrícia personalizada. Use como base o seguinte resumo da concorrência:\n\n{dados_concorrencia}",
            expected_output="Texto com 800+ palavras, estruturado e informativo.",
            agent=agente_meio
        ),
        Task(
            description="""Escreva a conclusão do post com chamada para ação (CTA) acolhedora e reforço da confiança no Dr. Gerson.
        Inclua, se fizer sentido, os seguintes links de contato em HTML:

        <p><a href="https://api.whatsapp.com/send?phone=5541999380202&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">Agende sua consulta pelo WhatsApp</a></p>
        <p><a href="https://www.instagram.com/dr.gersonrighetto/" target="_blank">Siga o Dr. Gerson no Instagram</a></p>""",
            expected_output="Parágrafo final com CTA para agendamento de consulta, incluindo links para WhatsApp e Instagram, se adequado.",
            agent=agente_conclusao
        ),
        Task(
            description="Unifique o conteúdo em HTML limpo, legível e com estrutura adequada para WordPress.",
            expected_output="HTML final com <h2>, <p> e <ul><li> bem formatados.",
            agent=agente_unificador
        ),
        Task(
            description="Revise o HTML com foco em clareza, linguagem médica apropriada e empatia feminina.",
            expected_output="Sugestões de revisão.",
            agent=agente_revisor
        ),
        Task(
            description="Aplique as revisões mantendo estrutura e integridade do conteúdo.",
            expected_output="HTML revisado e finalizado.",
            agent=agente_executor
        ),
        Task(
            description="Otimizar o HTML para SEO com foco em saúde íntima feminina, bem-estar e localização em Curitiba. Gerar meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Gerar JSON com campos: titulo, meta_description, html_body. Formatar para API de publicação.",
            expected_output="JSON final para WordPress.",
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

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

    agente_meio_h2 = Agent(
        role="Criador de Subtítulos sobre Saúde Feminina",
        goal="Criar subtítulos H2 claros e acolhedores para artigos sobre saúde íntima e ginecologia",
        backstory="Especialista em estruturar conteúdos médicos femininos com foco em clareza e empatia.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedor de Conteúdo para Saúde Feminina",
        goal="Escrever parágrafos explicativos e listas baseados nos subtítulos, sobre saúde da mulher e bem-estar",
        backstory="Profissional especializado em conteúdos de ginecologia e nutrologia, com abordagem acolhedora e técnica.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador de Artigos sobre Saúde Feminina",
        goal="Encerrar o texto reforçando o cuidado com a saúde da mulher, sem chamada para ação direta",
        backstory="Especialista em conclusões institucionais e conteúdos médicos femininos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Gerador de Assinatura Personalizada do Dr. Gerson",
        goal="Adicionar assinatura final personalizada conforme o tema do artigo, mantendo o padrão institucional",
        backstory="Responsável por reforçar a presença institucional do Dr. Gerson em todos os artigos, com tom acolhedor e profissional.",
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
            description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}', com base neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> relacionados à saúde íntima e bem-estar feminino.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> com base nos subtítulos sobre '{tema}'.
Explique os tratamentos e cuidados de forma clara e acolhedora.
Use este resumo como referência:\n\n{dados_concorrencia}""",
            expected_output="HTML explicativo e empático conforme os subtítulos.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando a importância do cuidado com a saúde da mulher e a confiança no atendimento, sem incluir chamada para ação direta.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Conclusão em HTML acolhedora, sem CTA direto.",
            agent=agente_conclusao
        ),

        Task(
            description="""Inclua no final do HTML a seguinte assinatura:

<p><strong>Clique aqui para agendar uma avaliação pelo WhatsApp</strong><br>
<a href="https://api.whatsapp.com/send?phone=5541999380202&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">https://api.whatsapp.com/send?phone=5541999380202&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações.</a></p>

<p><strong>Dr. Gerson Righetto Junior — Ginecologista e Obstetra</strong><br>
Av. Sete de Setembro, 4214 - cj 1707 - Batel, Curitiba – PR</p>""",
            expected_output="HTML com assinatura personalizada do Dr. Gerson Righetto Junior.",
            agent=agente_contato
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
            agente_intro, agente_meio_h2, agente_meio_lista, agente_conclusao,
            agente_contato, agente_unificador, agente_revisor, agente_executor,
            agente_seo, agente_finalizador
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew

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

    agente_meio_h2 = Agent(
        role="Criador de Subtítulos Dermatológicos",
        goal="Criar subtítulos H2 técnicos e claros para artigos sobre tratamentos dermatológicos modernos",
        backstory="Especialista em estruturar conteúdos sobre tecnologia dermatológica, estética e saúde da pele.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedor de Conteúdo Dermatológico",
        goal="Escrever parágrafos explicativos e listas sobre dermatologia moderna, baseando-se em subtítulos",
        backstory="Profissional especializado em conteúdos sobre rosácea, melasma, laser, rejuvenescimento e cuidados capilares.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador de Conteúdos Dermatológicos",
        goal="Encerrar o texto reforçando a importância do cuidado dermatológico, sem chamada para ação direta",
        backstory="Especialista em conclusões técnicas para conteúdos médicos, mantendo tom profissional e informativo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Gerador de Assinatura Personalizada do Dr. Gustavo Thá",
        goal="Criar assinatura final personalizada conforme o tema do artigo, mantendo o padrão institucional e reforçando a autoridade",
        backstory="Responsável pela assinatura oficial dos artigos do Dr. Gustavo, garantindo coerência e presença institucional.",
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

    agente_titulo_meta = Agent(
        role="Gerador de Título e Meta Description",
        goal="Criar título chamativo e meta description envolvente para dermatologia estética",
        backstory="Especialista em headlines para clínicas dermatológicas, com foco em impacto, clareza e SEO.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


    agente_finalizador = Agent(
        role="Empacotador para API",
        goal="Gerar JSON final para API com título, meta description e HTML",
        backstory="Responsável por consolidar o conteúdo final para publicação automática no site.",
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
            description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}', considerando as tendências observadas na concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> relevantes ao tema e ao público dermatológico.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> baseados nos subtítulos sobre '{tema}'.
Use linguagem técnica acessível e destaque as tecnologias e abordagens modernas.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="HTML detalhado e técnico conforme os subtítulos.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando a importância da avaliação dermatológica personalizada e o uso de tecnologias modernas, sem incluir chamada para ação direta.
Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Conclusão profissional em HTML, sem CTA.",
            agent=agente_conclusao
        ),

        Task(
            description="""Inclua no final do HTML a assinatura personalizada conforme o tema, mantendo este formato:

<p><strong>👉 Clique em saiba mais e agende sua consulta com o Dr. Gustavo Thá!</strong><br>
<a href="https://api.whatsapp.com/send?phone=5541991076623&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">https://api.whatsapp.com/send?phone=5541991076623&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações.</a></p>

<p><strong>Dr. Gustavo Thá — Dermatologista, especialista em tecnologias modernas de cuidado com a pele e cabelo em Curitiba</strong></p>""",
            expected_output="HTML com assinatura personalizada do Dr. Gustavo Thá.",
            agent=agente_contato
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
            description="Crie um título chamativo e uma meta description de até 160 caracteres para o conteúdo gerado sobre '{tema}'. O título deve ser impactante e adequado à dermatologia estética. A meta description deve resumir o conteúdo de forma envolvente.",
            expected_output="Título e meta description prontos.",
            agent=agente_titulo_meta
        ),
        Task(
            description="Receba o título, a meta description e o HTML final do artigo. Monte um JSON com os seguintes campos:\n{\n  'titulo': '...',\n  'meta_description': '...',\n  'html_body': '...'\n}\nO campo 'html_body' deve conter o conteúdo completo do artigo em HTML. Não altere nada do conteúdo original.",
            expected_output="JSON pronto para API.",
            agent=agente_finalizador
        ),

    ]

    crew = Crew(
        agents=[
            agente_intro, agente_meio_h2, agente_meio_lista, agente_conclusao,
            agente_contato, agente_unificador, agente_revisor,
            agente_executor, agente_seo, agente_finalizador
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew

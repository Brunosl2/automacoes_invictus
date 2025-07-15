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

def build_crew_karen(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redatora de Introdução Técnica Humanizada",
        goal="Criar uma introdução acessível e profissional com foco em ortopedia oncológica",
        backstory="Especialista em comunicação médica com empatia, capaz de introduzir temas complexos de forma clara e acolhedora.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_h2 = Agent(
        role="Criadora de Subtítulos Médicos",
        goal="Elaborar subtítulos H2 claros e técnicos para artigos sobre ortopedia oncológica e dor",
        backstory="Especialista em estruturar conteúdos médicos para facilitar a leitura e reforçar autoridade.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedora de Conteúdo Médico-Ortopédico",
        goal="Desenvolver parágrafos explicativos e listas baseados nos subtítulos, com clareza e rigor científico",
        backstory="Profissional especializada em conteúdos sobre dor, ortopedia e oncologia, com linguagem técnica acessível.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizadora de Artigos Médicos",
        goal="Concluir o texto reforçando a importância do diagnóstico e cuidado, sem chamada para ação direta",
        backstory="Especialista em encerramentos éticos e profissionais para textos de saúde.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Geradora de Assinatura Médica Personalizada",
        goal="Criar uma assinatura final personalizada conforme o tema do artigo, com foco em ortopedia e tratamentos",
        backstory="Responsável por reforçar a presença institucional da Dra. Karen em todos os artigos, com tom informativo e profissional.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificadora de HTML para Ortopedia Oncológica",
        goal="Organizar todo o conteúdo em HTML limpo, legível e bem estruturado para WordPress",
        backstory="Especialista em formatação de conteúdo médico técnico com acessibilidade e clareza visual.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisora da Dra. Karen Voltan",
        goal="Revisar o conteúdo para garantir correção, empatia e precisão técnica",
        backstory="Responsável por revisar conteúdos de ortopedia, oncologia e dor com foco em linguagem clara, ética e cientificamente rigorosa.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor de Revisões Técnicas",
        goal="Aplicar as correções sugeridas com fidelidade e clareza",
        backstory="Editor técnico com experiência em medicina regenerativa e oncologia musculoesquelética.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Médico-Ortopédico",
        goal="Otimizar o post para pesquisas sobre tumores ósseos, dor musculoesquelética e regeneração tecidual",
        backstory="Profissional de SEO voltado para clínicas especializadas, com foco em oncologia ortopédica e procedimentos não cirúrgicos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador para API WordPress",
        goal="Extrair título, meta_description e html_body formatado",
        backstory="Responsável por transformar conteúdo final em JSON estruturado para publicação automatizada.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva uma introdução clara e acolhedora para o tema '{tema}', com foco em ortopedia oncológica, dor ou regeneração. Inclua a palavra-chave '{palavra_chave}' naturalmente. Use este resumo da concorrência:\n\n{dados_concorrencia}",
            expected_output="2 parágrafos <p> introdutórios em HTML com tom técnico e empático.",
            agent=agente_intro
        ),
        Task(
            description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}', com base nas tendências da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> médicos adequados ao tema.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> com base nos subtítulos sobre '{tema}', abordando causas, diagnóstico e tratamentos.
Use linguagem técnica acessível, baseando-se neste resumo:\n\n{dados_concorrencia}""",
            expected_output="HTML explicativo e detalhado conforme os subtítulos.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando a importância do diagnóstico precoce e do cuidado contínuo, sem incluir chamada para ação direta.
Use este resumo como referência:\n\n{dados_concorrencia}""",
            expected_output="Conclusão em HTML com tom técnico e humanizado, sem CTA.",
            agent=agente_conclusao
        ),

        Task(
            description="""Adicione no final do HTML a seguinte assinatura personalizada conforme o tema do artigo:

<p><strong>Onde realizar o tratamento em São Paulo</strong><br>
Stem Ortopedia Biológica – Av. Brasil, 299, Jardim América – www.stemortopedia.com<br>
Hospital Israelita Albert Einstein – Unidade Klabin – Av. Ricardo Jafet, 1600 – www.einstein.br</p>

<p><strong>Sofre com [resumo do problema]?</strong><br>
Clique aqui agende uma avaliação com a Dra. Karen Voltan e descubra o tratamento mais indicado para você.<br>
<a href="https://drakarenvoltan.com/agendamento" target="_blank">https://drakarenvoltan.com/agendamento</a></p>""",
            expected_output="HTML com assinatura final personalizada da Dra. Karen Voltan.",
            agent=agente_contato
        ),
        Task(
            description="Unifique todo o conteúdo em HTML limpo, organizado e pronto para WordPress.",
            expected_output="HTML completo e coerente.",
            agent=agente_unificador
        ),
        Task(
            description="Revisar o HTML quanto à clareza, ética médica e precisão científica. Sugerir melhorias.",
            expected_output="Sugestões pontuais de revisão.",
            agent=agente_revisor
        ),
        Task(
            description="Aplicar as revisões no HTML final, mantendo fidelidade ao conteúdo original.",
            expected_output="HTML final corrigido.",
            agent=agente_executor
        ),
        Task(
            description=f"Otimizar o HTML para SEO com foco em ortopedia oncológica, regeneração e dor. Gerar uma meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Extrair título, meta_description e conteúdo HTML <body> para API. Formatar como JSON.",
            expected_output="JSON com os campos: titulo, meta_description, html_body.",
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

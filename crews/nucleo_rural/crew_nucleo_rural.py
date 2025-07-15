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

    agente_meio_h2 = Agent(
        role="Criador de Subtítulos Agropecuários",
        goal="Criar subtítulos H2 diretos e técnicos para conteúdos voltados ao produtor rural",
        backstory="Especialista em conteúdos agro, com foco em resultados práticos e clareza técnica.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedor de Conteúdo Agropecuário",
        goal="Desenvolver listas e parágrafos baseados nos subtítulos, explicando soluções para sanidade animal e produtividade",
        backstory="Profissional especializado em conteúdos para o campo, suplementação e manejo zootécnico.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador de Conteúdos Técnicos Rurais",
        goal="Concluir o texto reforçando a importância da solução apresentada, sem chamada direta para ação",
        backstory="Especialista em fechamentos institucionais para conteúdos do setor agropecuário.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Gerador de Assinatura Institucional Núcleo Rural",
        goal="Adicionar assinatura final personalizada conforme o tema do artigo, seguindo o padrão institucional da Núcleo Rural",
        backstory="Responsável por reforçar a presença institucional da Núcleo Rural com assinatura clara e objetiva.",
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
            description=f"""Crie subtítulos <h2> para um artigo agropecuário sobre '{tema}', considerando este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> técnicos e objetivos.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> baseados nos subtítulos, explicando o tema '{tema}'.
Use linguagem técnica e destaque as soluções práticas da Núcleo Rural.
Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="HTML técnico e direto com foco no produtor rural.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando os benefícios e o impacto das soluções apresentadas, sem incluir chamada direta para ação.
Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Conclusão técnica em HTML, sem CTA.",
            agent=agente_conclusao
        ),

        Task(
            description="""Inclua ao final do HTML esta assinatura:

<p><strong>Clique aqui fale com a equipe da Núcleo Rural e transforme o potencial dos seus bezerros em resultados reais!</strong><br>
<a href="https://api.whatsapp.com/send?phone=551735139264&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">https://api.whatsapp.com/send?phone=551735139264&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações.</a></p>""",
            expected_output="HTML final com assinatura personalizada da Núcleo Rural.",
            agent=agente_contato
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
            agente_intro, agente_meio_h2, agente_meio_lista, agente_conclusao,
            agente_contato, agente_unificador, agente_revisor, agente_executor,
            agente_seo, agente_finalizador
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew

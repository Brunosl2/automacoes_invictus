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

def build_crew_tatiana(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redatora de Introdução",
        goal="Criar uma introdução educativa e acolhedora sobre unhas, com a palavra-chave",
        backstory="Jornalista especializada em saúde dermatológica, com foco em acolher o leitor e introduzir conteúdos técnicos de forma acessível.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_h2 = Agent(
        role="Criadora de Subtítulos sobre Saúde das Unhas",
        goal="Criar subtítulos H2 educativos e relevantes para o tema",
        backstory="Especialista em conteúdo médico para público leigo, com foco em estrutura e clareza.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedora de Conteúdo sobre Saúde das Unhas",
        goal="Escrever parágrafos explicativos e listas com base nos subtítulos, com linguagem clara e técnica",
        backstory="Profissional especializada em educação médica e prevenção, com foco em clareza e empatia.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizadora do Conteúdo Médico",
        goal="Concluir o texto reforçando os cuidados com as unhas, sem chamada para ação",
        backstory="Especialista em encerramentos sutis e institucionais, mantendo o tom técnico e acolhedor.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Responsável pela Assinatura da Dra. Tatiana Gabbi",
        goal="Adicionar assinatura padrão e contatos da Dra. Tatiana ao final do HTML",
        backstory="Responsável por garantir a presença do contato oficial da Dra. Tatiana em todos os conteúdos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificadora de HTML",
        goal="Unir as partes do conteúdo em um HTML limpo e formatado para publicação",
        backstory="Profissional experiente em publicações de blogs médicos e científicos, com domínio de HTML semântico e limpo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisora da Dra. Tatiana Gabbi",
        goal="Revisar com foco técnico e clareza, mantendo o tom de autoridade médica e proximidade",
        backstory="Especialista em revisão de conteúdo médico com foco em unhas, mantendo rigor técnico sem perder leveza e didatismo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor Técnico de Revisões",
        goal="Aplicar as sugestões no HTML, mantendo estrutura, clareza e estilo profissional",
        backstory="Responsável pela implementação fiel das correções mantendo o formato e conteúdo adequado à publicação médica.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO para Saúde das Unhas",
        goal="Ajustar o texto para melhor ranqueamento com palavras-chave relevantes e gerar uma meta description atrativa",
        backstory="Consultor de SEO médico especializado em nichos dermatológicos, com foco em unhas, estética e prevenção.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Exportador para API",
        goal="Extrair título, meta description e corpo HTML para publicação via API",
        backstory="Responsável por preparar o conteúdo final de forma padronizada, limpo e pronto para publicação automatizada.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefa_intro = Task(
        description=f"""Escreva a introdução do artigo sobre '{tema}' com a palavra-chave '{palavra_chave}', usando <p>.
Fale sobre a importância da saúde das unhas e introduza o tema com empatia. Use este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="2 parágrafos HTML com introdução clara e acolhedora.",
        agent=agente_intro
    )

    tarefa_meio_h2 = Task(
        description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}' com base neste resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Lista de subtítulos <h2> adequados ao tema.",
        agent=agente_meio_h2
    )

    tarefa_meio_lista = Task(
        description=f"""Com base nos subtítulos fornecidos, desenvolva parágrafos <p> e listas <ul><li> sobre '{tema}'.
Aborde causas, sintomas, hábitos e dicas preventivas. Use este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="HTML com parágrafos e listas explicativos sobre saúde das unhas.",
        agent=agente_meio_lista
    )

    tarefa_conclusao = Task(
        description=f"""Conclua o artigo reforçando a importância dos cuidados com as unhas.
Não inclua chamada para ação.
Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Conclusão em HTML, com parágrafos finais sem CTA.",
        agent=agente_conclusao
    )

    tarefa_contato = Task(
        description="""Inclua ao final do HTML a seguinte assinatura:

<p>Clique aqui agende sua consulta via WhatsApp e receba avaliação especializada:<br>
<a href="https://api.whatsapp.com/send?phone=5511991578420&text=Oi!%20Encontrei%20seu%20contato%20no%20site%20e%20gostaria%20de%20mais%20informações" target="_blank">Fale com a Dra. Tatiana pelo WhatsApp</a></p>

<p>Siga o Instagram: @dratatianagabbi para acompanhar conteúdos exclusivos sobre saúde das unhas e cuidados com a pele no inverno.<br>
<a href="https://www.instagram.com/dratatianagabbi/" target="_blank">Instagram Dra. Tatiana Gabbi</a></p>

<p><strong>Dra. Tatiana Gabbi</strong> — Médica Dermatologista especializada em doenças das unhas em São Paulo</p>""",
        expected_output="HTML final com assinatura e contatos oficiais da Dra. Tatiana Gabbi.",
        agent=agente_contato
    )


    tarefa_unificar = Task(
        description="Una as partes em HTML formatado para WordPress com fluidez e linguagem coerente.",
        expected_output="HTML único, limpo e com >1000 palavras.",
        agent=agente_unificador
    )

    tarefa_revisar = Task(
        description="Revise o conteúdo com foco em clareza, precisão médica e estilo da Dra. Tatiana Gabbi.",
        expected_output="Lista de sugestões objetivas.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="Aplique as revisões no HTML, mantendo fidelidade ao conteúdo original.",
        expected_output="HTML final corrigido e revisado.",
        agent=agente_executor
    )

    tarefa_seo = Task(
        description=f"""Otimize o HTML para SEO com foco em saúde das unhas e gere uma meta description. Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="HTML otimizado + meta description em português.",
        agent=agente_seo
    )

    tarefa_finalizar = Task(
        description="Extraia e retorne o título, a meta description e o conteúdo <body> como JSON.",
        expected_output="JSON com campos: titulo, meta_description, html_body.",
        agent=agente_finalizador
    )

    crew_tatiana = Crew(
        agents=[
            agente_intro, agente_meio_h2, agente_meio_lista, agente_conclusao,
            agente_contato, agente_unificador, agente_revisor, agente_executor,
            agente_seo, agente_finalizador
        ],
        tasks=[
            tarefa_intro, tarefa_meio_h2, tarefa_meio_lista, tarefa_conclusao, tarefa_contato,
            tarefa_unificar, tarefa_revisar, tarefa_corrigir, tarefa_seo, tarefa_finalizar
        ],
        verbose=True
    )

    return crew_tatiana

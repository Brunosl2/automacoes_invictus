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

    agente_meio_h2 = Agent(
        role="Criadora de Subtítulos Dermatológicos",
        goal="Criar subtítulos H2 claros e objetivos sobre dermatologia clínica e estética",
        backstory="Especialista em estruturar conteúdos sobre saúde da pele, envelhecimento e tratamentos estéticos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedora de Conteúdo Dermatológico",
        goal="Escrever parágrafos e listas detalhadas sobre cuidados dermatológicos e protocolos estéticos",
        backstory="Profissional com experiência em dermatologia clínica e estética, focada em conteúdo educativo e humanizado.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizadora de Conteúdos Dermatológicos",
        goal="Encerrar o texto reforçando a importância do cuidado contínuo com a pele, sem chamada direta para ação",
        backstory="Especialista em conclusões técnicas e institucionais para conteúdos médicos dermatológicos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Geradora de Assinatura Personalizada da Dra. Francine Costa",
        goal="Adicionar uma assinatura final personalizada conforme o tema do artigo, seguindo o padrão institucional",
        backstory="Responsável pela assinatura oficial da Dra. Francine, com foco em reforçar autoridade e confiança.",
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

    tarefa_meio_h2 = Task(
        description=f"""Crie subtítulos <h2> para o artigo sobre '{tema}', considerando o público de dermatologia clínica e estética.
    Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Lista de subtítulos <h2> claros e relevantes.",
        agent=agente_meio_h2
    )

    tarefa_meio_lista = Task(
        description=f"""Desenvolva parágrafos <p> e listas <ul><li> com base nos subtítulos sobre '{tema}'.
    Foque em orientação dermatológica, estética, prevenção e cuidados.
    Use este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="HTML com conteúdo detalhado e estruturado.",
        agent=agente_meio_lista
    )

    tarefa_conclusao = Task(
        description=f"""Finalize o artigo reforçando a importância dos cuidados dermatológicos, sem chamada para ação direta.
    Use este resumo da concorrência como referência:\n\n{dados_concorrencia}""",
        expected_output="Conclusão suave e técnica em HTML, sem CTA.",
        agent=agente_conclusao
    )

    tarefa_contato = Task(
        description="""Inclua ao final do HTML esta assinatura personalizada:

    <p><strong>Agende sua consulta e aprenda como manter sua pele jovem, firme e luminosa após os 40!</strong><br>
    <a href="https://api.whatsapp.com/send?phone=5511966189853&text=Oi!%20Encontrei%20seu%20perfil%20no%20Google%20e%20gostaria%20de%20mais%20informações" target="_blank">https://api.whatsapp.com/send?phone=5511966189853&text=Oi!%20Encontrei%20seu%20perfil%20no%20Google%20e%20gostaria%20de%20mais%20informações</a></p>

    <p><strong>Dra Francine Costa — Dermatologista em Porto Alegre</strong></p>""",
        expected_output="HTML final com assinatura personalizada da Dra. Francine Costa.",
        agent=agente_contato
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

    return crew_francine

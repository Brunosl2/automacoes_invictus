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
        role="Introdução Técnica e Informativa",
        goal="Criar uma introdução técnica, objetiva e clara sobre o tema proposto, explicando sua relevância na dermatologia clínica ou tricologia.",
        backstory="Dermatologista especializada em tricologia e dermatologia clínica, que escreve introduções informativas, sem tom comercial ou acolhedor excessivo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_h2 = Agent(
        role="Criadora de Subtítulos Técnicos e Parágrafos Curtos",
        goal="Escrever quatro subtítulos (<h2>), cada um seguido de um parágrafo curto (máximo 60 palavras), com foco técnico e linguagem médica clara.",
        backstory="Dermatologista-redatora que apresenta temas técnicos em blocos curtos e objetivos, ideais para leitura online e entendimento rápido.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Listadora Técnica Explicativa",
        goal="Criar uma lista em HTML com pelo menos cinco itens, explicando cada um de forma detalhada e objetiva, com embasamento médico.",
        backstory="Dermatologista-redatora especializada em criar listas educativas e técnicas, sem tom promocional, focadas em dermatologia clínica e tricologia.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Conclusão Técnica e Ética",
        goal="Finalizar o texto reforçando a importância da avaliação médica especializada, mantendo tom profissional e objetivo.",
        backstory="Dermatologista experiente que conclui artigos ressaltando a necessidade da consulta especializada, sem chamadas apelativas ou tom comercial.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Inseridor de Contatos e Informações Finais",
        goal="Adicionar no final do texto os links de contato, uma frase objetiva e as informações da Dra. Angélica Bauer.",
        backstory="Profissional responsável por inserir contatos e informações de rodapé médico de forma padrão e adequada para publicação.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificadora de Conteúdo HTML Médico",
        goal="Unir todas as partes do texto em HTML limpo e estruturado para WordPress, mantendo clareza e padrão médico.",
        backstory="Especialista em estruturação de conteúdo médico em HTML, garantindo consistência e formatação técnica para publicação.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisora Técnica e Linguagem Médica",
        goal="Revisar o texto garantindo clareza, precisão médica e evitando termos promocionais ou acolhedores demais.",
        backstory="Revisora experiente em dermatologia clínica, especializada em manter a objetividade, evitando termos como 'renomada', 'referência', 'personalizado' ou 'humanizado'.",
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
        role="Especialista em SEO para Dermatologia Técnica",
        goal="Otimizar o conteúdo para SEO, mantendo a clareza médica e adequação para buscas locais em Porto Alegre.",
        backstory="Especialista em SEO médico, focado em dermatologia clínica e tricologia, garantindo otimização sem perder o rigor técnico.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Formatadora Final para API",
        goal="Gerar o JSON final com título, meta description e HTML formatado, conforme padrão de publicação.",
        backstory="Responsável por consolidar o conteúdo técnico em formato padronizado para publicação via API, mantendo clareza e formatação correta.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva a introdução para o tema '{tema}', focando na importância médica da tricologia ou do mapeamento corporal total de nevos. Use linguagem técnica, objetiva e clara. Evite expressões como 'personalizado', 'renomada' ou 'humanizado'.",
            expected_output="Introdução com até dois parágrafos <p>, linguagem técnica, sem apelo promocional.",
            agent=agente_intro
        ),
        Task(
            description=f"Escreva quatro blocos de conteúdo sobre '{tema}'. Cada bloco deve conter um <h2> técnico e um parágrafo <p> curto (até 60 palavras), explicando aspectos relevantes da tricologia ou do mapeamento corporal total de nevos. Linguagem médica, objetiva e clara. Não use termos como 'personalizado' ou 'renomada'.",
            expected_output="Quatro <h2> seguidos de <p> cada, com até 60 palavras por parágrafo.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"Crie uma lista <ul><li> com pelo menos cinco itens sobre '{tema}', cada um acompanhado de uma explicação técnica clara e objetiva. Os itens devem abordar aspectos da tricologia ou do mapeamento corporal total. Evitar expressões promocionais ou acolhedoras demais, especialmente 'personalizado'.",
            expected_output="Lista em HTML com pelo menos cinco itens e explicações técnicas.",
            agent=agente_meio_lista
        ),
        Task(
            description=f"Conclua o artigo reforçando a importância da avaliação médica especializada em tricologia ou no mapeamento corporal total de nevos. Mantenha tom técnico e profissional.",
            expected_output="Conclusão técnica sem CTA, mantendo tom profissional.",
            agent=agente_conclusao
        ),
        Task(
            description="""Inclua ao final do conteúdo o seguinte bloco em HTML, mantendo o padrão abaixo:

        <p>Clique aqui e agende sua consulta com a Dra. Angélica Bauer</p>
        <p><a href="https://api.whatsapp.com/send?phone=5551999216941&text=Oi!%20Encontrei%20seu%20perfil%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">Agende sua consulta pelo WhatsApp</a></p>
        <p><a href="https://www.instagram.com/angelicabauerdermato/" target="_blank">Siga a Dra. Angélica no Instagram</a></p>
        <p>Dra. Angélica Bauer | Dermatologista em Porto Alegre<br>
        Rua 24 de Outubro, 1440, salas 1106/1107, Auxiliadora – Porto Alegre (RS)</p>

        Mantenha o tom objetivo e a estrutura limpa, sem adicionar outras mensagens ou variações de tom.""",
            expected_output="Bloco HTML final com contatos e endereço da Dra. Angélica.",
            agent=agente_contato
        ),

        Task(
            description="Unifique todas as partes do artigo em um HTML limpo, padronizado para WordPress, mantendo clareza, estrutura e tom técnico.",
            expected_output="HTML final estruturado e pronto para publicação.",
            agent=agente_unificador
        ),
        Task(
            description="Revise o conteúdo completo garantindo clareza, precisão médica e adequação ao tom profissional. Certifique-se de que não foram usados termos como 'personalizado', 'renomada', 'referência' ou 'humanizado'.",
            expected_output="Sugestões de ajustes técnicos e de linguagem.",
            agent=agente_revisor
        ),
        Task(
            description="Aplique as correções sugeridas na revisão, mantendo o tom técnico e a objetividade do texto.",
            expected_output="HTML revisado e finalizado.",
            agent=agente_executor
        ),
        Task(
            description="Otimize o conteúdo em HTML para SEO relacionado à tricologia e ao mapeamento corporal total em Porto Alegre. Gere também uma meta description de até 160 caracteres, mantendo o tom técnico.",
            expected_output="HTML otimizado para SEO + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Gere um JSON contendo: título, meta_description e html_body, seguindo o padrão de publicação da API.",
            expected_output="JSON pronto para API.",
            agent=agente_finalizador
        ),

    ]

    crew = Crew(
        agents=[
            agente_intro, agente_meio_h2 , agente_meio_lista, agente_conclusao, agente_contato, agente_unificador,
            agente_revisor, agente_executor, agente_seo, agente_finalizador,
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew

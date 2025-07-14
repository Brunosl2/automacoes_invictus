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

def build_crew_guilherme(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redator de Introdução Médica",
        goal="Criar uma introdução clara, técnica e confiável sobre câncer de pele e diagnóstico precoce",
        backstory="Especialista em comunicação médica, com foco em introduções precisas para temas sensíveis como oncologia cutânea.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio = Agent(
        role="Redator de Conteúdo Técnico",
        goal="Descrever exames, tecnologias e abordagens cirúrgicas com clareza e base científica",
        backstory="Profissional com experiência em conteúdo médico de alta complexidade, com foco em câncer de pele e procedimentos avançados como cirurgia de Mohs.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Encerramento com Autoridade",
        goal="Concluir com reforço da importância da detecção precoce e incentivo à consulta com o Dr. Guilherme",
        backstory="Jornalista médico com habilidade em gerar confiança e orientar o paciente para a tomada de decisão.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador de HTML Médico",
        goal="Organizar todo o conteúdo técnico em HTML limpo e estruturado para WordPress",
        backstory="Especialista em publicação de conteúdo médico, garantindo fluidez entre as seções e marcação correta.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor de Conteúdo do Dr. Guilherme Gadens",
        goal="Revisar clareza técnica, autoridade médica e consistência com a especialidade do Dr. Guilherme",
        backstory="Revisor com experiência em dermatologia oncológica e linguagem ética, precisa e confiável.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor Técnico",
        goal="Aplicar as sugestões de revisão mantendo a integridade e estrutura do HTML",
        backstory="Editor técnico com foco em precisão, fluidez e fidelidade à proposta médica original.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Médico para Oncologia Cutânea",
        goal="Otimizar o post com palavras-chave como cirurgia de Mohs, mapeamento corporal, câncer de pele",
        backstory="Especialista em SEO de nicho médico, com domínio de conteúdo técnico e práticas para ranqueamento orgânico de clínicas especializadas.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador para API",
        goal="Extrair título, descrição e corpo do HTML em formato JSON limpo",
        backstory="Responsável por preparar o conteúdo final para integração automatizada com WordPress.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva a introdução do post sobre '{tema}' com a palavra-chave '{palavra_chave}', ressaltando a importância do diagnóstico precoce e a expertise do Dr. Guilherme Gadens. Use este resumo da concorrência:\n\n{dados_concorrencia}",
            expected_output="HTML com 2 parágrafos introdutórios com autoridade médica.",
            agent=agente_intro
        ),
        Task(
            description=f"Desenvolva o corpo do artigo com <h2>, <p> e <ul><li>, explicando exames como dermatoscopia digital, mapeamento corporal e cirurgia de Mohs. Baseie-se neste resumo:\n\n{dados_concorrencia}",
            expected_output="Corpo do artigo com linguagem técnica, 800+ palavras e estrutura HTML.",
            agent=agente_meio
        ),
        Task(
            description="""Conclua o post com reforço sobre a importância de avaliar lesões precocemente e convite à consulta com o Dr. Guilherme Gadens.
        Inclua, se fizer sentido, os seguintes links de contato em HTML:

        <p><a href="https://api.whatsapp.com/send?phone=5541987877858&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">Agende sua consulta pelo WhatsApp</a></p>
        <p><a href="https://www.instagram.com/gadensguilherme/" target="_blank">Siga o Dr. Guilherme no Instagram</a></p>""",
            expected_output="Conclusão com CTA discreto e profissional, incluindo links para WhatsApp e Instagram, se adequado.",
            agent=agente_conclusao
        ),
        Task(
            description="Una todas as partes em HTML limpo e formatado para WordPress.",
            expected_output="HTML completo e coeso.",
            agent=agente_unificador
        ),
        Task(
            description="Revise o HTML com foco em consistência técnica, linguagem médica e clareza.",
            expected_output="Sugestões de revisão pontuais.",
            agent=agente_revisor
        ),
        Task(
            description="Aplique as revisões mantendo fidelidade ao conteúdo original.",
            expected_output="HTML revisado e final.",
            agent=agente_executor
        ),
        Task(
            description=f"Otimizar o conteúdo final para SEO técnico com foco em câncer de pele e Curitiba. Gerar meta description com até 160 caracteres.",
            expected_output="HTML otimizado e meta description pronta.",
            agent=agente_seo
        ),
        Task(
            description="Analise o conteúdo final em HTML. Crie um título chamativo e técnico para o artigo, uma meta description envolvente com até 160 caracteres e mantenha o HTML do corpo como 'html_body'. Gere um JSON assim:\n{\n  'titulo': '...',\n  'meta_description': '...',\n  'html_body': '...'\n}",
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

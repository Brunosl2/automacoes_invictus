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

    agente_meio_h2 = Agent(
        role="Criador de Subtítulos Médicos",
        goal="Criar subtítulos H2 claros e técnicos sobre câncer de pele, diagnóstico e tratamentos",
        backstory="Especialista em estruturar conteúdos médicos oncológicos com foco em clareza e autoridade.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedor de Conteúdo Médico",
        goal="Escrever parágrafos explicativos e listas baseados nos subtítulos, abordando exames e tratamentos oncológicos",
        backstory="Profissional especializado em conteúdos sobre dermatologia oncológica e cirurgia dermatológica.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador de Conteúdos Médicos",
        goal="Encerrar o artigo reforçando a importância do diagnóstico precoce, sem chamada para ação direta",
        backstory="Especialista em conclusões técnicas para artigos médicos, mantendo tom profissional e informativo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Gerador de Assinatura do Dr. Guilherme Gadens",
        goal="Criar assinatura final personalizada conforme o tema do artigo, mantendo o padrão institucional",
        backstory="Responsável por garantir a presença institucional do Dr. Guilherme em todos os artigos, com tom técnico e acolhedor.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_faq = Agent(
        role="Criador de FAQ Médico",
        goal="Gerar um FAQ relevante com perguntas e respostas sobre o tema do artigo, com foco em dúvidas frequentes de pacientes",
        backstory="Especialista em gerar conteúdos de apoio informativo, com foco em esclarecimento e prevenção em dermatologia oncológica.",
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
            description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}', com base neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> relacionados ao tema oncológico.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> com base nos subtítulos sobre '{tema}', abordando diagnósticos e tratamentos.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="HTML explicativo e detalhado conforme os subtítulos.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando a importância do diagnóstico precoce e do acompanhamento médico, sem CTA.
Use este resumo como referência:\n\n{dados_concorrencia}""",
            expected_output="Conclusão técnica em HTML, sem chamada direta para ação.",
            agent=agente_conclusao
        ),

        Task(
            description="""Adicione ao final do HTML a seguinte assinatura:

<p><strong>👉 Clique em saiba mais e agende sua consulta com o Dr. Guilherme Gadens!</strong><br>
<a href="https://api.whatsapp.com/send?phone=5541987877858&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">https://api.whatsapp.com/send?phone=5541987877858&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações.</a></p>

<p><strong>Dr. Guilherme Gadens — Dermatologista especializado em Cirurgia de Mohs e Dermatoscopia Digital em Curitiba</strong></p>""",
            expected_output="HTML com assinatura personalizada do Dr. Guilherme.",
            agent=agente_contato
        ),

        Task(
            description=f"""Crie um FAQ em HTML relacionado ao tema '{tema}', contendo pelo menos 3 perguntas e respostas objetivas.
Use linguagem clara e técnica, voltada para pacientes em busca de informações sobre diagnóstico, prevenção ou tratamento.
Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Seção FAQ em HTML com perguntas <h3> e respostas <p>.",
            agent=agente_faq
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
            description="Analise o conteúdo final em HTML. Crie um título chamativo e técnico para o artigo, uma meta description envolvente com até 160 caracteres e mantenha o HTML do corpo como 'html_body'. Gere um JSON VÁLIDO assim:\n{\n  \"titulo\": \"...\",\n  \"meta_description\": \"...\",\n  \"html_body\": \"...\"\n}\nNão escreva nada antes ou depois do JSON.",
            expected_output="JSON pronto para API.",
            agent=agente_finalizador
        )

    ]

    crew = Crew(
        agents=[
            agente_intro, agente_meio_h2, agente_meio_lista, agente_conclusao,
            agente_contato, agente_faq, agente_unificador, agente_revisor,
            agente_executor, agente_seo, agente_finalizador
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew

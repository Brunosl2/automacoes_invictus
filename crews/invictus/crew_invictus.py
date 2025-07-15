
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

def build_crew_invictus(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redator de Introducao",
        goal="Criar uma introducao atrativa com a palavra-chave",
        backstory="Especialista em copywriting para blogs, focado em criar introducoes envolventes, contextualizadas e com a palavra-chave",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_h2 = Agent(
        role="Redator de Subtítulos",
        goal="Criar subtítulos H2 relevantes para o tema",
        backstory="Especialista em estruturação de artigos com foco em escaneabilidade e SEO.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Redator de Listas",
        goal="Escrever listas e parágrafos explicativos com base nos subtítulos",
        backstory="Criador de conteúdo focado em listas práticas e explicações objetivas.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão",
        goal="Encerrar o texto reforçando o conteúdo apresentado, sem CTA",
        backstory="Especialista em encerramentos naturais para artigos institucionais.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Responsável por Contato e Assinatura",
        goal="Adicionar assinatura padrão da Invictus e link de WhatsApp no final do HTML",
        backstory="Responsável por garantir a presença da assinatura institucional em todos os posts.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador de Conteudo HTML",
        goal="Unir as partes em um post HTML completo e coerente",
        backstory="Responsavel por garantir fluidez entre as seções e formatacao HTML limpa para WordPress",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor de Blog Post da Invictus",
        goal="Revisar e sugerir melhorias no texto do post",
        backstory="Revisor com foco em clareza, correção gramatical e consistência com a linguagem da marca Invictus.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar as sugestões de revisão no HTML mantendo a estrutura",
        backstory="Desenvolvedor e redator técnico, com foco em HTML limpo e fidelidade ao conteúdo original revisado.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO da Invictus",
        goal="Avaliar o post quanto às melhores práticas de SEO e sugerir uma meta description",
        backstory="Especialista em otimização de conteúdo para motores de busca, com foco em estrutura, palavras-chave e conversão orgânica.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Formatador de Resposta para API",
        goal="Extrair apenas o título, a meta description e o conteúdo do <body> do HTML final corrigido",
        backstory="Você é responsável por formatar a saída final de um post HTML, extraindo apenas as partes úteis para publicação via API.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefa_intro = Task(
        description=f"""Escreva a introducao do artigo sobre '{tema}' com a palavra-chave '{palavra_chave}', em PT-BR.
    Use <p> e linguagem clara, conectando com a dor do leitor. Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Introducao em HTML com dois paragrafos contendo a palavra-chave de forma natural.",
        agent=agente_intro
    )

    tarefa_meio_h2 = Task(
        description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}' com base neste resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Lista de subtítulos <h2> relevantes.",
        agent=agente_meio_h2
    )

    tarefa_meio_lista = Task(
        description=f"""Com base nos subtítulos fornecidos, escreva parágrafos <p> e listas <ul><li> explicativos sobre '{tema}'.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="HTML com parágrafos e listas relacionados aos subtítulos.",
        agent=agente_meio_lista
    )

    tarefa_conclusao = Task(
        description=f"""Escreva a conclusão do artigo sobre '{tema}' sem chamada para ação.
Faça um fechamento natural e coeso, em HTML.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Conclusão em HTML com <p>, sem CTA.",
        agent=agente_conclusao
    )

    tarefa_contato = Task(
        description="""Adicione no final do HTML esta assinatura institucional:
<p>Clique aqui e agende uma reunião com nossos especialistas.<br>
<a href="https://api.whatsapp.com/send?phone=5511947974924&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">Fale conosco pelo WhatsApp</a></p>
<p><strong>Invictus Marketing</strong><br>Av. Casa Verde, 751 – São Paulo - SP</p>""",
        expected_output="HTML final com assinatura e WhatsApp no final.",
        agent=agente_contato
    )


    tarefa_unificar = Task(
        description="Una introducao, corpo e conclusao em um unico HTML com coerencia e fluidez. Use tags válidas e garanta >1000 palavras.",
        expected_output="HTML completo com formatação WordPress.",
        agent=agente_unificador
    )

    tarefa_revisar = Task(
        description=f"""Revise o HTML quanto à gramática, clareza e consistência com o tom da Invictus. Sugira melhorias com base neste resumo:\n\n{dados_concorrencia}""",
        expected_output="Lista de sugestões de revisão.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="Aplique as revisões no HTML, mantendo estrutura e fluidez.",
        expected_output="HTML revisado e pronto para publicação.",
        agent=agente_executor
    )

    tarefa_seo = Task(
        description=f"""Otimize o HTML para SEO (títulos, palavras-chave) e gere meta description. Use o seguinte resumo como referência:\n\n{dados_concorrencia}""",
        expected_output="HTML otimizado + meta description.",
        agent=agente_seo
    )

    tarefa_finalizar = Task(
        description="Extraia <title>, meta description e conteúdo <body>. Retorne como JSON com os campos 'titulo', 'meta_description', 'html_body'.",
        expected_output="JSON com os campos esperados.",
        agent=agente_finalizador
    )

    crew_invictus = Crew(
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

    return crew_invictus


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

    agente_meio = Agent(
        role="Redator de Desenvolvimento",
        goal="Escrever o corpo do artigo com subtopicos claros",
        backstory="Redator experiente em marketing de conteudo, focado em explicacoes claras e organizadas com subtitulos e listas",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusao",
        goal="Escrever a conclusao com chamada para acao",
        backstory="Especialista em encerramentos persuasivos para blog posts com CTA eficaz",
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

    tarefa_meio = Task(
        description=f"""Escreva o corpo do artigo com subtitulos <h2>, paragrafos explicativos <p> e listas <ul><li> com dicas praticas.
    Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="HTML com ao menos 800 palavras, com <h2>, <p> e <ul><li>.",
        agent=agente_meio
    )

    tarefa_conclusao = Task(
        description=f"""Escreva a conclusao do artigo com um resumo e chamada para acao (CTA) clara, em HTML.
    Inclua, se fizer sentido, um link para contato via WhatsApp no seguinte formato:
    <p><a href="https://api.whatsapp.com/send?phone=5511947974924&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank">Fale conosco pelo WhatsApp</a></p>
    Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
        expected_output="Paragrafos finais com CTA em <p> e link para WhatsApp, se adequado.",
        agent=agente_conclusao
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
            agente_intro, agente_meio, agente_conclusao, agente_unificador,
            agente_revisor, agente_executor, agente_seo, agente_finalizador
        ],
        tasks=[
            tarefa_intro, tarefa_meio, tarefa_conclusao,
            tarefa_unificar, tarefa_revisar, tarefa_corrigir,
            tarefa_seo, tarefa_finalizar
        ],
        verbose=True
    )

    return crew_invictus

import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI


load_dotenv()
llm = ChatOpenAI(temperature=0.4)

def buscar_concorrentes_serpapi_resumido(palavra_chave, limite=3):
    search = GoogleSearch({
        "q": palavra_chave,
        "hl": "pt-br",
        "gl": "br",
        "num": limite,
        "api_key": os.getenv("SERPAPI_API_KEY")
    })
    results = search.get_dict()
    output = []
    for res in results.get("organic_results", []):
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        output.append(f"- {titulo}: {snippet}")
    return "\n".join(output)


def build_crew_villapuppy(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi_resumido(palavra_chave)


    agente_intro = Agent(
        role="Redator Pet Lover",
        goal="Criar uma introdução envolvente e carinhosa para tutores de pets",
        backstory="Especialista em escrever com empatia para quem ama animais, iniciando o texto com afeto e conexão com o leitor.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_h2 = Agent(
        role="Criador de Subtítulos para Conteúdo Pet",
        goal="Elaborar subtítulos H2 informativos sobre serviços e cuidados pet",
        backstory="Especialista em conteúdos petshop, com foco em escaneabilidade e atratividade.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedor de Conteúdo Pet",
        goal="Escrever parágrafos explicativos e listas baseados nos subtítulos, destacando serviços e diferenciais",
        backstory="Especialista em conteúdos para pet shop e cuidados com animais, focado em linguagem acessível e envolvente.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizador de Texto Pet",
        goal="Encerrar o texto reforçando os diferenciais, sem chamada direta para ação",
        backstory="Profissional experiente em conclusões institucionais e conteúdos acolhedores para o setor pet.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Responsável pela Assinatura Personalizada Villa Puppy",
        goal="Criar uma assinatura final personalizada de acordo com o tema do artigo, mantendo o padrão institucional",
        backstory="Especialista em comunicação afetiva e institucional da Villa Puppy, garantindo assinatura com destaque adequado ao conteúdo.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificador HTML para Pet Shop",
        goal="Formatar o conteúdo completo em HTML amigável para blogs de petshop",
        backstory="Profissional com experiência em estruturação de conteúdos afetivos para sites pet friendly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisor da Villa Puppy",
        goal="Revisar a linguagem, empatia, clareza e consistência com a identidade da marca",
        backstory="Revisor com olhar cuidadoso para manter o tom acessível, carinhoso e profissional da Villa Puppy.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor Final de Texto",
        goal="Aplicar as correções mantendo o estilo leve e a formatação HTML intacta",
        backstory="Editor com experiência em conteúdos pet, que sabe alinhar clareza e afeto em cada frase.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Petshop",
        goal="Otimizar o conteúdo para buscas relacionadas a pet shop, banho e tosa e filhotes com pedigree",
        backstory="Consultor SEO especializado em e-commerce e pet care, com foco em conversão local para São Paulo e Shopping Villa Lobos.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador para API",
        goal="Gerar JSON com título, meta description e HTML formatado para publicação",
        backstory="Responsável por empacotar o conteúdo para publicação automática no site da Villa Puppy.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"Escreva a introdução do post sobre '{tema}', usando a palavra-chave '{palavra_chave}'. A linguagem deve ser carinhosa, simpática e voltada a tutores de pets que valorizam cuidado e profissionalismo.",
            expected_output="Introdução em HTML com 2 parágrafos afetivos e envolventes.",
            agent=agente_intro
        ),
        Task(
            description=f"""Crie subtítulos <h2> para um post sobre '{tema}', baseado nas tendências da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> relacionados ao tema.",
            agent=agente_meio_h2
        ),
        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> com base nos subtítulos.
Use linguagem afetuosa e informativa, destacando serviços e diferenciais da Villa Puppy.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="HTML com parágrafos e listas relacionados ao tema e subtítulos.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando os diferenciais do tema '{tema}', sem chamada para ação direta.
Baseie-se nas tendências observadas na concorrência:\n\n{dados_concorrencia}""",
            expected_output="Parágrafos finais de conclusão em HTML, sem CTA.",
            agent=agente_conclusao
        ),

        Task(
            description="""Adicione ao final do HTML uma assinatura personalizada, mantendo este formato:
Quer conhecer [destaque relacionado ao tema]? Agende sua visita na Villa Puppy Pet Shop:
📍 Shopping VillaLobos, Av. Dra. Ruth Cardoso, 4777 – Jardim Universidade Pinheiros, São Paulo/SP

Clique aqui e fale conosco agora pelo WhatsApp!
https://api.whatsapp.com/send?phone=5511917411212&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações

Villa Puppy – [chamada final relacionada ao tema]""",
            expected_output="HTML final com assinatura personalizada conforme tema do post.",
            agent=agente_contato
        ),
        Task(
            description="Una introdução, corpo e conclusão em HTML limpo e bem estruturado para WordPress.",
            expected_output="HTML completo com boa fluidez.",
            agent=agente_unificador
        ),
        Task(
            description="Revise o HTML com foco em tom afetivo, empatia e consistência com a marca Villa Puppy.",
            expected_output="Lista de melhorias sugeridas.",
            agent=agente_revisor
        ),
        Task(
            description="Aplique as revisões sugeridas no HTML, mantendo fidelidade ao tom afetivo e à estrutura.",
            expected_output="HTML final revisado.",
            agent=agente_executor
        ),
        Task(
            description="Ajuste o conteúdo final para SEO local e relevante para termos como pet shop, banho e tosa, filhotes com pedigree. Gere uma meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),
        Task(
            description="Analise o HTML completo gerado. Crie um título chamativo e adequado para o artigo, uma meta description envolvente de até 160 caracteres e mantenha o conteúdo HTML original como 'html_body'. Gere um JSON com: titulo, meta_description, html_body.",
            expected_output="JSON pronto para API.",
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

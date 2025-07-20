from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from crews.invictus.crew_invictus import build_crew_invictus
from crews.dra_francine.crew_francine import build_crew_francine
from crews.dra_tati.crew_tati import build_crew_tatiana
from crews.dr_gustavo.crew_gustavo import build_crew_gustavo
from crews.dr_guilherme.crew_guilherme import build_crew_guilherme
from crews.dra_karen.crew_karen import build_crew_karen
from crews.nucleo_rural.crew_nucleo_rural import build_crew_nucleorural
from crews.dr_gerson.crew_gerson import build_crew_gerson
from crews.villa_puppy.crew_villa_puppy import build_crew_villapuppy
from crews.dra_angelica.crew_angelica import build_crew_angelica
from crews.dra_erika.crew_erika import build_crew_erika





app = FastAPI()

@app.get("/invictus")
def executar_crew_invictus(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_invictus(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_francine")
def executar_crew_francine(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_francine(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_tati")
def executar_crew_tatiana(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_tatiana(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dr_gustavo")
def executar_crew_gustavo(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_gustavo(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dr_guilherme")
def executar_crew_guilherme(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_guilherme(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_karen")
def executar_crew_karen(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_karen(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/nucleo_rural")
def executar_crew_nucleorural(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_nucleorural(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dr_gerson")
def executar_crew_gerson(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_gerson(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/villa_puppy")
def executar_crew_villapuppy(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_villapuppy(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())


@app.get("/dra_angelica")
def executar_crew_angelica(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_angelica(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_erika")
def executar_crew_erika(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_erika(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

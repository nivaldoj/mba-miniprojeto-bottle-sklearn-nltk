#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Disciplina: Gestão do Conhecimento e Big Data
Professor: Ricardo Roberto de Lima
Aluno: Nivaldo Mariano de Carvalho Junior

Projeto PaaS com Bottle
"""

from bottle import default_app, template, request, post, get
from bottle import run
from sklearn.naive_bayes import GaussianNB
import joblib


@get('/')
def index():
    # renderiza a página inicial do projeto
    return template("index.html")


@get('/projeto_mamiferos')
def mamiferos_get():
    # renderiza o formulário de classificação de mamíferos
    return template("forms/form_mamifero.html", animal="-", classificacao="-", probabilidade="-")


@post('/projeto_mamiferos')
def mamiferos_post():
    # obtêm valores informados no formulário
    animal = request.forms.get('animal')
    sangue = int(request.forms.get('sangue'))
    bota_ovo = int(request.forms.get('bota_ovo'))
    voa = int(request.forms.get('voa'))
    mora_agua = int(request.forms.get('mora_agua'))

    # carrega o modelo
    modelo_nb = GaussianNB()
    modelo_nb = joblib.load('models/model_mamifero.pkl')

    # executa a classificação
    res = modelo_nb.predict([[sangue, bota_ovo, voa, mora_agua]])

    # encontra o valor da confidência
    prb = modelo_nb.predict_proba([[sangue, bota_ovo, voa, mora_agua]])

    if res == 1:
        clf = "Mamífero"
    elif res == 0:
        clf = "Não mamífero"
    else:
        clf = "Indefinido"

    # renderiza o template
    return template('forms/form_mamifero.html', animal=animal, classificacao=clf, probabilidade=prb)


# executa a aplicação bottle
application = default_app()

# necessário para executar localmente
#run(application, host="localhost", port=80)

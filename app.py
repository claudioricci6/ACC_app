# -*- coding: utf-8 -*-
"""
ACC2 — calcolatore dell'incidenza cumulativa dopo resezione radicale di
carcinoma corticosurrenalico.

Avvio in locale:
    cd app
    streamlit run app.py

Il file è autosufficiente: gli servono solo app.py, modello_app.json e
requirements.txt, e come librerie numpy, pandas, altair e streamlit. Il modello
è esportato in forma numerica da script/08_esporta_app.py, così il
calcolatore non dipende da scikit-survival né dalle versioni esatte con cui è
stato addestrato: è quello che lo rende installabile su un servizio di hosting.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title='Carcinoma corticosurrenalico — rischi competitivi',
                   layout='wide')

# Il file del modello sta accanto a questo script. Cerco anche nella cartella
# superiore e in dati/, così l'app funziona sia dal repository completo sia da
# una cartella piatta in cui è stata copiata per la pubblicazione.
QUI = Path(__file__).resolve().parent
CANDIDATI = [QUI / 'modello_app.json',
             QUI.parent / 'modello_app.json',
             QUI.parent / 'app' / 'modello_app.json',
             QUI.parent / 'dati' / 'modello_app.json']


@st.cache_resource
def carica():
    for p in CANDIDATI:
        if p.exists():
            with open(p, encoding='utf-8') as f:
                m = json.load(f)
            m['griglia'] = np.asarray(m['griglia'], float)
            for c in m['cause'].values():
                c['beta'] = np.asarray(c['beta'], float)
                c['H0'] = np.asarray(c['H0'], float)
            return m
    return None


M = carica()
if M is None:
    st.error('**modello_app.json non trovato.** Deve stare accanto ad app.py. '
             'Per rigenerarlo: `python script/08_esporta_app.py`.')
    st.caption('Cercato in: ' + ' · '.join(str(p) for p in CANDIDATI))
    st.stop()

VAR = M['variabili']
GRID = M['griglia']


def curve(valori):
    """
    Incidenze cumulative di morte per ACC e per altra causa.

    Due modelli di Cox causa-specifici,  H_c(t|x) = H0_c(t)·exp(x·beta_c),
    combinati a tempo discreto: in ogni mese la probabilità di morire è
    S(u-)·[1 - exp(-dH)], ripartita fra le due cause in proporzione ai loro
    incrementi di rischio. Le tre probabilità sommano esattamente a 1.
    """
    x = np.array([float(valori.get(v, M['mediane'][v])) for v in VAR])
    H = {c: M['cause'][c]['H0'] * np.exp(x @ M['cause'][c]['beta']) for c in ('1', '2')}
    dH1 = np.diff(H['1'], prepend=0.0)
    dH2 = np.diff(H['2'], prepend=0.0)
    dH = dH1 + dH2
    S = np.exp(-np.cumsum(dH))
    S_prec = np.concatenate([[1.0], S[:-1]])
    salto = S_prec * (-np.expm1(-dH))
    quota1 = np.divide(dH1, dH, out=np.zeros_like(dH), where=dH > 0)
    return GRID, np.cumsum(salto * quota1), np.cumsum(salto * (1 - quota1))


st.title('Carcinoma corticosurrenalico resecato radicalmente')
st.caption('Stima dell’incidenza cumulativa di morte per carcinoma corticosurrenalico '
           'e di morte per altra causa, da due modelli di Cox causa-specifici su 1.086 '
           'pazienti sottoposti a surrenectomia totale o resezione radicale en bloc nel '
           'registro SEER (2000–2023). Il tempo decorre dall’intervento.')

st.warning(
    '**Strumento di ricerca, non di decisione clinica.** Il modello deriva da un registro '
    'amministrativo, non è stato validato su casistiche esterne e non contiene margini '
    'di resezione, indice Ki-67, punteggio di Weiss, funzionalità ormonale né terapia '
    'con mitotano. Le stime vanno lette come probabilità medie di pazienti con '
    'caratteristiche simili, non come prognosi individuale.', icon='⚠️')

c1, c2 = st.columns([1, 2], gap='large')

with c1:
    st.subheader('Caratteristiche del paziente')
    eta = st.slider('Età alla diagnosi, anni', 18, 90, 55)
    sesso = st.radio('Sesso', ['Femminile', 'Maschile'], horizontal=True)
    stadio = st.selectbox('Stadio alla diagnosi (SEER Summary Stage)',
                          ['Localizzato', 'Regionale', 'Metastatico'])
    dim = st.slider('Diametro del tumore, mm', 10, 300, 100, step=5)
    dim_nota = st.checkbox('Diametro non disponibile', value=False)

    st.markdown('**Intervento e stadiazione patologica**')
    tipo_res = st.radio('Tipo di resezione radicale',
                        ['Surrenectomia totale', 'Resezione allargata en bloc'])
    ln = st.selectbox('Stato linfonodale',
                      ['Non valutati (Nx)', 'Negativi (N0)', 'Positivi (N+)'])
    grado = st.selectbox('Grado istologico',
                         ['Non registrato', 'Basso (G1 / low grade)',
                          'Alto (G2-G4 / high grade)'])

    st.markdown('**Terapie registrate**')
    chemio = st.checkbox('Chemioterapia', value=False)
    rt = st.checkbox('Radioterapia', value=False)

    multipli = st.checkbox('Altri tumori primitivi', value=False)

    with st.expander('Altre variabili del modello'):
        anno = st.slider('Anno di diagnosi', 2000, 2023, 2018)
        lato = st.radio('Lato', ['Sinistro o non specificato', 'Destro'], horizontal=True)
        sede = st.radio('Sede', ['Corticale (C74.0)', 'Surrene NOS (C74.9)'], horizontal=True)
        etnia = st.radio('Etnia', ['Bianca non ispanica', 'Altra'], horizontal=True)
        coniugato = st.checkbox('Coniugato', value=True)

valori = {
    'eta': eta,
    'maschio': 1 if sesso == 'Maschile' else 0,
    'stadio_reg': 1 if stadio == 'Regionale' else 0,
    'stadio_met': 1 if stadio == 'Metastatico' else 0,
    'dim10': dim / 10,
    'dim_mancante': 1 if dim_nota else 0,
    'en_bloc': 1 if tipo_res.startswith('Resezione allargata') else 0,
    'grado_alto': 1 if grado.startswith('Alto') else 0,
    'grado_ignoto': 1 if grado == 'Non registrato' else 0,
    'ln_pos': 1 if ln.startswith('Positivi') else 0,
    'ln_nx': 1 if ln.startswith('Non valutati') else 0,
    'chemio': 1 if chemio else 0,
    'rt': 1 if rt else 0,
    'anno_c': anno - 2012,
    'lato_dx': 1 if lato == 'Destro' else 0,
    'sede_cortex': 1 if sede.startswith('Corticale') else 0,
    'non_bianco': 0 if etnia == 'Bianca non ispanica' else 1,
    'coniugato': 1 if coniugato else 0,
    'tumori_multipli': 1 if multipli else 0,
}

g, cif1, cif2 = curve(valori)
viva = 1 - cif1 - cif2

with c2:
    st.subheader('Incidenza cumulativa prevista')
    k1, k2, k3 = st.columns(3)
    for col, anni_ in zip((k1, k2, k3), (1, 3, 5)):
        m = anni_ * 12
        col.metric(f'Morte per ACC a {anni_} ann{"o" if anni_ == 1 else "i"}',
                   f'{cif1[m]*100:.1f} %')
        col.caption(f'Altra causa {cif2[m]*100:.1f} % · vivo {viva[m]*100:.1f} %')

    tab = pd.DataFrame({'Mesi dall’intervento': g.astype(int),
                        'Morte per ACC': cif1 * 100,
                        'Morte per altra causa': cif2 * 100,
                        'Vivo': viva * 100}).set_index('Mesi dall’intervento')
    lungo = (tab.reset_index()
                .melt('Mesi dall’intervento', var_name='Stato', value_name='%'))
    ordine = ['Morte per ACC', 'Morte per altra causa', 'Vivo']
    st.altair_chart(
        alt.Chart(lungo).mark_area().encode(
            x=alt.X('Mesi dall’intervento:Q', title='Mesi dall’intervento'),
            y=alt.Y('%:Q', stack='zero', title='Pazienti, %',
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Stato:N', sort=ordine,
                            scale=alt.Scale(domain=ordine,
                                            range=['#eb6834', '#eda100', '#1baf7a']),
                            legend=alt.Legend(title=None, orient='bottom')),
            order=alt.Order('Stato:N', sort='ascending'),
            tooltip=[alt.Tooltip('Mesi dall’intervento:Q', title='Mesi'),
                     'Stato:N', alt.Tooltip('%:Q', format='.1f')]
        ).properties(height=380), use_container_width=True)
    st.caption('Le tre aree sommano a 100 a ogni istante: è la scomposizione completa '
               'dello stato del paziente nel tempo.')

    st.markdown('**Valori a tempi fissi**')
    st.dataframe(
        tab.loc[[12, 24, 36, 60, 120]].round(1).rename_axis('Mesi').reset_index(),
        hide_index=True, use_container_width=True)

    # Il grado arriva dopo l'intervento: finché manca, è utile vedere di quanto
    # si sposta la stima nei due scenari possibili.
    if st.checkbox('Mostra quanto pesa il grado, a parità di tutto il resto'):
        scenari = {'Basso grado': {'grado_alto': 0, 'grado_ignoto': 0},
                   'Alto grado': {'grado_alto': 1, 'grado_ignoto': 0},
                   'Grado non registrato': {'grado_alto': 0, 'grado_ignoto': 1}}
        righe = []
        for nome, mod in scenari.items():
            _, a, _ = curve({**valori, **mod})
            righe += [{'Mesi': int(t), 'Scenario': nome, '%': a[int(t)] * 100}
                      for t in g]
        cl = pd.DataFrame(righe)
        st.altair_chart(
            alt.Chart(cl).mark_line(strokeWidth=2).encode(
                x=alt.X('Mesi:Q', title='Mesi dall’intervento'),
                y=alt.Y('%:Q', title='Morte per ACC, incidenza cumulativa %'),
                color=alt.Color('Scenario:N',
                                scale=alt.Scale(domain=list(scenari),
                                                range=['#1baf7a', '#eb6834', '#898781']),
                                legend=alt.Legend(title=None, orient='bottom')),
                tooltip=['Mesi:Q', 'Scenario:N', alt.Tooltip('%:Q', format='.1f')]
            ).properties(height=300), use_container_width=True)
        a5 = {n: curve({**valori, **mo})[1][60] * 100 for n, mo in scenari.items()}
        st.caption(
            f"A 5 anni: basso grado {a5['Basso grado']:.1f} %, alto grado "
            f"{a5['Alto grado']:.1f} %, grado non registrato "
            f"{a5['Grado non registrato']:.1f} %. Nella coorte il grado non registrato "
            'si comporta quasi come l’alto grado (sHR 2,89 contro 3,41): non è una '
            'categoria neutra, è un gruppo a prognosi peggiore della media.')

st.divider()
st.caption('Modello: Cox causa-specifico per morte per ACC e per altra causa, combinati '
           'nell’incidenza cumulativa. Nella validazione interna ripetuta '
           '(25 suddivisioni 70/30) l’indice di concordanza era 0,709, e random survival '
           'forest (0,709) e gradient boosting (0,713) non hanno fatto meglio. '
           'Pendenza di calibrazione a 5 anni 0,91.')

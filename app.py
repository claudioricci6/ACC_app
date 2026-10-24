# -*- coding: utf-8 -*-
"""
ACC2 — app di predizione dell'incidenza cumulativa.

Avvio:
    cd app
    streamlit run app.py

Oppure, da Spyder, eseguire la cella finale di questo file che lancia il
comando al posto tuo.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'script'))
from predizione import Predittore          # noqa: E402

st.set_page_config(page_title='Carcinoma corticosurrenalico — rischio a rischi competitivi',
                   layout='wide')


@st.cache_resource
def carica():
    return Predittore(BASE / 'dati' / 'modello_finale.pkl')


try:
    P = carica()
except FileNotFoundError:
    st.error('Modello non trovato. Esegui prima gli script 01-04 '
             '(oppure `python script/esegui_tutto.py`).')
    st.stop()

st.title('Carcinoma corticosurrenalico resecato')
st.caption('Stima dell’incidenza cumulativa di morte per carcinoma corticosurrenalico '
           'e di morte per altra causa, da un modello di Cox causa-specifico su 1.416 '
           'pazienti resecati del registro SEER (2000–2023). Il tempo decorre '
           'dall’intervento.')

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
                         ['Non registrato', 'Basso (G1 / low grade)', 'Alto (G2-G4 / high grade)'])

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

g, cif1, cif2 = P.curve(valori)
viva = 1 - cif1 - cif2

with c2:
    st.subheader('Incidenza cumulativa prevista')
    k1, k2, k3 = st.columns(3)
    for col, anni_ in zip((k1, k2, k3), (1, 3, 5)):
        m = anni_ * 12
        col.metric(f'Morte per ACC a {anni_} ann{"o" if anni_==1 else "i"}',
                   f'{cif1[m]*100:.1f} %')
        col.caption(f'Altra causa {cif2[m]*100:.1f} % · vivo {viva[m]*100:.1f} %')

    tab = pd.DataFrame({'Mesi dall’intervento': g.astype(int),
                        'Morte per ACC': cif1 * 100,
                        'Morte per altra causa': cif2 * 100,
                        'Vivo': viva * 100}).set_index('Mesi dall’intervento')
    lungo = (tab.reset_index()
                .melt('Mesi dall\u2019intervento', var_name='Stato', value_name='%'))
    ordine = ['Morte per ACC', 'Morte per altra causa', 'Vivo']
    st.altair_chart(
        alt.Chart(lungo).mark_area().encode(
            x=alt.X('Mesi dall\u2019intervento:Q', title='Mesi dall\u2019intervento'),
            y=alt.Y('%:Q', stack='zero', title='Pazienti, %',
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Stato:N', sort=ordine,
                            scale=alt.Scale(domain=ordine,
                                            range=['#eb6834', '#eda100', '#1baf7a']),
                            legend=alt.Legend(title=None, orient='bottom')),
            order=alt.Order('Stato:N', sort='ascending'),
            tooltip=[alt.Tooltip('Mesi dall\u2019intervento:Q', title='Mesi'),
                     'Stato:N', alt.Tooltip('%:Q', format='.1f')]
        ).properties(height=380), use_container_width=True)
    st.caption('Le tre aree sommano a 100 a ogni istante: è la scomposizione completa '
               'dello stato del paziente nel tempo.')

    st.markdown('**Valori a tempi fissi**')
    st.dataframe(
        tab.loc[[12, 24, 36, 60, 120]].round(1)
           .rename_axis('Mesi').reset_index(),
        hide_index=True, use_container_width=True)

    confronto = st.checkbox('Confronta resezione radicale e non radicale '
                            'a parità di tutte le altre variabili')
    if confronto:
        v2 = dict(valori); v2['radicale'] = 1 - valori['radicale']
        _, a1, _ = P.curve({**valori, 'radicale': 1})
        _, b1, _ = P.curve({**valori, 'radicale': 0})
        cc = pd.DataFrame({'Mesi': g.astype(int), 'Radicale': a1 * 100,
                           'Non radicale': b1 * 100}).set_index('Mesi')
        cl = cc.reset_index().melt('Mesi', var_name='Strategia', value_name='%')
        st.altair_chart(
            alt.Chart(cl).mark_line(strokeWidth=2).encode(
                x=alt.X('Mesi:Q', title='Mesi dall\u2019intervento'),
                y=alt.Y('%:Q', title='Morte per ACC, incidenza cumulativa %'),
                color=alt.Color('Strategia:N',
                                scale=alt.Scale(domain=['Totale', 'En bloc'],
                                                range=['#2a78d6', '#eb6834']),
                                legend=alt.Legend(title=None, orient='bottom')),
                tooltip=['Mesi:Q', 'Strategia:N', alt.Tooltip('%:Q', format='.1f')]
            ).properties(height=300), use_container_width=True)
        st.caption(f'Differenza a 5 anni: {(a1[60]-b1[60])*100:+.1f} punti percentuali. '
                   'Nell’analisi causale di questo studio la differenza fra le due '
                   'strategie non era distinguibile da zero (sHR 1,04; IC 95% 0,86–1,26): '
                   'la distanza fra le due curve qui sopra riflette il modello predittivo, '
                   'non un effetto dimostrato della chirurgia.')

st.divider()
st.caption('Modello: Cox causa-specifico per morte per ACC e per altra causa, combinati '
           'nell’incidenza cumulativa. Nella validazione interna ripetuta '
           '(25 suddivisioni 70/30) l’indice di concordanza era 0,69 e random survival '
           'forest e gradient boosting non hanno fatto meglio.')

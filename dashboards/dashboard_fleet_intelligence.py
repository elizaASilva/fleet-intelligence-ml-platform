"""Fleet Intelligence ML Platform — dashboard Streamlit, dados sintéticos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'synthetic' / 'predicted_trips.csv'
EVALUATION = ROOT / 'models' / 'evaluation.json'

st.set_page_config(page_title='Fleet Intelligence | ML', page_icon='🚚', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.5rem; max-width: 1450px}
h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stMetric"] {border:1px solid #e2e8f0;border-radius:12px;padding:14px}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(path, stamp):
    d = pd.read_csv(path)
    needed = {'trip_id','date','route_id','origin_region','destination_region','vehicle_model',
              'capacity_m3','estimated_volume_m3','estimated_occupancy_rate','volume_quality','volume_source'}
    missing = needed - set(d.columns)
    if missing:
        raise ValueError(f'Colunas obrigatórias ausentes: {sorted(missing)}')
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    for c in ['capacity_m3','estimated_volume_m3','estimated_occupancy_rate']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['occupancy_pct'] = 100 * d['estimated_occupancy_rate']
    d['route_label'] = d['route_id'].astype(str) + ' · ' + d['origin_region'].astype(str) + ' → ' + d['destination_region'].astype(str)
    return d

if not DATA.is_file():
    st.error(f'Arquivo não encontrado: {DATA}. Execute o script de treinamento na pasta src antes de abrir o dashboard.')
    st.stop()
try:
    df = load_data(DATA, DATA.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'Falha ao carregar os dados: {exc}')
    st.stop()

st.title('Fleet Intelligence ML Platform')
st.caption('Fleet occupancy · Machine learning · Data quality | Demonstração com dados sintéticos')

with st.sidebar:
    st.header('Filtros globais')
    valid_dates = df['date'].dropna()
    if valid_dates.empty:
        st.error('Datas inválidas no arquivo.')
        st.stop()
    start, end = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.date_input('Período', value=(start,end), min_value=start, max_value=end)
    regions = st.multiselect('Região de origem', sorted(df['origin_region'].dropna().unique()), default=sorted(df['origin_region'].dropna().unique()))
    vehicles = st.multiselect('Modelo de veículo', sorted(df['vehicle_model'].dropna().unique()), default=sorted(df['vehicle_model'].dropna().unique()))
    routes = st.multiselect('Rota', sorted(df['route_id'].dropna().unique()), default=sorted(df['route_id'].dropna().unique()))
    st.caption('Todos os painéis usam o mesmo conjunto filtrado. Ocupação calculada a partir do volume final auditado.')

if len(date_range) != 2:
    st.info('Selecione as duas datas do período.')
    st.stop()
start_filter, end_filter = date_range
f = df.loc[(df['date'].dt.date >= start_filter) & (df['date'].dt.date <= end_filter)
           & df['origin_region'].isin(regions) & df['vehicle_model'].isin(vehicles)
           & df['route_id'].isin(routes)].copy()
if f.empty:
    st.warning('Nenhum registro para os filtros selecionados.')
    st.stop()

# Métricas únicas, compartilhadas por todas as abas.
n = len(f)
mean_occ = f['occupancy_pct'].mean()
critical = int((f['occupancy_pct'] < 20).sum())
low = int((f['occupancy_pct'] < 50).sum())
model_rows = int((f['volume_source'] == 'model').sum())
violations = int(((f['estimated_volume_m3'] > f['capacity_m3'] + 1e-8) | (f['estimated_volume_m3'] < 0)).sum())

pages = st.tabs(['Executive Overview','Fleet Performance','Machine Learning','Data Quality & Audit'])
with pages[0]:
    a,b,c,d,e = st.columns(5)
    a.metric('Viagens',f'{n:,}')
    b.metric('Ocupação média',f'{mean_occ:.1f}%')
    c.metric('Ocupação crítica (<20%)',f'{critical:,}',f'{critical/n:.1%} das viagens',delta_color='off')
    d.metric('Abaixo de 50%',f'{low:,}',f'{low/n:.1%} das viagens',delta_color='off')
    e.metric('Volumes estimados',f'{model_rows:,}',f'{model_rows/n:.1%} das viagens',delta_color='off')
    st.caption('Ocupação crítica e abaixo de 50% são indicadores descritivos; não representam economia financeira comprovada.')
    left,right = st.columns([1.7,1])
    with left:
        daily = f.groupby('date',as_index=False).agg(occupancy_pct=('occupancy_pct','mean'), trips=('trip_id','size'))
        fig = px.line(daily,x='date',y='occupancy_pct',markers=True,title='Evolução da ocupação média',labels={'date':'Data','occupancy_pct':'Ocupação (%)'})
        fig.update_yaxes(range=[0,100]); st.plotly_chart(fig,use_container_width=True)
    with right:
        bins = [-0.001,20,40,60,80,100.001]
        names = ['0–20%','20–40%','40–60%','60–80%','80–100%']
        cluster = pd.cut(f['occupancy_pct'].clip(0,100),bins=bins,labels=names,include_lowest=True)
        counts = cluster.value_counts().reindex(names,fill_value=0).reset_index()
        counts.columns=['Faixa','Viagens']
        fig = px.bar(counts,x='Faixa',y='Viagens',title='Distribuição por faixa de ocupação',color='Faixa',
                     color_discrete_map=dict(zip(names,['#c0392b','#e67e22','#f1c40f','#7cb342','#197d49'])))
        fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
    st.subheader('Ocupação por região de origem')
    reg = f.groupby('origin_region',as_index=False).agg(occupancy_pct=('occupancy_pct','mean'),trips=('trip_id','size'))
    st.plotly_chart(px.bar(reg.sort_values('occupancy_pct'),x='origin_region',y='occupancy_pct',color='trips',
                            labels={'origin_region':'Região','occupancy_pct':'Ocupação média (%)','trips':'Viagens'}),use_container_width=True)

with pages[1]:
    l,r = st.columns(2)
    with l:
        v = f.groupby('vehicle_model',as_index=False).agg(occupancy_pct=('occupancy_pct','mean'),trips=('trip_id','size'))
        st.plotly_chart(px.bar(v.sort_values('occupancy_pct'),x='vehicle_model',y='occupancy_pct',hover_data=['trips'],
                                title='Ocupação por modelo de veículo',labels={'vehicle_model':'Veículo','occupancy_pct':'Ocupação (%)'}),use_container_width=True)
    with r:
        route = f.groupby('route_label',as_index=False).agg(occupancy_pct=('occupancy_pct','mean'),trips=('trip_id','size'))
        route = route.sort_values('occupancy_pct').head(15)
        st.plotly_chart(px.bar(route,x='occupancy_pct',y='route_label',orientation='h',hover_data=['trips'],
                                title='15 rotas com menor ocupação média',labels={'route_label':'Rota','occupancy_pct':'Ocupação (%)'}),use_container_width=True)
    st.subheader('Viagens por rota e sequência')
    seq = f.groupby(['route_label','trip_sequence'],as_index=False).agg(occupancy_pct=('occupancy_pct','mean'),trips=('trip_id','size')) if 'trip_sequence' in f else pd.DataFrame()
    if not seq.empty:
        st.plotly_chart(px.line(seq,x='trip_sequence',y='occupancy_pct',color='route_label',markers=True,
                                hover_data=['trips'],labels={'trip_sequence':'Sequência da viagem','occupancy_pct':'Ocupação média (%)','route_label':'Rota'}),use_container_width=True)
    else: st.info('Coluna trip_sequence indisponível.')

with pages[2]:
    report = {}
    if EVALUATION.is_file():
        try: report = json.loads(EVALUATION.read_text(encoding='utf-8'))
        except (OSError,ValueError): st.warning('Não foi possível ler evaluation.json.')
    winner = report.get('selected_model','Não informado')
    st.subheader('Avaliação do modelo')
    st.write(f'Modelo selecionado: **{winner}**')
    metrics = report.get('holdout_metrics',{})
    if metrics:
        rows = [{'Modelo':k,**v} for k,v in metrics.items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('Métricas de holdout temporal do treinamento, não recalculadas pelos filtros do dashboard.')
    else: st.info('Coloque models/evaluation.json para mostrar métricas de holdout.')
    if {'true_volume_m3','predicted_volume_m3'}.issubset(f.columns):
        st.subheader('Volume real vs. previsão do modelo')
        p = f[['true_volume_m3','predicted_volume_m3','vehicle_model','volume_quality']].dropna()
        fig = px.scatter(p,x='true_volume_m3',y='predicted_volume_m3',color='vehicle_model',opacity=.35,
                         labels={'true_volume_m3':'Volume real sintético (m³)','predicted_volume_m3':'Volume previsto (m³)'})
        top = max(p['true_volume_m3'].max(),p['predicted_volume_m3'].max())
        fig.add_shape(type='line',x0=0,y0=0,x1=top,y1=top,line=dict(color='gray',dash='dash'))
        st.plotly_chart(fig,use_container_width=True)
        st.caption('O volume real é conhecido porque este é um conjunto sintético. A dispersão inclui previsões de treino e teste; não use este gráfico como métrica de generalização.')
        bad = p[p['volume_quality'].isin(['missing','incorrect'])].copy()
        if not bad.empty:
            bad['absolute_error_m3'] = (bad['true_volume_m3']-bad['predicted_volume_m3']).abs()
            st.metric('MAE dos registros com volume ausente/incorreto (todos os períodos)',f"{bad['absolute_error_m3'].mean():.3f} m³")
            st.caption('Diagnóstico descritivo; inclui registros de treinamento. Para desempenho fora da amostra, consulte o holdout.')
    else: st.info('Colunas de referência sintética ou previsão não disponíveis.')

with pages[3]:
    a,b,c,d = st.columns(4)
    a.metric('Volumes observados válidos',f"{(f['volume_quality']=='valid').sum():,}")
    b.metric('Volumes ausentes',f"{(f['volume_quality']=='missing').sum():,}")
    c.metric('Volumes incorretos',f"{(f['volume_quality']=='incorrect').sum():,}")
    d.metric('Violações finais de capacidade',f'{violations:,}')
    l,r = st.columns(2)
    with l:
        q = f['volume_quality'].value_counts().rename_axis('Qualidade').reset_index(name='Viagens')
        st.plotly_chart(px.bar(q,x='Qualidade',y='Viagens',title='Qualidade do volume observado'),use_container_width=True)
    with r:
        s = f['volume_source'].value_counts().rename_axis('Origem').reset_index(name='Viagens')
        st.plotly_chart(px.bar(s,x='Origem',y='Viagens',title='Origem do volume final'),use_container_width=True)
    if 'audit_status' in f:
        st.subheader('Status de auditoria')
        st.dataframe(f['audit_status'].value_counts().rename_axis('Status').reset_index(name='Viagens'),hide_index=True,use_container_width=True)
    else:
        st.warning('Este CSV não possui audit_status. Execute a versão corrigida do treinamento para exibir a auditoria detalhada.')
    st.subheader('Rastreabilidade por viagem')
    cols = [c for c in ['trip_id','date','route_id','vehicle_id','vehicle_model','capacity_m3','volume_quality',
                         'observed_volume_m3','predicted_volume_m3','estimated_volume_m3','estimated_occupancy_rate',
                         'volume_source','audit_status'] if c in f.columns]
    st.dataframe(f[cols],hide_index=True,use_container_width=True)
    st.download_button('Exportar registros filtrados (CSV)',f[cols].to_csv(index=False).encode('utf-8-sig'),
                       file_name='fleet_audit_filtered.csv',mime='text/csv')

st.divider()
st.caption('Projeto demonstrativo — dados inteiramente sintéticos; sem dados internos de empresas. Métricas de ocupação são estimativas, não resultados operacionais verificados.')

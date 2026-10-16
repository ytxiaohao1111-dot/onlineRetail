"""Run: python -m streamlit run app_onlineRetail.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
def load_retail(path='OnlineRetail.csv'):
    raw = pd.read_csv(Path(path), encoding='ISO-8859-1', dtype={'CustomerID': 'string', 'InvoiceNo': 'string', 'StockCode': 'string'})
    df = raw.drop_duplicates().copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'Quantity', 'UnitPrice'])
    df = df.loc[df.UnitPrice > 0].copy()
    df['Revenue'] = df.Quantity * df.UnitPrice
    df['IsReturn'] = df.InvoiceNo.str.upper().str.startswith('C') | df.Quantity.lt(0)
    return raw, df


def customers(df, as_of=None):
    as_of = pd.Timestamp(as_of) if as_of is not None else df.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    history = df.loc[df.InvoiceDate < as_of]
    sales = history.loc[~history.IsReturn & history.Quantity.gt(0) & history.CustomerID.notna()]
    rfm = sales.groupby('CustomerID').agg(LastOrder=('InvoiceDate', 'max'), FirstOrder=('InvoiceDate', 'min'), Frequency=('InvoiceNo', 'nunique'), GrossSpend=('Revenue', 'sum'))
    rfm['NetSpend'] = history.groupby('CustomerID').Revenue.sum().reindex(rfm.index)
    rfm['Country'] = sales.sort_values('InvoiceDate').groupby('CustomerID').Country.last()
    rfm['Recency'] = (as_of.normalize() - rfm.LastOrder.dt.normalize()).dt.days
    # Calendar exposure includes inactive months, not just months with purchases.
    months = (as_of.year-rfm.FirstOrder.dt.year)*12 + as_of.month-rfm.FirstOrder.dt.month + 1
    rfm['MonthlySpend'] = rfm.NetSpend / months.clip(lower=1)
    rfm['Tier'] = pd.cut(rfm.MonthlySpend, [-float('inf'), 1000, 10000, float('inf')], labels=['C', 'B', 'A'], right=False)
    rfm['AtRisk'] = rfm.Tier.isin(['A', 'B']) & rfm.Recency.gt(60)
    return rfm, as_of


st.set_page_config(page_title='Wholesale retention', layout='wide')
st.title('Wholesale customer retention')
st.write('Decision: call high-value accounts that have not ordered for more than 60 days.')

@st.cache_data
def data():
    return load_retail(Path(__file__).with_name('OnlineRetail.csv'))

raw, df = data()
rfm, as_of = customers(df)
st.caption(f'Frozen snapshot: {df.InvoiceDate.min():%Y-%m-%d} to {df.InvoiceDate.max():%Y-%m-%d}. Recency as of {as_of:%Y-%m-%d}; not live data. Currency: GBP.')
countries = st.sidebar.multiselect('Country', sorted(rfm.Country.dropna().unique()))
tiers = st.sidebar.multiselect('Customer tier', ['A', 'B', 'C'], default=['A', 'B', 'C'])
st.sidebar.caption('A: ≥£10,000/month; B: £1,000–9,999/month; C: <£1,000/month. Wholesale is a spending proxy: no wholesale flag exists.')
selected = rfm.loc[rfm.Tier.isin(tiers)]
if countries:
    selected = selected.loc[selected.Country.isin(countries)]
tx = df.loc[df.CustomerID.isin(selected.index)]
if countries:
    tx = tx.loc[tx.Country.isin(countries)]
month = df.InvoiceDate.max().to_period('M')
current = tx.loc[tx.InvoiceDate.dt.to_period('M').eq(month)]
risk = selected.loc[selected.AtRisk].sort_values(['MonthlySpend', 'Recency'], ascending=False)
a,b,c = st.columns(3)
a.metric(f'{month} net revenue (partial month)', f'£{current.Revenue.sum():,.0f}')
b.metric('Active high-value accounts this month', current.loc[~current.IsReturn & current.CustomerID.isin(selected.loc[selected.Tier.isin(['A','B'])].index), 'CustomerID'].nunique())
c.metric('Accounts to call (>60 days)', len(risk))
st.caption('Metrics follow selected customer cohorts. Revenue includes signed returns; anonymous sales are excluded. The final month is incomplete.')
if selected.empty:
    st.info('No customers match these filters. Select another country or tier.')
else:
    st.subheader('Monthly net revenue')
    monthly = tx.set_index('InvoiceDate').Revenue.resample('MS').sum()
    st.line_chart(monthly.rename('Net revenue (£)'))
    st.subheader('Customer recency and lifetime net spend')
    st.scatter_chart(selected.reset_index(), x='Recency', y='NetSpend', color='Tier')
    st.caption(f'Median customer net spend: £{selected.NetSpend.median():,.0f}. Large orders create a long tail; averages can mislead. Recency is in days.')
st.subheader('Monday call list')
st.dataframe(risk.reset_index()[['CustomerID','Country','Tier','LastOrder','Recency','MonthlySpend','NetSpend','Frequency']], hide_index=True)
st.download_button('Download call list', risk.to_csv().encode('utf-8'), 'retention_call_list.csv', 'text/csv')
st.caption('Production proposal: ingest orders daily, review every Monday, and alert on newly overdue A accounts. Negative quantities reduce net revenue; cancellations do not reset last purchase.')

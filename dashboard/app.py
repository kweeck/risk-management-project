import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── загрузка данных ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    with open('C:/Users/kweec/python_files/risk-project/models/model_lr_v2.pkl', 'rb') as f:
        model_lr = pickle.load(f)
    with open('C:/Users/kweec/python_files/risk-project/models/scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model_lr, scaler

@st.cache_data
def load_portfolio():
    con = duckdb.connect(
        'C:/Users/kweec/python_files/risk-project/notebooks/credit_risk.db',
        read_only=True
    )
    return con.execute("SELECT * FROM mart.features").df()

@st.cache_data
def compute_pd(_model_lr, _scaler, df):
    FEATURES = [
        'dti', 'fico_avg', 'annual_inc', 'loan_amnt',
        'installment', 'open_acc', 'revol_util', 'total_acc',
        'loan_to_income', 'payment_to_income',
        'term_months', 'home_ownership', 'purpose'
    ]
    df_port = df[FEATURES + ['ead', 'is_default', 'issue_date']].dropna()
    df_port = pd.get_dummies(df_port,
                              columns=['home_ownership', 'purpose'],
                              drop_first=True)
    FEATURES_ENC = [c for c in df_port.columns
                    if c not in ['ead', 'is_default', 'issue_date']]
    df_port['pd_pred'] = _model_lr.predict_proba(
        _scaler.transform(df_port[FEATURES_ENC])
    )[:, 1]
    df_port['lgd'] = 0.697
    df_port['el']  = df_port['pd_pred'] * df_port['lgd'] * df_port['ead']
    return df_port

def run_montecarlo(pds, lgds, eads, rho=0.15, n_sim=5000, conf=0.99):
    np.random.seed(42)
    K      = scipy_stats.norm.ppf(np.clip(pds, 0.001, 0.999))
    losses = np.zeros(n_sim)
    for i in range(n_sim):
        Z        = np.random.normal(0, 1)
        epsilon  = np.random.normal(0, 1, len(pds))
        Y        = np.sqrt(rho) * Z + np.sqrt(1 - rho) * epsilon
        defaults = (Y < K).astype(int)
        losses[i] = np.sum(defaults * lgds * eads)
    el  = np.mean(losses)
    var = np.percentile(losses, conf * 100)
    es  = losses[losses >= var].mean()
    return el, var, es, losses

model_lr, scaler = load_models()
df               = load_portfolio()
df_port          = compute_pd(model_lr, scaler, df)

# ── заголовок ────────────────────────────────────────────────────────────────

st.title("Credit Risk Pipeline")
st.caption("Lending Club 2007–2018  |  Logistic Regression PD  |  ASRF Monte Carlo VaR")

tab1, tab2, tab3 = st.tabs([
    "Portfolio Overview",
    "Model Performance",
    "Stress Testing"
])

# ── вкладка 1: portfolio overview ────────────────────────────────────────────

with tab1:
    st.subheader("Portfolio metrics")

    el_total  = df_port['el'].sum()
    ead_total = df_port['ead'].sum()
    pd_mean   = df_port['pd_pred'].mean()
    n_loans   = len(df_port)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Кредитов в портфеле",  f"{n_loans:,}")
    c2.metric("Общий EAD",            f"${ead_total/1e9:.2f}B")
    c3.metric("Expected Loss",         f"${el_total/1e9:.3f}B")
    c4.metric("Средний PD",           f"{pd_mean:.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Распределение PD**")
        fig = px.histogram(
            df_port.sample(50000, random_state=42),
            x='pd_pred', nbins=60,
            color='is_default',
            barmode='overlay',
            opacity=0.6,
            labels={'pd_pred': 'PD', 'is_default': 'Дефолт'},
            color_discrete_map={0: 'steelblue', 1: 'tomato'}
        )
        fig.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**EL по сегментам (purpose)**")
        el_by_purpose = (
            df_port.groupby('purpose')['el']
            .sum()
            .sort_values(ascending=True)
            .reset_index()
        ) if 'purpose' in df_port.columns else None

        if el_by_purpose is not None:
            fig2 = px.bar(
                el_by_purpose, x='el', y='purpose',
                orientation='h',
                labels={'el': 'Expected Loss', 'purpose': ''},
            )
            fig2.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("purpose не доступен после get_dummies")

    st.markdown("**EL по времени (по дате выдачи)**")
    df_port['year'] = pd.to_datetime(df_port['issue_date']).dt.year
    el_by_year = df_port.groupby('year')['el'].sum().reset_index()
    fig3 = px.bar(el_by_year, x='year', y='el',
                  labels={'el': 'Expected Loss', 'year': 'Год'})
    fig3.update_layout(height=300, margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)

# ── вкладка 2: model performance ─────────────────────────────────────────────

with tab2:
    st.subheader("Model performance")

    from sklearn.metrics import roc_auc_score, roc_curve

    auc  = roc_auc_score(df_port['is_default'], df_port['pd_pred'])
    gini = 2 * auc - 1
    ks   = scipy_stats.ks_2samp(
        df_port[df_port['is_default']==1]['pd_pred'],
        df_port[df_port['is_default']==0]['pd_pred']
    ).statistic

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC",  f"{auc:.4f}")
    c2.metric("Gini", f"{gini:.4f}")
    c3.metric("KS",   f"{ks:.4f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ROC-кривая**")
        sample = df_port.sample(100000, random_state=42)
        fpr, tpr, _ = roc_curve(sample['is_default'], sample['pd_pred'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'ROC (AUC={auc:.3f})',
            line=dict(color='steelblue', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Random'
        ))
        fig_roc.update_layout(
            xaxis_title='FPR', yaxis_title='TPR',
            height=350, margin=dict(t=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.markdown("**Gini по годам**")
        yearly = []
        df_port['year'] = pd.to_datetime(df_port['issue_date']).dt.year
        for year in sorted(df_port['year'].unique()):
            sub = df_port[df_port['year'] == year]
            if sub['is_default'].sum() < 100:
                continue
            a = roc_auc_score(sub['is_default'], sub['pd_pred'])
            yearly.append({'year': year, 'Gini': round(2*a-1, 3)})
        df_yearly = pd.DataFrame(yearly)
        fig_gini = px.line(df_yearly, x='year', y='Gini', markers=True)
        fig_gini.add_hline(y=0.3, line_dash='dash',
                            line_color='red',
                            annotation_text='Min acceptable')
        fig_gini.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig_gini, use_container_width=True)

    st.markdown("**Калибровка модели**")
    from sklearn.calibration import calibration_curve
    sample2 = df_port.sample(50000, random_state=42)
    frac, mean_pd = calibration_curve(
        sample2['is_default'], sample2['pd_pred'], n_bins=15
    )
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=mean_pd, y=frac, mode='lines+markers',
        name='Logit', line=dict(color='steelblue')
    ))
    fig_cal.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Идеальная'
    ))
    fig_cal.update_layout(
        xaxis_title='Предсказанный PD',
        yaxis_title='Реальная доля дефолтов',
        height=300, margin=dict(t=20)
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ── вкладка 3: stress testing ─────────────────────────────────────────────────

with tab3:
    st.subheader("Stress testing — scenario analysis")

    st.markdown("Выбери сценарий стресса по PD портфеля:")

    base_pd = df_port['pd_pred'].mean()

    scale = st.slider(
        "Коэффициент роста PD (1.0 = базовый)",
        min_value=1.0, max_value=6.0,
        value=1.0, step=0.5
    )

    scenarios_fixed = {
        'Базовый':              1.0,
        'Стресс 1 (+50%)':      1.5,
        'Стресс 2 (+150%)':     2.5,
        'Стресс 3 — кризис':    5.0,
    }

    st.markdown("**Быстрый выбор сценария:**")
    cols = st.columns(4)
    for i, (name, sc) in enumerate(scenarios_fixed.items()):
        if cols[i].button(name):
            scale = sc

    st.divider()

    # считаем VaR для выбранного сценария
    pd_stressed = np.clip(df_port['pd_pred'].values * scale, 0.001, 0.999)
    lgds        = np.full(len(df_port), 0.697)
    eads        = df_port['ead'].values

    with st.spinner("Считаем Monte Carlo VaR..."):
        el, var, es, losses = run_montecarlo(pd_stressed, lgds, eads)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Средний PD",  f"{pd_stressed.mean():.3f}",
              delta=f"{(pd_stressed.mean() - base_pd):+.3f}")
    c2.metric("EL",          f"${el/1e9:.3f}B")
    c3.metric("VaR 99%",     f"${var/1e9:.3f}B")
    c4.metric("VaR / EAD",   f"{var/eads.sum():.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Распределение потерь**")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Histogram(
            x=losses, nbinsx=80,
            marker_color='steelblue', opacity=0.7,
            name='Потери'
        ))
        fig_loss.add_vline(x=el,  line_color='green',
                            line_dash='dash',
                            annotation_text=f'EL={el/1e9:.2f}B')
        fig_loss.add_vline(x=var, line_color='orange',
                            line_dash='dash',
                            annotation_text=f'VaR={var/1e9:.2f}B')
        fig_loss.add_vline(x=es,  line_color='red',
                            line_dash='dash',
                            annotation_text=f'ES={es/1e9:.2f}B')
        fig_loss.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        st.markdown("**Сравнение сценариев**")
        results = []
        for name, sc in scenarios_fixed.items():
            pd_s = np.clip(df_port['pd_pred'].values * sc, 0.001, 0.999)
            el_s, var_s, es_s, _ = run_montecarlo(pd_s, lgds, eads)
            results.append({
                'Сценарий': name,
                'PD':       round(pd_s.mean(), 3),
                'EL (B)':   round(el_s/1e9, 3),
                'VaR (B)':  round(var_s/1e9, 3),
                'ES (B)':   round(es_s/1e9, 3),
            })
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        fig_sc = px.bar(
            df_res, x='Сценарий', y='VaR (B)',
            color='Сценарий',
            labels={'VaR (B)': 'VaR 99% (млрд)'}
        )
        fig_sc.update_layout(height=250, margin=dict(t=20),
                              showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)
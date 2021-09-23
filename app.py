'''Account book dashboard.
'''

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from SessionState import get
from DataMerger import DataMerger


def main():
    # The sidebar menu of Streamlit.

    st.sidebar.markdown('# Menu')
    session_state = get(button_id="", color_to_label={})
    PAGES = {
        'Table Uploader': new_table,
        'Account Book Dashboard': dashboard,
        'Link to Google Sheet': sheet,
        'Re-train model': train,
    }
    page = st.sidebar.selectbox(
        'Select page:', options=list(PAGES.keys()))
    PAGES[page](session_state)


def new_table(session_state):
    # The table uploader.

    col1, col2 = st.columns(2)

    with col1:
        account = st.selectbox(
            'Choose bank',
            ('prestia', 'sony', 'skandia', 'swedbank', 'nordea'))

        user = st.selectbox(
            'Choose user name',
            ('susumu', 'hiromi'))

        file = st.file_uploader('Choose the file to be merged', type=[
                                'csv', 'xlsx', 'xls'])

    if col1.button('Upload'):
        dm = DataMerger()
        dm.update(account, file, user)
        df = dm.get_master()
        col2.dataframe(df)

# @st.cache


def load_df():
    dm = DataMerger()
    df = dm.get_master()
    df['Income'] = df[df['Amount'] > 0]['Amount']
    df['Expense'] = df[df['Amount'] < 0]['Amount'] * (-1)
    return df


def dashboard(session_state):
    # The dashboard.

    df = load_df()
    st.markdown('# Monthly view')

    monthly_columns = st.columns(2)

    month = monthly_columns[0].selectbox(
        label='Month',
        options=df['Transaction date'].dt.strftime('%Y-%m').unique()[::-1],
    )
    filt_month = (df['Transaction date'].dt.strftime('%Y-%m') == month)

    monthly_user = monthly_columns[1].multiselect(
        label='Income from specific user',
        options=df[filt_month]['User'].unique(),
        default=df[filt_month]['User'].unique(),
    )

    filt_month_transfer = df['Category'] != 'transfer'
    filtered_df = df[filt_month & filt_month_transfer]
    filt_month_user = np.array(
        [True if x in monthly_user else False for x in filtered_df['User']])
    expense_df = filtered_df.dropna(subset=['Expense'])
    income_df = filtered_df[filt_month_user].dropna(subset=['Income'])
    filtered_df = pd.concat([expense_df, income_df])

    categories = filtered_df['Category'].unique()

    chart_columns = st.columns([1, 1, 2])

    expense_pie = px.pie(expense_df, values='Expense',
                         names='Category', title='Breakdown of expense')
    income_pie = px.pie(income_df, values='Income',
                        names='Category', title='Breakdown of income')

    income_mask = [True if 'income' in x else False for x in categories]
    expense_mask = [
        False if 'income' in x else False if 'transfer' in x else True for x in categories]
    num_income = sum(income_mask)
    num_expense = sum(expense_mask)

    agg_df = filtered_df.groupby('Category').sum()

    data = agg_df.loc[[cat for cat, mask in zip(categories, income_mask) if mask]]['Amount'].values.tolist() + [agg_df['Income'].sum()] \
        + agg_df.loc[[cat for cat, mask in zip(categories, expense_mask) if mask]
                     ]['Amount'].values.tolist() + [agg_df['Amount'].sum()]

    waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"]*num_income + ["total"] +
        ["relative"]*num_expense + ["total"],
        x=[cat for cat, mask in zip(categories, income_mask) if mask] + ["Income"] + [
            cat for cat, mask in zip(categories, expense_mask) if mask] + ["Total"],
        textposition="outside",
        text=[str(round(x, 1)) for x in data],
        y=data,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    waterfall.update_layout(
        title=f'Overview of {month}',
        showlegend=False
    )

    chart_columns[0].plotly_chart(expense_pie, use_container_width=True)
    chart_columns[1].plotly_chart(income_pie, use_container_width=True)
    chart_columns[2].plotly_chart(waterfall, use_container_width=True)

    for cat in filtered_df['Category'].unique():
        st.markdown(f'### Category: {cat}')
        st.dataframe(filtered_df[filtered_df['Category'] == cat].iloc[:, :-3])

    st.markdown('----')
    st.markdown('# History view')

    input_columns = st.columns(4)

    radio_transfer = input_columns[0].radio(
        'Include or exclude "transfer"',
        ('Exclude', 'Include'),
    )

    radio_moving = input_columns[1].radio(
        'Include only after moving to Sweden',
        ('After moving', 'All'),
    )
    if radio_moving == 'After moving':
        filt_time = (df['Transaction date'] >= '2020-08-01')
    else:
        filt_time = (df['Transaction date'] >= df['Transaction date'].min())

    accounts = input_columns[2].multiselect(
        label='Bank account',
        options=df['Account'].unique(),
        default=df['Account'].unique(),
    )
    filt_account = np.array(
        [True if x in accounts else False for x in df['Account']])

    users = input_columns[3].multiselect(
        label='User',
        options=df['User'].unique(),
        default=df['User'].unique(),
    )
    filt_user = np.array([True if x in users else False for x in df['User']])

    if radio_transfer == 'Include':
        history_df = df[filt_time & filt_account & filt_user].set_index(
            'Transaction date').groupby('Category').resample('1m').sum().reset_index()
    else:
        filt_transfer = df['Category'] != 'transfer'
        history_df = df[filt_time & filt_account & filt_user & filt_transfer].set_index(
            'Transaction date').groupby('Category').resample('1m').sum().reset_index()

    history_df['Transaction date'] = history_df['Transaction date'].dt.strftime(
        '%Y-%m')
    history_bar = px.bar(
        history_df,
        x='Transaction date', y='Amount',
        color='Category',
        title='Transaction history')

    total_df = history_df.groupby('Transaction date')[
        'Amount'].sum().reset_index()
    history_bar.add_trace(
        go.Scatter(x=total_df['Transaction date'], y=total_df['Amount'], mode='lines+markers', name='Total'))

    history_bar.update_xaxes(nticks=len(total_df['Transaction date'])+1)
    st.plotly_chart(history_bar, use_container_width=True)


def sheet(session_state):
    # The link to the sheet.

    st.sidebar.markdown(
        f'# [Go to Google Sheet]({st.secrets["private_gsheets_url"]})')


def train(session_state):
    # Re-train the model.

    if st.sidebar.button('Train'):
        dm = DataMerger()
        dm.train()


def authentication():
    # Simple password authentication.
    session_state = get(password='')

    if session_state.password != st.secrets["password"]:
        pwd_placeholder = st.sidebar.empty()
        pwd = pwd_placeholder.text_input(
            "Password:", value="", type="password")
        session_state.password = pwd
        if session_state.password == st.secrets["password"]:
            pwd_placeholder.empty()
            st.success('Logged in')
            main()
        elif session_state.password != '':
            st.error("Incorrect password")
        else:
            st.info("Please enter the password")

    else:
        main()


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    authentication()
    # main()

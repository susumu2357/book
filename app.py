'''Doc
'''

import os
import streamlit as st
import pandas as pd
import plotly.express as px

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
            ('prestia', 'skandia', 'sony'))

        user = st.selectbox(
            'Choose user name',
            ('susumu', 'hiromi'))

        file = st.file_uploader('Choose the file to be merged', type=['csv', 'xlsx', 'xls'])

    if col1.button('Upload'):
        dm = DataMerger()
        dm.update(account, file, user)
        df = dm.get_master()
        col2.dataframe(df)

@st.cache
def load_df():
    dm = DataMerger()
    df = dm.get_master()
    df['Income'] = df[df['Amount'] > 0]['Amount']
    df['Expense'] = df[df['Amount'] < 0]['Amount'] * (-1)
    return df

def dashboard(session_state):
    # The dashboard.

    df = load_df()

    col1, col2 = st.columns(2)

    with col1:
        date_range = st.slider(
            label='Select the range of transaction date',
            min_value=df['Transaction date'].min().date(),
            max_value=df['Transaction date'].max().date(),
            value=(df['Transaction date'].min().date(), df['Transaction date'].max().date()),
        )

        accounts = st.multiselect(
            label='Select the bank account',
            options=df['Account'].unique(),
            default=df['Account'].unique(),
        )

        users = st.multiselect(
            label='Select the user',
            options=df['User'].unique(),
            default=df['User'].unique(),
        )

        filt_date = (df['Transaction date'].dt.date >= date_range[0]) & (df['Transaction date'].dt.date <= date_range[1]) 
        filt_account = [True if x in accounts else False for x in df['Account']]
        filt_user = [True if x in users else False for x in df['User']]

        st.dataframe(df[filt_date & filt_account & filt_user])

    with col2:
        expense_df = df[filt_date & filt_account & filt_user].dropna(subset=['Expense'])

        fig_pie = px.pie(expense_df, values='Expense', names='Description', title='Breakdown of expense')
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_line = px.bar(expense_df.set_index('Transaction date').resample('1m').sum().reset_index(), x='Transaction date', y='Expense')
        st.plotly_chart(fig_line, use_container_width=True)


def sheet(session_state):
    # The link to the sheet.

    st.sidebar.markdown(f'# [Go to Google Sheet]({st.secrets["private_gsheets_url"]})')


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    main()

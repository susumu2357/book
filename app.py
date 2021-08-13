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
            ('prestia', 'sony', 'skandia', 'swedbank', 'nordea'))

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
        date_upper = df['Transaction date'].max().date()
        
        date_lower = st.date_input(
            label='Select the lower limit of the transaction date',
            min_value=df['Transaction date'].min().date(),
            max_value=date_upper,
            value=df['Transaction date'].min().date(),
        )

        date_upper = st.date_input(
            label='Select the upper limit of the transaction date',
            min_value=date_lower,
            max_value=df['Transaction date'].max().date(),
            value=df['Transaction date'].max().date(),
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

        categories = st.multiselect(
            label='Select the category',
            options=df[df['Category'] != '']['Category'].unique(),
            default=df[df['Category'] != '']['Category'].unique(),
        )

        filt_date = (df['Transaction date'].dt.date >= date_lower) & (df['Transaction date'].dt.date <= date_upper) 
        filt_account = [True if x in accounts else False for x in df['Account']]
        filt_user = [True if x in users else False for x in df['User']]
        filt_category = [True if x in categories else False for x in df['Category']]

        st.dataframe(df[filt_date & filt_account & filt_user & filt_category])

    with col2:
        expense_df = df[filt_date & filt_account & filt_user & filt_category].dropna(subset=['Expense'])
        income_df = df[filt_date & filt_account & filt_user & filt_category].dropna(subset=['Income'])

        expense_pie = px.pie(expense_df, values='Expense', names='Category', title='Breakdown of expense')
        st.plotly_chart(expense_pie, use_container_width=True)

        income_pie = px.pie(income_df, values='Income', names='Category', title='Breakdown of income')
        st.plotly_chart(income_pie, use_container_width=True)

        fig_line = px.bar(df[filt_date & filt_account & filt_user & filt_category].set_index('Transaction date').resample('1m').sum().reset_index(), x='Transaction date', y='Amount')
        st.plotly_chart(fig_line, use_container_width=True)


def sheet(session_state):
    # The link to the sheet.

    st.sidebar.markdown(f'# [Go to Google Sheet]({st.secrets["private_gsheets_url"]})')

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

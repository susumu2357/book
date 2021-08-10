'''Doc
'''

import os
import streamlit as st
import pandas as pd
import subprocess
from PIL import Image
from os import listdir
from google.oauth2 import service_account
from gspread_pandas import Spread, Client

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

    account = st.selectbox(
        'Choose bank',
        ('prestia', 'skandia', 'sony'))

    user = st.selectbox(
        'Choose user name',
        ('susumu', 'hiromi'))

    file = st.file_uploader('Choose the file to be merged:', type=['csv', 'xlsx', 'xls'])

    if st.button('Update'):
        dm = DataMerger()
        dm.update(account, file, user)
        df = dm.get_master()
        st.dataframe(df)


def dashboard(session_state):
    # The dashboard.

    dm = DataMerger()
    df = dm.get_master()
    st.dataframe(df)


def sheet(session_state):
    # The link to the sheet.

    st.sidebar.markdown(f'# [Go to Google Sheet]({st.secrets["private_gsheets_url"]})')


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    main()

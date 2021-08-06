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

    # with st.sidebar:
    #     st.markdown("---")
    #     st.markdown(
    #         '[How to use this application](https://gitlab.com/savanticab/projects/ncc/ama-scan-prototype/-/wikis/How-to-use-the-Streamlit-dashboard)')
    #     st.markdown("---")
        # st.image('/app/ama-scan-prototype/__data/assets/logo.png')
        # st.markdown(
        #     '<div class="left"><img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/NCC_%28Unternehmen%29_logo.svg" width="250"></div>',
        #     unsafe_allow_html=True,
        # )


def new_table(session_state):
    # The table uploader.

    file = st.sidebar.file_uploader('New table:', type=['csv', 'xlsx', 'xls'])

    if file:
        if 'csv' in file.name:
            df = pd.read_csv(file, encoding='Shift-JIS', header=None)
        elif 'xls' in file.name:
            df = pd.read_excel(file)
        else:
            st.sidebar.markdown('File Type Error')
        st.dataframe(df)



def dashboard(session_state):
    # The dashboard.

    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )

    client = Client(scope=scopes, creds=credentials)
    spread = Spread(st.secrets['private_gsheets_url'], client=client)

    test_df = spread.sheet_to_df(header_rows=None, index=None)

    st.markdown('# Account Book')
    st.dataframe(test_df)



def sheet(session_state):
    # The link to Kibana.

    st.sidebar.markdown(f'# [Go to Google Sheet]({st.secrets["private_gsheets_url"]})')


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    main()

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from gspread_pandas import Spread, Client

class DataMerger():
    def __init__(self):
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ]

        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes,
        )
        
        client = Client(scope=scopes, creds=credentials)
        self.spread = Spread(st.secrets["private_gsheets_url"], client=client)
        self.columns = ['Transaction date', 'Description', 'Amount', 'Booked balance', 'Account', 'User']
        
    def cast(self, df):
        df[self.columns[0]] = pd.to_datetime(df[self.columns[0]], format='%Y-%m-%d').astype(str)
        df[self.columns[1]] = df[self.columns[1]].astype(str)
        df[self.columns[2:4]] = df[self.columns[2:4]].astype(float)
        df[self.columns[4:]] = df[self.columns[4:]].astype(str)
        return df
    
    def merge(self, tmp_df):
        master_df = self.spread.sheet_to_df(header_rows=1, index=None)
        master_df = self.cast(master_df)
        merged_df = pd.concat([master_df, tmp_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates()
        merged_df = merged_df.sort_values(by=self.columns[0])
        self.spread.df_to_sheet(merged_df, index=False)
        
    def swedbank(self, path, name):
        tmp_df = pd.read_excel(path, header=7)
        tmp_df = tmp_df[self.columns[:4]]
        tmp_df[self.columns[4]] = 'Swedbank'
        tmp_df[self.columns[5]] = name
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)

    def prestia(self, path, name):
        tmp_df = pd.read_csv(path, header=None, encoding='Shift-JIS')
        tmp_df = tmp_df.iloc[:,[0,2]]
        tmp_df.columns = [col for col in self.columns if col == 'Transaction date' or col == 'Amount']
        tmp_df[self.columns[0]] = tmp_df[self.columns[0]].str.replace('/', '-')
        tmp_df[self.columns[2]] = tmp_df[self.columns[2]].str.replace('SEK', '')
        tmp_df[self.columns[2]] = tmp_df[self.columns[2]].str.replace(',', '')
        tmp_df[self.columns[1]] = 'Not available'
        tmp_df[self.columns[3]] = 0
        tmp_df[self.columns[4]] = 'Prestia'
        tmp_df[self.columns[5]] = name
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)
        
    def skandia(self, path, name):
        tmp_df = pd.read_excel(path, header=3)
        tmp_df.columns = self.columns[:4]
        tmp_df[self.columns[4]] = 'Skandia'
        tmp_df[self.columns[5]] = name
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)
        
    def sony(self, path, name):
        tmp_df = pd.read_excel(path, header=0)
        tmp_df[['お預け入れ額', 'お引き出し額']] = tmp_df[['お預け入れ額', 'お引き出し額']].astype(float)
        tmp_df[['お預け入れ額', 'お引き出し額']] = tmp_df[['お預け入れ額', 'お引き出し額']].fillna(0)
        tmp_df = tmp_df.rename(columns={k:v for k,v in zip(['お取り引き日', '摘要', 'お預け入れ額', '差し引き残高'], self.columns[:4])})
        tmp_df[self.columns[0]] = pd.to_datetime(tmp_df[self.columns[0]], unit='D', origin='1899-12-30')
        tmp_df[self.columns[2]] = tmp_df[self.columns[2]] - tmp_df['お引き出し額']
        tmp_df[self.columns[4]] = 'Sony'
        tmp_df[self.columns[5]] = name
        tmp_df = tmp_df[self.columns]
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)

    def update(self, account, path, name):
        if account == 'prestia':
            self.prestia(path, name)
        elif account == 'skandia':
            self.skandia(path, name)
        elif account == 'sony':
            self.sony(path, name)
        else:
            raise ValueError()

    def get_master(self):
        master_df = self.spread.sheet_to_df(header_rows=1, index=None)
        return master_df

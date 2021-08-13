import glob
from datetime import datetime, timezone
import pytz
from tzlocal import get_localzone
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from gspread_pandas import Spread, Client

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
from joblib import dump, load


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
        self.spread = Spread(st.secrets["private_gsheets_url"], client=client, sheet='book')
        self.columns = ['Transaction date', 'Description', 'Amount', 'Booked balance', 'Account', 'User', 'Category']
        
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
        merged_df = merged_df.drop_duplicates(subset=self.columns[:6])
        merged_df = merged_df.sort_values(by=self.columns[0])
        self.spread.df_to_sheet(merged_df, index=False)
        
    def swedbank(self, path, name):
        tmp_df = pd.read_excel(path, header=7)
        tmp_df = tmp_df[self.columns[:4]]
        tmp_df[self.columns[4]] = 'Swedbank'
        tmp_df[self.columns[5]] = name
        tmp_df[self.columns[6]] = self.pred_category(tmp_df[[x for x in self.columns[1:6] if x != 'Booked balance']])
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
        tmp_df[self.columns[6]] = self.pred_category(tmp_df[[x for x in self.columns[1:6] if x != 'Booked balance']])
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)
        
    def skandia(self, path, name):
        tmp_df = pd.read_excel(path, header=3)
        tmp_df.columns = self.columns[:4]
        tmp_df[self.columns[4]] = 'Skandia'
        tmp_df[self.columns[5]] = name
        tmp_df[self.columns[6]] = self.pred_category(tmp_df[[x for x in self.columns[1:6] if x != 'Booked balance']])
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)
        
    def sony(self, path, name):
        # tmp_df = pd.read_excel(path, header=0)
        tmp_df = pd.read_csv(path, header=0, encoding='Shift-JIS', dtype={'お取り引き日': str, '摘要': str, 'お預け入れ額': str, 'お引き出し額': str, '差引残高': str},
        na_filter=False,
        )
        tmp_df[['お預け入れ額', 'お引き出し額', '差引残高']] = tmp_df[['お預け入れ額', 'お引き出し額', '差引残高']].replace('', '0')
        tmp_df[['お預け入れ額', 'お引き出し額', '差引残高']] = tmp_df[['お預け入れ額', 'お引き出し額', '差引残高']].applymap(lambda x: str(x).replace(',', ''))
        tmp_df[['お預け入れ額', 'お引き出し額', '差引残高']] = tmp_df[['お預け入れ額', 'お引き出し額', '差引残高']].astype(float)
        tmp_df = tmp_df.rename(columns={k:v for k,v in zip(['お取り引き日', '摘要', 'お預け入れ額', '差引残高'], self.columns[:4])})
        tmp_df[self.columns[0]] = pd.to_datetime(tmp_df[self.columns[0]], format='%Y年%m月%d日')
        tmp_df[self.columns[2]] = tmp_df[self.columns[2]] - tmp_df['お引き出し額']
        tmp_df[self.columns[4]] = 'Sony'
        tmp_df[self.columns[5]] = name
        tmp_df = tmp_df[self.columns[:6]]
        tmp_df[self.columns[6]] = self.pred_category(tmp_df[[x for x in self.columns[1:6] if x != 'Booked balance']])
        # tmp_df[self.columns[6]] = ''
        print(tmp_df.shape)
        tmp_df = self.cast(tmp_df)
        self.merge(tmp_df)
        
    def backup(self):
        master_df = self.get_master()
        master_df = self.cast(master_df)
        name = f'backup_{self.CET_today().strftime("%Y-%m-%d-%H:%M:%S")}'
        self.spread.df_to_sheet(master_df, index=False, sheet=name)
    
    def update(self, account, path, name):
        self.backup()
        
        if account == 'prestia':
            self.prestia(path, name)
            return
        elif account == 'skandia':
            self.skandia(path, name)
            return
        elif account == 'sony':
            self.sony(path, name)
            return
        elif account == 'swedbank':
            self.swedbank(path, name)
            return
        else:
            raise ValueError()

    def get_master(self):
        master_df = self.spread.sheet_to_df(header_rows=1, index=None, sheet='book')
        master_df = self.cast(master_df)
        master_df[self.columns[0]] = pd.to_datetime(master_df[self.columns[0]], format='%Y-%m-%d')
        return master_df

    def train(self, test_size=0.25, random_state=42):
        train_df = self.get_master()
        filt = train_df[self.columns[6]] != ''
        train_df = train_df[filt]
        X_train, X_test, y_train, y_test = train_test_split(train_df[[x for x in self.columns[1:6] if x != 'Booked balance']], train_df[self.columns[6]], test_size=test_size, random_state=random_state)
        
        preprocessing = ColumnTransformer([
            ('numeric', StandardScaler(), [self.columns[2]]),
            ('categorical', OneHotEncoder(), self.columns[4:6]),
            ('text', TfidfVectorizer(), self.columns[1]),
        ])
        
        model = Pipeline([
            ('preprocess', preprocessing),
            ('classifier', LGBMClassifier(random_state=random_state))
        ])
        
        model.fit(X_train, y_train)
        dump(model, f'./model/model_{self.CET_today().strftime("%Y-%m-%d-%H:%M:%S")}.joblib') 
        
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
    def pred_category(self, pred_df, model_id=-1):
        names = glob.glob('./model/model_*')
        timestamps = pd.to_datetime([x.replace('./model/model_', '').replace('.joblib', '') for x in names], format='%Y-%m-%d-%H:%M:%S').sort_values()
        model = load(f'./model/model_{timestamps[model_id].strftime("%Y-%m-%d-%H:%M:%S")}.joblib')
        
        return model.predict(pred_df)

    def CET_today(self):
        utc_dt = datetime.now(timezone.utc)
        CET = pytz.timezone('CET')
        return utc_dt.astimezone(CET)
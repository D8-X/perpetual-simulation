#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
from sqlalchemy import create_engine
import pandas as pd

file_src = "results/res180-180_2021-11-1.csv"
df = pd.read_csv(file_src)

# pickle
#df.to_pickle('results/res.pkl')

# postgres
engine = create_engine('postgresql://postgres:postgres@localhost:5432/amm_sim')
df.to_sql('results', engine)


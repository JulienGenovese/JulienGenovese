import sys

import pandas as pd

sys.append("./")

# todo: parte in cui richiama le classi che gli servono
from rule_executor import RuleExectuor
# todo: parte in cui scarica i dati
df_transaction = pd.read_csv("df_transaction.csv")
# todo: parte in cui lancia l'esecutore sui dati


parameters = ["......"]
re = RuleExectuor(parameters)
df_final_result = re.run_executor(df_transaction)
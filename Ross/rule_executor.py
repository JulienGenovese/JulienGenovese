import pandas as pd


class RuleExectuor:
    def __init__(self, parameters):
        self.parameters = parameters

    @classmethod
    def list_of_rules(cls):
        return [
            rule_1,
            rule_2
        ]
    def instantiate_rules(self):
        list_of_rules = []
        for rule in list_of_rules:
            list_of_rules.append(rule(self.parameters))

        return list_of_rules

    def run_executor(self, df_transactions):
        list_of_rules = self.instantiate_rules()
        result_df_list = []
        for rule in list_of_rules:
            result_df = rule.run(df_transactions)
            result_df_list.append(result_df)
        final_result_df = pd.concat(result_df_list)
        return final_result_df

from src.model.classifier import Classifier

class Evaluator:

    def __init__(self, test_table, target_column, conditional_dict):
        self._test_table = test_table
        self._target_column = target_column
        self._conditional_dict = conditional_dict
        self._success_counter = 0


    def testing(self):
        for row in self._test_table.iterrows():
            cf = Classifier(row[1].to_dict(), self._target_column, self._conditional_dict).classifier()
            if cf[1] == row[1][self._target_column]:
                self._success_counter += 1
        return (self._success_counter / len(self._test_table)) * 100


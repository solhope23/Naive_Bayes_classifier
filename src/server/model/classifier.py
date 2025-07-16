class Classifier:

    def __init__(self, question_dict, target_column, conditional_dict):
        self._question_dict = question_dict
        self._target_column = target_column
        self._conditional_dict = conditional_dict
        self._answer = {}


    def classifier(self):
        used = False
        for target_value, features in self._conditional_dict.items():
            self._answer[target_value] = 1
            for feature, feature_values in features.items():
                if (feature in self._question_dict) and (feature != self._target_column) :
                    if self._question_dict[feature] in feature_values:
                        self._answer[target_value] *= feature_values[self._question_dict[feature]]
                        used = True
                    # else:
                    #     print(f'A value was inserted that does not exist in the model.')
        if used:
            return self._answer, max(self._answer, key=lambda k: self._answer[k])














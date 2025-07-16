from collections import defaultdict

class Trainer:

    def train(self, table,  target_column):

    def

    def _laplace_smoothing(self):








    def train(self, target_column):
        target_values = self._table[target_column].unique()
        columns = [col for col in self._table.columns if target_column != col]
        for target_value in target_values:
            new_table = self._table[self._table[target_column] == target_value]
            self.value_counts_dict[target_value] = {}
            for column in columns:
                self.value_counts_dict[target_value][column] = new_table[column].value_counts().to_dict()




    # def train(self):
    #     target_values = self._table[self._target_column].unique()
    #     columns = [col for col in self._table.columns if col != self._target_column]
    #
    #     for target_value in target_values:
    #         new_table = self._table[self._table[self._target_column] == target_value]
    #         self._update_value_counts(target_value, new_table, columns)
    #
    # def _update_value_counts(self, value, target_value, columns):
    #     for column in columns:
    #         counts = target_value[column].value_counts().to_dict()
    #         for val, count in counts.items():
    #             self.value_counts_dict[value][column][val] += count


    # def _1_target_values_to_dict(self, target_column):
    #     target_values = self.table[target_column].unique()
    #     for target_value in target_values:
    #         self.conditional_table[target_value] = {}
    #
    # def _2_columns_to_target_values_in_dict(self, target_column):
    #     columns = [col for col in self.table.columns if col != target_column]
    #     for column in columns:
    #         for target_value in self.conditional_table:
    #             self.conditional_table[target_value][column] = {}
    #
    # def _3_laplace_smoothing_to_columns_in_dict(self, target_column):
    #     for target_value in self.conditional_table:
    #         for column in self.conditional_table[target_value]:
    #             column_values = self.table[column].unique()
    #             for column_value in column_values:
    #                 self.conditional_table[target_value][column][column_value] = 1
    #
    # def _4_




    # def reset(self):
    #     self.__init__()

    @staticmethod
    def _ensure_linked_args_provided(frequency_dict, model_schema):
        if (frequency_dict is None) != (model_schema is None):
            raise ValueError("Both 'frequency_dict' and 'model_schema' must be provided together or not at all.")

    @staticmethod
    def _update_frequency_default_dict(frequency_dict):
        if frequency_dict is None:
            return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        return defaultdict(lambda: defaultdict(lambda: defaultdict(int)),frequency_dict)

    @staticmethod
    def _update_model_schema_default_dict(model_schema):
        if model_schema is None:
            return defaultdict(set)
        return defaultdict(set, model_schema)

    def update_training_data(self, table, target_column):
        self._table = table
        self._target_column = target_column
        self._ready_for_training = True

    def train(self):
        if self._ready_for_training:
            self._build_frequency_table()
            self._update_model_schema()
            self._conditional_dict = copy.deepcopy(self._frequency_dict)
            self._laplace_smoothing()
            self._compute_conditional_probabilities()
            self._ready_for_training = False
            return Trainer.default_dict_to_dict(self._conditional_dict), Trainer.default_dict_to_dict(self._model_schema)
        else:
            raise RuntimeError("Training failed: You must first set a new training table before calling Train().")

    @staticmethod
    def default_dict_to_dict(default_dict):
        if isinstance(default_dict, defaultdict):
            result = {}
            for k, v in default_dict.items():
                result[k] = Trainer.default_dict_to_dict(v)
            return result
        return default_dict

    def _build_frequency_table(self):
        target_values = self._table[self._target_column].unique()
        columns = self._get_feature_columns()
        self._process_all_target_values(target_values, columns)

    def _process_all_target_values(self, target_values, columns):
        for target_value in target_values:
            filtered_table = self._table[self._table[self._target_column] == target_value]
            self._update_value_counts(target_value, filtered_table, columns)

    def _update_value_counts(self, target_value, table, columns):
        for column in columns:
            counts = table[column].value_counts().to_dict()
            self._add_counts_to_dict(target_value, column, counts)

    def _add_counts_to_dict(self, target_value, column, counts_dict):
        for val, count in counts_dict.items():
            self._frequency_dict[target_value][column][val] += count

    def _get_feature_columns(self):
        return [col for col in self._table.columns if col != self._target_column]

    def _update_model_schema(self):
        for column in self._get_feature_columns():
            for value in self._table[column].unique():
                self._model_schema[column].add(value)

    def _laplace_smoothing(self):
        for target_value in self._conditional_dict:
            for column in self._model_schema:
                for value in self._model_schema[column]:
                    self._conditional_dict[target_value][column][value] += 1

    def _compute_conditional_probabilities(self):
        for target_value in self._conditional_dict:
            for column in self._conditional_dict[target_value]:
                total = sum(self._conditional_dict[target_value][column].values())
                self._normalize_column_counts(target_value, column, total)

    def _normalize_column_counts(self, target_value, column, total):
        for value in self._conditional_dict[target_value][column]:
            count = self._conditional_dict[target_value][column][value]
            self._conditional_dict[target_value][column][value] = count / total
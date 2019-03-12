from copy import deepcopy
from random import uniform


class Schema:
    def __init__(self):
        self.col_seq = []
        self.tab_seq = []
        self.par_tnum = []
        self.foreign_keys = []

    def import_from_data(self, data, idx, table_type):
        ts = data[idx]["ts"]
        tname_toks = [x.split(" ") for x in ts[0]]
        col_type = ts[2]
        cols = [x.split(" ") for xid, x in ts[1]]
        tab_seq = [xid for xid, x in ts[1]]
        cols_add = []
        for tid, col, ct in zip(tab_seq, cols, col_type):
            col_one = [ct]
            if tid == -1:
                tabn = ["all"]
            else:
                if table_type == "no":
                    tabn = []
                elif table_type == "struct":
                    tabn = []
                else:
                    tabn = tname_toks[tid]
            for t in tabn:
                if t not in col:
                    col_one.append(t)
            col_one.extend(col)
            cols_add.append(col_one)
        self.col_seq = cols_add
        self.tab_seq = tname_toks
        self.par_tnum = tab_seq
        self.foreign_keys = deepcopy(ts[3])
        return ts[4]  # db name

    def reduce_table(self, reduce_portion, reduce_first_column, maintain_columns, maintain_tables):
        new_tab_num = []
        new_col_num = []

        new_col_seq = []
        new_tab_seq = []
        new_par_tnum = []
        new_foreign_keys = []
        for tab_num, tab in enumerate(self.tab_seq):
            reduce = False
            if uniform(0., 1.) < reduce_portion:
                reduce = True
            if tab_num in maintain_tables:
                reduce = False
            if not reduce:
                new_tab_num.append(tab)

        for idx, tab_num in enumerate(self.par_tnum):
            new_num = 0
            for living_tab in new_tab_num:
                if living_tab < tab_num:
                    new_num += 1
            self.par_tnum[idx] = new_num

        for col_num, col in enumerate(self.col_seq):
            reduce = False
            if reduce_first_column:
                reduce = True
            if uniform(0., 1.) < reduce_portion:
                reduce = True
            if self.par_tnum[col_num] not in new_tab_num:
                reduce = True
            if col_num in maintain_columns:
                reduce = False

            if not reduce:
                new_col_num.append(col)
        for col_num in new_col_num:
            new_par_tnum.append(self.par_tnum[col_num])
        for idx, (f, p) in enumerate(self.foreign_keys):
            new_f_num = 0
            for living_col in new_col_num:
                if living_col < new_f_num:
                    new_f_num += 1
            if f not in new_col_num:
                new_f_num = -1
            new_p_num = 0
            for living_col in new_col_num:
                if living_col < new_p_num:
                    new_p_num += 1
            if f not in new_col_num:
                new_p_num = -1
            self.foreign_keys[idx] = [new_f_num, new_p_num]
        for f, p in self.foreign_keys:
            if f != -1 and p != -1:
                new_foreign_keys.append([f, p])
        for tab in new_tab_num:
            new_tab_seq.append(self.tab_seq[tab])
        for col in new_col_num:
            new_col_seq.append(self.col_seq[col])
        self.col_seq = new_col_seq
        self.tab_seq = new_tab_seq
        self.par_tnum = new_par_tnum
        self.foreign_keys = new_foreign_keys

    def append_table(self, schema):
        pass

    @staticmethod
    def dict_from_data(data, table_type, use_mono_schema):
        schema_dict = {}
        for idx in range(len(data)):
            ts = data[idx]["ts"]
            if len(ts[0]) <= 1 and use_mono_schema:
                continue
            new_schema = Schema()
            db_name = new_schema.import_from_data(data, idx, table_type)
            schema_dict[db_name] = new_schema
        return schema_dict

    @staticmethod
    def make_batch(schema_list):
        col_seq_batch = []
        tab_seq_batch = []
        par_tnum_batch = []
        foreign_key_batch = []
        for schema in schema_list:
            col_seq_batch.append(schema.col_seq)
            tab_seq_batch.append(schema.tab_seq)
            par_tnum_batch.append(schema.par_tnum)
            foreign_key_batch.append(schema.foreign_keys)
        return col_seq_batch, tab_seq_batch, par_tnum_batch, foreign_key_batch

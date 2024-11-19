def get_outer_columns(all_columns, num_columns_each_end=10):
    if len(all_columns) <= num_columns_each_end * 2:
        return list(all_columns)
    return all_columns[:num_columns_each_end] + all_columns[-num_columns_each_end:]

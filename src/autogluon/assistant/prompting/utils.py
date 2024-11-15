def get_outer_columns(all_columns, num_columns_each_end=10):
    return all_columns[:num_columns_each_end] + all_columns[-num_columns_each_end:]

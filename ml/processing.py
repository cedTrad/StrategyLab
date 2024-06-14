import sqlalchemy



def var_to_str(data):
    data.columns = [str(col) if isinstance(col, sqlalchemy.sql.elements.quoted_name) else col for col in data.columns]

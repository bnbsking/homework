import pandas as pd


def get_age(birth: str):
    birth_date = pd.to_datetime(birth)
    current_date = pd.to_datetime("today")
    age = current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
    return age

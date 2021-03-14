from dateutil.relativedelta import relativedelta
import pandas as pd

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay


def get_shifted_day(ths_today_date, ths_shifted_day_dict, ths_operation="past"):
    if len(ths_shifted_day_dict) != 1:
        print("Your time shift dictionary contains more than one entry!")
        raise ValueError
    for ths_key, value in ths_shifted_day_dict.items():
        ths_shift_type = ths_key
        ths_shift_val = value
        if ths_key == "year" or ths_key == "years":
            if ths_operation == "past":
                ths_shifted_day = ths_today_date - relativedelta(
                    years=ths_shift_val)
            if ths_operation == "future":
                ths_shifted_day = ths_today_date + relativedelta(
                    years=ths_shift_val)
        elif ths_key == "day" or ths_key == "days":
            if ths_operation == "past":
                ths_shifted_day = ths_today_date - relativedelta(
                    days=ths_shift_val)
            if ths_operation == "future":
                ths_shifted_day = ths_today_date + relativedelta(
                    days=ths_shift_val)
        elif ths_key == "month" or ths_key == "months":
            if ths_operation == "past":
                ths_shifted_day = ths_today_date - relativedelta(
                    months=ths_shift_val)
            if ths_operation == "future":
                ths_shifted_day = ths_today_date + relativedelta(
                    months=ths_shift_val)
        else:
            raise ValueError("Your time shift type {} isn't implemented! Try 'year', 'month' or 'day'.". \
                             format(ths_shift_type))
        return ths_shifted_day


def convert_date_to_pandas_datetime(this_date):
    try:
        this_date = pd.to_datetime(
            this_date, errors='ignore', format="%Y-%m-%d").date()
        this_date = this_date.isoformat()
        return this_date
    except TypeError:
        return this_date


def t_to_sql_string(t, format='%Y-%m-%d'):
    if not isinstance(t, str):
        try:
            t = t.strftime(format)
        except Exception as e:
            print(e.args[0])
            raise TypeError(
                "Cannot convert t to string! Please provide t as string Y-m-d or datetime!"
            )
    else:
        return t.replace('_', '-')
    return t


def t_to_str_file_name(t, format='%Y_%m_%d'):
    format = format.replace('-', '_')
    if not isinstance(t, str):
        try:
            t = t.strftime(format)
        except Exception as e:
            print(e.args[0])
            raise TypeError(
                "Cannot convert t to string! Please provide t as string Y_m_d or datetime!"
            )
    else:
        return t.replace('-', '_')
    return t


def return_time_delta_list(start, finish):
    """
    Returns the time delta list between t2 and start
    :param start:
    :param finish:
    :return:
    """
    import datetime
    # we want finish be greater than start: finish>start
    if start > finish:
        start = finish
        finish = start
    else:
        start = start
        finish = finish
    ths_time_delta = finish - start
    ths_total_seconds = ths_time_delta.total_seconds()

    ths_hours = divmod(ths_total_seconds, 3600)[0]
    ths_minutes = divmod(ths_total_seconds - ths_hours * 3600, 60)[0]
    ths_seconds = ths_total_seconds - ths_hours * 3600 - ths_minutes * 60

    new_t2 = start + datetime.timedelta(
        hours=ths_hours, minutes=ths_minutes, seconds=ths_seconds)
    status = new_t2 == finish

    if not status:
        raise ValueError(
            "The calculated value of time difference is wrong! Check return_time_delta_list function!"
        )
    else:
        return [ths_hours, ths_minutes, ths_seconds, status]


import datetime as dt




class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]


def get_trading_close_holidays(year):
    inst = USTradingCalendar()

    return inst.holidays(dt.datetime(year-1, 12, 31), dt.datetime(year, 12, 31))
"""
Scripts for various utils
Created in 2023
@author: Nico Röttcher
"""

import pandas as pd
import itertools
import math
import warnings
import numpy as np
import importlib.util
import sys
import zipfile
from pathlib import Path


def check_type(
    name,
    value,
    allowed_None=False,
    allowed_types=None,
    str_int_to_list=False,
):
    """
    check the type of a varibale and raise error if type is not matched.
    :param name: str
        name of the variable, used for the error message
    :param value: Any
        value of the variable to be checked in type
    :param allowed_None: bool
        whether None type is allowed or not
    :param allowed_types: list of types
        list of allwoed types
    :param str_int_to_list: bool
        whether to transform variable of type str or int to a list
    :return: value
    """
    if allowed_types is None:
        allowed_types = []
    if allowed_None:
        if not (value is None or type(value) in allowed_types):
            raise ValueError(
                f"{name} must be None or of type: "
                + ", ".join([str(dtype) for dtype in allowed_types])
            )
    else:
        if not (type(value) in allowed_types):
            raise ValueError(
                f"{name} must be of type: "
                + ", ".join([str(dtype) for dtype in allowed_types])
            )
    if str_int_to_list and type(value) in [str, int]:
        if type(list()) not in allowed_types:
            raise Exception(
                f"{name} is transformed to list although not given in allowed_types."
            )
        value = list([value])
    return value


# pandas related tools
def singleindex_to_multiindex_list(index):
    """
    Takes pandas index or multiindex and transform it to list of tuples and the name of the index columns as list
    :param index: pandas.Index or pandas.MultiIndex
        pandas Index or MultiIndex
    :return: index, names
        index as list of tuples
        names as list
    """
    names = index.names
    if type(index) == pd.MultiIndex:
        index = index.tolist()
    else:
        index = [(val,) for val in index.tolist()]
    return index, names


def multiindex_from_product_indices(index1, index2):
    """
    Creates a pandas.MultiIndex from the product of two indices (index1, index2). These can be both Index or MultiIndex
    :param index1: pandas.Index or pandas.MultiIndex
        pandas Index or MultiIndex
    :param index2: pandas.Index or pandas.MultiIndex
        pandas Index or MultiIndex
    :return: pandas.MultiIndex
        product of both indices
    """
    index1, name1 = singleindex_to_multiindex_list(index1)
    index2, name2 = singleindex_to_multiindex_list(index2)

    print(index1, index2)
    product_index = list(itertools.product(index1, index2))
    print([row for row in product_index])
    # print(product_index[0])
    tuples = [sum(row, ()) for row in product_index]

    index = pd.MultiIndex.from_tuples(tuples, names=name1 + name2)

    return index

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier - 0.5) / multiplier


def count_digits_strip0(value):
    # Convert the number to string and remove any non-digit characters
    return len([digit for digit in str(value).strip('0') if digit.isdigit()])


def count_decimal_digits_strip0(value):
    # Convert the number to string and remove any non-digit characters
    value_str = str(float(value)).strip('0')
    return len([digit for digit in value_str[value_str.find('.'):] if digit.isdigit()])


def count_nondecimal_digits_strip0(value):
    # Convert the number to string and remove any non-digit characters
    value_str = str(float(value)).strip('0')
    print(value_str)
    return len([digit for digit in value_str[:value_str.find('.') + 1] if digit.isdigit()])


def round_digits(value,
                 digits=1,
                 method="half_up",
                 return_type='intfloat',
                 return_str_scientific=False):
    """
    round a float value to specified number of digits (sum of digits before and after comma).
    You can specify method whether to round always up, down, half_up, half_down.
    Please verify as there are some errors in some cases.
    :param value: float
        value to be transformed
    :param digits: int
        number of digits to be returned
    :param method: str one of ['up', 'down', 'half_up', 'half_down']
        up: always round up
        down: always round down
        half_up: round up if first cut digit >=5 and down if <5
        half_down: round up if first cut digit >5 and down if <=5
    :param return_type:  str one of ['intfloat', 'str', 'str_scientific'] default 'list'
        type of return
        - intfloat retrun th number (int or float depending on whether decimal or not)
        - str return string of decimal number
        - str_scientific return string of number in scientific notation
    :param return_str_scientific: bool
        deprecated
        whether to return as scientific notation string or as number (int or float depending on whether decimal or not)
    :return: str or int or float
    """
    methods = {"up": round_up,
               "down": round_down,
               "half_up": round_half_up,
               "half_down": round_half_down
               }

    if method not in methods.keys():
        raise Exception("method must be one of %s" % methods.keys())

    if value == 0 or np.isnan(value):
        round_value = 0
    else:
        decimal_of_first_digit = -(math.floor(math.log10(abs(value))) + 1)  # derive the order of magnitude of the value
        # print(decimal_of_first_digit)
        round_to_decimals = decimal_of_first_digit + digits

        # perform rounding
        round_value = methods[method](n=value, decimals=round_to_decimals)

        # transform to int if only 0 decimals
        if round_value == int(round_value):
            round_value = int(round_value)

    # adjust return type
    if return_str_scientific:
        print("\x1b[33m", "parameter return_str_scientific is deprecated. Please use return_type='str_scientific'", "\x1b[0m")
        return_type = 'str_scientific'
    return_types = ['intfloat', 'str', 'str_scientific']

    if return_type == 'intfloat':
        return round_value
    elif return_type == 'str':
        formatted_decimal_places = digits + decimal_of_first_digit
        formatted_decimal_places = formatted_decimal_places if formatted_decimal_places > 0 else 0
        return_str = f"%.{formatted_decimal_places}f"
        return return_str % round_value
    elif return_type == 'str_scientific':
        return_str = f"%.{digits - 1}E"
        return return_str % round_value
    else:
        raise Exception("return_type must be one of %s" % return_types)


def round_significant_digits(value, error,
                             digits_error=1,
                             method_value='half_up',
                             method_error='up',
                             return_type='list'):
    """
    Round a value to the number of significant digits according to corresponding error.
    Number of digits for the error can be specified. By default error is round_up and value is round half_up.
    Please verify as there are some errors in some cases.
    :param value:
    float
        value to be transformed
    :param error: float
        corresponding error of the value
    :param digits_error: int
        number of digits the error should be round to
    :param method_value: str one of ['up', 'down', 'half_up', 'half_down'] default 'half_up'
        method used to round the value
        up: always round up
        down: always round down
        half_up: round up if first cut digit >=5 and down if <5
        half_down: round up if first cut digit >5 and down if <=5
    :param method_error: str one of ['up', 'down', 'half_up', 'half_down'] default 'up'
        method used to round the error
        up: always round up
        down: always round down
        half_up: round up if first cut digit >=5 and down if <5
        half_down: round up if first cut digit >5 and down if <=5
    :param return_type:  str one of ['list', 'str', 'str_scientific'] default 'list'
        type of return
    :return: list with [value, error] or formatted str
    """
    methods = {"up": round_up,
               "down": round_down,
               "half_up": round_half_up,
               "half_down": round_half_down
               }
    if method_value not in methods.keys():
        raise Exception("method_value must be one of %s" % methods.keys())
    if method_error not in methods.keys():
        raise Exception("method_error must be one of %s" % methods.keys())

    return_types = ['list', 'str', 'str_scientific']
    if return_type not in return_types:
        raise Exception("return_type must be one of %s" % return_types)

    # problem one of the value/error 0
    if error == 0 or value == 0 or np.isnan(error) or np.isnan(value):
        round_value = value
        round_error = error
        if return_type == 'list':
            return [round_value, round_error]
        else:
            warnings.warn('Value or error = 0, cannot handle this.')
            return f"${round_value} \pm {round_error}$"

    error_decimal_of_first_digit = -(math.floor(math.log10(abs(round_digits(value=error,
                                                                            digits=1,
                                                                            method=method_error
                                                                            )
                                                               ))) + 1)
    # round error first to one digit - in case error = 0.09xx --> error = 0.1 (higher order of magnitude)
    # same might need to be applied to value?
    value_decimal_of_first_digit = -(math.floor(math.log10(abs(value))) + 1)

    # Problem if error larger than value
    if error_decimal_of_first_digit < value_decimal_of_first_digit:
        # Error at least one order of magnitude larger than value
        warnings.warn('Error at least one order of magnitude larger than value.')
        round_error = round_digits(value=error, digits=digits_error, method=method_error)
        round_value = round_digits(value=value, digits=1, method=method_value)
        return_type = 'str' if return_type == 'str_scientific' else return_type

        if return_type == 'list':
            return [round_value, round_error]
        else:
            return f"${round_value} \pm {round_error}$"

    # Normal behavious
    # print(error_decimal_of_first_digit)
    round_error = round_digits(value=error, digits=digits_error, method=method_error)
    round_value = methods[method_value](n=value, decimals=error_decimal_of_first_digit + 1)

    # transform to int if only 0 decimals
    if round_value == int(round_value) and round_error == int(round_error):
        round_value = int(round_value)
        round_error = int(round_error)

    # adjust return type
    if return_type == 'list':
        return [round_value, round_error]
    elif return_type[:3] == 'str':

        if return_type == 'str':
            decimal_places = abs(
                error_decimal_of_first_digit + 1) if round_error != 0 else 0  # abs() because this is only affected if error <1
            formatted_value = f"{round_value:.{decimal_places}f}"
            return f"${formatted_value} \pm {round_error}$"

        elif return_type == 'str_scientific':
            decimal_places = (error_decimal_of_first_digit + 1) if round_error != 0 else 0
            # print(value_decimal_of_first_digit, error_decimal_of_first_digit)
            # print(decimal_places-value_decimal_of_first_digit-1)
            formatted_decimal_places = decimal_places - value_decimal_of_first_digit - 1
            if formatted_decimal_places >= 0:
                formatted_value = f"{round_value * 10 ** (value_decimal_of_first_digit + 1):.{formatted_decimal_places}f}"
                formatted_error = f"{round_error * 10 ** (value_decimal_of_first_digit + 1):.{formatted_decimal_places}f}"
                # print('value_decimal_of_first_digit', value_decimal_of_first_digit)
                return "$(%s \pm %s) \cdot 10^{%s}$" % (
                formatted_value, formatted_error, -value_decimal_of_first_digit - 1)
                # return "(%s $\pm$ %s)E-%s" %(formatted_value, formatted_error, value_decimal_of_first_digit)
            else:
                formatted_value = f"{round_value * 10 ** (value_decimal_of_first_digit + 1)}"
                formatted_error = f"{round_error * 10 ** (value_decimal_of_first_digit + 1)}"
                # print('value_decimal_of_first_digit', value_decimal_of_first_digit)
                return "$(%s \pm %s) \cdot 10^{%s}$" % (
                formatted_value, formatted_error, -value_decimal_of_first_digit - 1)

    else:
        return round_value, error


def round_significant_digits_df(df, col_value, col_error, **kwargs):
    """
    round_significant_digits for pd.DataFrame storing a column for value and error
    :param df: pd.DataFrame
    :param col_value: str
        name of the column in which the value is stored
    :param col_error: str
        name of the column in which the error is stored
    :param kwargs: kwargs of round_significant_digits()
    :return: list
    """
    out = []
    for idx, row in df.iterrows():
        out += [round_significant_digits(row.loc[col_value], row.loc[col_error],
                             **kwargs)
               ]
    return out


def round_digits_df(df, col_value, return_type_list='list', **kwargs):
    """
    round_significant_digits for pd.DataFrame storing a column for value and error
    :param df: pd.DataFrame
    :param col_value: str
        name of the column in which the value is stored
    :param return_type_list:  str one of ['list', 'pd.Series', ] default 'list'
        type of return
        - list return rounded values as list
        - pd.Series  return rounded values as pd.Series
    :param kwargs: kwargs of round_significant_digits()
    :return: list or pd.Series
    """
    out = []
    for idx, row in df.iterrows():
        out += [round_digits(row.loc[col_value],
                             **kwargs)
                ]
    if return_type_list == 'pd.Series':
        out = pd.Series(out, index=df.index)
    return out


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def zipfiles(zip_dst, src_files, compresslevel=5):
    """
    Zip files into zip-archive
    :param zip_dst: list of str
        list of files to be included into the archive
    :param src_files: str
        filename of the zip-archive
    :param compresslevel: int 0-9
        see zipfile.ZipFile(compresslevel=)
    :return:
    """

    with zipfile.ZipFile(zip_dst, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as archive:
        for filename in src_files:
            archive.write(filename, arcname=Path(filename).name)


def unzipfiles(zip_src, dst_path='.'):
    """
    Zip files into zip-archive
    :param zip_src: str
        filename of the zip-archive
    :param dst_path: str
        path  to where to extract to, default current location
    :return:
    """
    with zipfile.ZipFile(zip_src, 'r') as zip_ref:
        zip_ref.extractall(path=dst_path)


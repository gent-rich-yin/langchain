import traceback
from datetime import datetime, date
from io import BytesIO
from typing import TypedDict, Literal, List, Optional, Annotated
from zipfile import ZipFile
from xml.etree import ElementTree
import requests
import os

from langchain_core.tools import tool

supported_currencies = ['USD', 'JPY', 'EUR', 'GBP', 'CHF', 'AUD']

class CurvePoint(TypedDict):
    """
    Interest rate for a tenor
    """
    tenor: Annotated[str, 'tenor of the interest rate. (examples: tenor 1M is One month interest rate, tenor 1Y is One year interest rate)']
    maturity_date: Annotated[date, 'date that the tenor expires (matured)']
    rate: Annotated[float, 'interest rate of the tenor']

class IRCurve(TypedDict):
    """
    Interest rate curve holds interest rates on the effective date for the currency
    """
    effective_date: Annotated[date, 'the date that the interest rates were observed']
    currency: Annotated[Literal['USD', 'JPY', 'EUR', 'GBP', 'CHF', 'AUD'], 'the currency of the interest rate']
    curve_points: Annotated[List[CurvePoint], 'interest rates for the tenors (examples: tenor 1M is One month interest rate, tenor 1Y is One year interest rate)']

def _build_ir(xml: str, currency: Literal['USD', 'JPY', 'EUR', 'GBP', 'CHF', 'AUD']) -> IRCurve:
    root = ElementTree.fromstring(xml)
    effective_date_str = root.find('interestRateCurve').find('ois').find('snaptime').text
    effective_date = datetime.strptime(effective_date_str, '%Y-%m-%dT%H:%M:%S').date()
    curve_points = []
    for curve_point in root.find('interestRateCurve').find('ois').findall('curvepoint'):
        tenor = curve_point.find('tenor').text
        maturity_date_str = curve_point.find('maturitydate').text
        maturity_date = datetime.strptime(maturity_date_str, '%Y-%m-%d').date()
        rate = float(curve_point.find('parrate').text)
        curve_points.append(CurvePoint(tenor=tenor, maturity_date=maturity_date, rate=rate))
    return IRCurve(effective_date=effective_date, currency=currency, curve_points=curve_points)

@tool
def load_interest_rate_curve(date_obj: date, currency: Literal['USD', 'JPY', 'EUR', 'GBP', 'CHF', 'AUD']) \
        -> Annotated[Optional[IRCurve], 'Interest rate object in a Python dictionary']:
    """
    Loads interest rate curve for the given date and given currency
    Returns Interest rate object (IRCurve, which is a Python dictionary) if the interest rates can be found
    Returns None if the interest rates cannot be found
    """
    try:
        url = f'https://rfr.ihsmarkit.com/InterestRates_{currency}_{date_obj.strftime('%Y%m%d')}.zip?email={os.environ['ir_curve_email']}'
        req = requests.get(url)
        zip = ZipFile(BytesIO(req.content))
        file = f'InterestRates_{currency}_{date_obj.strftime('%Y%m%d')}.xml'
        lines = zip.open(file).readlines()
        xml = "".join([line.decode('utf-8') for line in lines])
        return _build_ir(xml, currency)
    except Exception as e:
        print(traceback.format_exc())
        return None

# if __name__ == '__main__':
#     print(load_interest_rate_curve.invoke(date(2025, 1, 24), 'USD')))

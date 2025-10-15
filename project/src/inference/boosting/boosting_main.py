# import pandas as pd
# import numpy as np
# import json

import inference as ml
import joblib
import log

import utils


def predict(str_xml):
    log.log("predict() vector: ")
    log.log(str_xml.decode("utf8"))
    log.log_request(str_xml.decode("utf8"))

    model = joblib.load("cl_old_model.pkl")

    result = ml.exec_ml(str_xml, model)
    log.log("predict() answer: " + str(result))
    log.log("predict() -------------")
    log.log_response(result)

    return result


async def read_body(receive):
    """
    Read and return the entire body from an incoming ASGI message.
    """
    body = b""
    more_body = True

    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)

    return body


async def app(scope, receive, send):
    """
    Echo the request body back in an HTTP response.
    """
    body = await read_body(receive)
    result = predict(body)

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type", b"application/xml"],
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": bytes(result, "utf-8"),
        }
    )


x = b'<REQUEST xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dtstart>2021-08-11 15:16:00</dtstart><sex>1.0</sex><birthday>1964-06-30 00:00:00</birthday><citizenshipid>417</citizenshipid><martialid>2.0</martialid><dependents>0</dependents><sitename>tele2</sitename><DOC>45</DOC><averagemonthlyincome>45000</averagemonthlyincome><Days_since_last_credit>0.0</Days_since_last_credit><Max_overdue>0.0</Max_overdue><Nb_delays_90plus_ever_eq xsi:nil="true"/><CH_length_eq xsi:nil="true"/><S_hare_active_credit xsi:nil="true"/><Score>0.0</Score><MatchingLevel>0.0</MatchingLevel><INTEGRALSCOREValueId>0.129831</INTEGRALSCOREValueId><LIFETIMEBINValueId>4.0</LIFETIMEBINValueId><requested_amount>14955.37</requested_amount></REQUEST>'

print(predict(x))

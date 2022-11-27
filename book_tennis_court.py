# python=3.9.15, requests=2.28.1
# coding:utf-8
import os
import requests
import json
import time
import datetime
import logging
import urllib3
import random
from utils import SMSBot

urllib3.disable_warnings()
random.seed(1234)
logging.basicConfig(format='%(asctime)s  %(message)s', filename=os.path.join(__file__, '../log/book_court.log'),
                    level=logging.INFO)


class HuangLongCourtBooker:
    def __init__(self, venue, item, weekend_free_time='10:00', weekday_free_time='20:00'):
        self.base_url = "https://field.hulasports.com/api/orderlists/get/book"
        # self.base_url = "https://field.hulasports.com/api/venues/59cc96c142fa6b6703843bbf"
        self.headers = {
            'Host': 'field.hulasports.com',
            'Content-Type': 'application/json',
            'Content-Length': '208',
            'Accept': '*/*',
            'Accept-Language': 'en-ca',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, '
                          'like Gecko) Mobile/15E217 MicroMessenger/6.8.0(0x16080000) NetType/WIFI Language/en '
                          'Branch/Br_trunk MiniProgramEnv/Mac',
            'Referer': 'https://servicewechat.com/wx6ecc186775895db1/111/page-frame.html',
            'appId': 'wx6ecc186775895db1',  # small program id
            'memberId': '61ce7fe9b7ef552966c2b707',  # my personal account id
            'If-None-Match': 'W / "29c5-M36aWhykYp0pomdtDwAcjEf7xTY"',
        }
        self.data = {
            "orderDateNum": 1669392000000,
            "_venue": "59cc96c142fa6b6703843bbf",
            "_item": "5a393753d5fcfb9d3ec54265",
            "passBaseOn": "start",
            "showLine": "row",
            "showPassTime": False,
            "_org": "59cb5c718e1e92a702eca340",
            "delayMins": 0
        }
        self.offset = 10
        self.weekend_free_time = weekend_free_time
        self.weekday_free_time = weekday_free_time
        self.venue = venue
        self.item = item

    def validate_data(self):
        try:
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(self.data), verify=False)
            if response.status_code == 200:
                logging.info(response.text)
        except Exception as e:
            logging.error(e)

    def analyze_data(self):
        start_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + datetime.timedelta(days=7)
        rst = '有空闲场地啦! '
        while start_date <= end_date:
            try:
                cur_unix = int(time.mktime(start_date.timetuple()) * 1000)
                free_spot = self.weekend_free_time if start_date.weekday() in (
                    5, 6) else self.weekday_free_time  # Sat/Sun = 5/6
                self.data = {
                    "orderDateNum": cur_unix,
                    "_venue": self.venue,
                    "_item": self.item,
                    "passBaseOn": "start",
                    "showLine": "row",
                    "showPassTime": False,
                    "_org": "59cb5c718e1e92a702eca340",
                    "delayMins": 0
                }
                response = requests.post(self.base_url, headers=self.headers, data=json.dumps(self.data), verify=False)
                datas = json.loads(response.text)
                # print(datas['data']['booking_array'])
                row_num = len(datas['data']['booking_array'])
                col_num = len(datas['data']['booking_array'][0]['booking_infos'])

                for i in range(row_num):
                    for j in range(col_num):
                        content = datas['data']['booking_array'][i]['booking_infos'][j]
                        if content['state']['state'] == '可预订' and content['showStartTime'] >= free_spot:
                            rst += '{0}({1}) {2} {3} 如需预约请回复: {4}\n'.format(start_date.strftime('%Y-%m-%d'),
                                                                                        start_date.strftime('%A')[:3],
                                                                                        content['showTime'],
                                                                                        content['fieldName'],
                                                                                        random.randint(1111, 9999)
                                                                           )
            except Exception as e:
                logging.error(e)

            start_date = start_date + datetime.timedelta(days=1)

        logging.info(rst)
        return rst


if __name__ == '__main__':
    smsbot = SMSBot()
    receivers = '18560155097, 19121922357, 19357660781'
    logging.info('Start scrape court info')

    # squash booker
    s = HuangLongCourtBooker(venue='59cc96c142fa6b6703843bbf', item='5a393753d5fcfb9d3ec54265')
    rst = s.analyze_data()
    if len(rst) > 10:
        smsbot.send_sms(receiver=receivers, body='【Shakewin】' + rst)
    # tennis booker
    t = HuangLongCourtBooker(venue='59cdec566f1d65ba08316176', item='59cc95ca42fa6b6703843bba')
    rst = t.analyze_data()
    if len(rst) > 10:
        smsbot.send_sms(receiver=receivers, body='【Shakewin】' + rst)

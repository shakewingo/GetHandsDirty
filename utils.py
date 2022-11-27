import requests


class SMSBot:
    smsapi = 'https://api.smsbao.com/sms?'

    def __init__(self):
        self.user = 'yaoyingyoyo'
        self.api_key = '7715237cb2eb4abe8883bace010348b3'
        self.statusStr = {
            '0': '短信发送成功',
            '-1': '参数不全',
            '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
            '30': '密码错误',
            '40': '账号不存在',
            '41': '余额不足',
            '42': '账户已过期',
            '43': 'IP地址限制',
            '50': '内容含有敏感词'
        }

    def send_sms(self, receiver, body):
        data = f'u={self.user}&p={self.api_key}&m={receiver}&c={body}'
        send_url = SMSBot.smsapi + data
        try:
            response = requests.get(send_url)
            if response.status_code == 200:
                print(response.text)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    smsbot = SMSBot()
    receivers = '18560155097, 19357660781'
    smsbot.send_sms(receiver=receivers, body='【Shakewin】Find available spot on {0} from {1} at {2}'
                    .format('2022-11-26', '17:00-18:00', '壁球01'))
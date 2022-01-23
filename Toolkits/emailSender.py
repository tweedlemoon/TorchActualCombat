import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

# 这里是Email的配置处
# 发件邮箱
EMAIL_ADDRESS = 'e_lyffly@126.com'
# 发件邮箱密码
EMAIL_PASSWORD = '197379865'
# 收件邮箱
SEND_TO = '197379865@qq.com'


class EmailSender:
    def __init__(self, proName='', logAdd=None, message=''):
        # 程序名
        self.proName = proName
        # 日志地址
        self.logAdd = logAdd
        # 发送的信息
        self.message = message

        # 上面三个预设
        self.emailSender = EMAIL_ADDRESS
        self.senderPasswd = EMAIL_PASSWORD
        self.reciever = SEND_TO

        # 发送邮件的函数
        self.sendEmail()

    def sendEmail(self):
        my_user = self.reciever  # 收件人邮箱账号，我这边发送给自己
        ret = True
        try:
            # 邮件本体
            msg = MIMEText('这是一封测试邮件', 'plain', 'utf-8')
            msg['From'] = formataddr(("From：efly的程序", self.emailSender))  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
            msg['To'] = formataddr(("FK", my_user))  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
            msg['Subject'] = "这是一封测试邮件"  # 邮件的主题，也可以说是标题

            server = smtplib.SMTP_SSL("smtp.126.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25，qq和网易为465
            server.login(self.emailSender, self.senderPasswd)  # 括号中对应的是发件人邮箱账号、邮箱密码
            server.sendmail(self.emailSender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
            server.quit()  # 关闭连接
        except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
            ret = False

        return ret

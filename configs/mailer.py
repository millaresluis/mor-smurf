import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

class Mailer:

    """
    This script initiaties the email alert function.
    """
    def __init__(self):
        # Enter your email below. This email will be used to send alerts.
        # E.g., "email@gmail.com"
        self.EMAIL = "coolzein22@gmail.com"
        # Enter the email password below. Note that the password varies if you have secured
        # 2 step verification turned on. You can refer the links below and create an application specific password.
        # Google mail has a guide here: https://myaccount.google.com/lesssecureapps
        # For 2 step verified accounts: https://support.google.com/accounts/answer/185833
        # Example: aoiwhdoaldmwopau
        self.PASS = "hghejnhxauqfvdwf"
        self.PORT = 465
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)

    # Email Alert
    def send(self, mail):
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)
        # message to be sent
        SUBJECT = 'ALERT!'
        TEXT = f'Social distancing violations exceeded!'
        message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)

        # sending the mail
        self.server.sendmail(self.EMAIL, mail, message)
        self.server.quit()

    # Email Data
    def sendData(self, mail):
        SUBJECT = 'Recorded Data'
        TEXT = f'Attached is the csv file containing the system\'s recorded data'
        message = '{}'.format(TEXT)

        msg = MIMEMultipart()
        body_part = MIMEText(message, 'plain')
        msg['Subject'] = SUBJECT
        # Add body to email
        msg.attach(body_part)

        with open('recordedData.csv','rb') as file:
        # Attach the file with filename to the email
            msg.attach(MIMEApplication(file.read(), Name='recordedData.csv'))

        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)

        self.server.sendmail(self.EMAIL, mail, msg.as_string())
        self.server.quit()

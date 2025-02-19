import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import os

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
        # Define the HTML document
        html = open("configs/html-email/index.html")

        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)

        email_message = MIMEMultipart()
        email_message['Subject'] = f'ALERT!'
        html_email = MIMEText(html.read(), 'html')
        # Attach the html doc defined earlier, as a MIMEText html content type to the MIME message
        email_message.attach(html_email)
        # Convert it as a string
        email_string = email_message.as_string()

        # sending the mail
        self.server.sendmail(self.EMAIL, mail, email_string)
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

        files = ['recordedData.csv', 'analytics/recordedChart.jpg']
        for a_file in files:
            attachment = open(a_file, 'rb')
            file_name = os.path.basename(a_file)
            msg.attach(MIMEApplication(attachment.read(), Name=file_name))
            
            
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)

        self.server.sendmail(self.EMAIL, mail, msg.as_string())
        self.server.quit()

from .mailer import Mailer
from .config import MAIL
from twilio.rest import Client 

def sms_email_notification():
    Mailer().send(MAIL)
    account_sid = 'AC67d82c2b1cf7ae7ddd8bd3e5a2096fd6' 
    auth_token = 'b138eabbee8359096b2376f161147023' 
    client = Client(account_sid, auth_token) 
    message = client.messages.create(  
                                messaging_service_sid='MG6cf9a8edf7c73ae932b2e6ac7ba1eab5', 
                                body='Multiple violators have been identified',      
                                to='+639162367611' 
                            ) 
    
    print(message.sid)
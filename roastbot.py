from roastedbyai import Conversation, Style


convo = Conversation(Style.valley_girl)  

def generate_roast(user_input):
    response = convo.send(user_input) 
    return response

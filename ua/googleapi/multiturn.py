from basic import get_gemini_client

def multi_turn_example():
    client=get_gemini_client()
    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message("I have 2 dogs in my house.")
    print(response.text)

    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    for message in chat.get_history():
        print(f'role - {message.role}',end=": ")
        print(message.parts[0].text)
def multi_turn_stream():
    client = get_gemini_client()
    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message_stream("I have 2 dogs in my house.")
    for chunk in response:
        print(chunk.text, end="")

    response = chat.send_message_stream("How many paws are in my house?")
    for chunk in response:
        print(chunk.text, end="")

    for message in chat.get_history():
        print(f'role - {message.role}', end=": ")
        print(message.parts[0].text)
def main():
    #multi_turn_example()
    multi_turn_stream()
if __name__=="__main__":
    main()

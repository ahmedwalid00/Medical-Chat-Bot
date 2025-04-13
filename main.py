from src.RagPipline import langchain_pipeline

file_path = './storage'
chat_chain = langchain_pipeline(file_path=file_path)

chat_history = []
while True:
    user_input = input("You : ")
    if user_input.lower() == 'exit' :
        print("Thanks for using our app")
        break
    
    response = chat_chain.invoke({'question': user_input})
    answer = response.get('answer', 'Sorry, I could not process that response.')
    print(f"Bot : {answer}")

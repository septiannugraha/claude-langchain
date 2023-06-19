from flask import Flask, jsonify, request
from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    messages = data.get("message")
    llm = ChatAnthropic()

    input = ""
    message_list = []
    for message in messages:
        if message['role'] == 'user':
            message_list.append(
                HumanMessagePromptTemplate.from_template(message['content'])
            )
            input = message['content']
        elif message['role'] == 'assistant':
            message_list.append(
                AIMessagePromptTemplate.from_template(message['content'])
            )

    # Adding SystemMessagePromptTemplate at the beginning of the message_list
    message_list.insert(0, SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. The AI will respond with plain string, replace new lines with \\n which can be easily parsed and stored into JSON, and will try to keep the responses condensed, in as few lines as possible."
    ))

    message_list.insert(1, MessagesPlaceholder(variable_name="history"))

    message_list.insert(-1, HumanMessagePromptTemplate.from_template("{input}"))

    prompt = ChatPromptTemplate.from_messages(message_list)

    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    result = conversation.predict(input=input)

    print(result)
    return jsonify({"status": "success", "message": result})

@app.route('/search', methods=['POST'])
def search_with_assistant():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    messages = data.get("message")

    llm = ChatAnthropic()


    # Get the last message with 'user' role
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    last_user_message = user_messages[-1] if user_messages else None

    # If there is no user message, return an error response
    if not last_user_message:
        return jsonify({"error": "No user message found"}), 400

    input = last_user_message['content']

    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory, 
        agent_kwargs = {
            "memory_prompts": [chat_history],
            "input_variables": ["input", "agent_scratchpad", "chat_history"]
        }
    )
    result = agent_chain.run(input=input)

    print(result)
    return jsonify({"status": "success", "message": result})


if __name__ == '__main__':
    app.run()
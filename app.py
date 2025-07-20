import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
## from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

## Set up Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🫡")
st.title("Text to Math Problem Solver Using Google Gemma 2")

## Taking the API
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

## initialize llm model
if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

## Initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned."
)

## Initialize the Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only mathematical expression needs to be provided."
)

## Prompt Template for reasoning
prompt = """
You are an agent tasked for solving users mathematical questions. Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below.
Question:{question}
Answer:
""" 

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions."
)

## Initialize agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

## Set session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a Math Chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## Start the interaction
## Set a default question
question = st.text_area(
    "Enter your question:", 
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
)

## Handle button click
if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")
        
        
        
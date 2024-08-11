#Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
api_wrapper=WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

#ArXiv tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
arxiv_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

#customised tool (Self made)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()
from langchain.tools.retriever import create_retriever_tool
retriever_tool=create_retriever_tool(retriever,"langsmith_search","Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

tools=[wiki,arxiv,retriever_tool]

#getting llm model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

#predefined prompt of langchain hub
from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

### Agents
from langchain.agents import create_openai_tools_agent
agent=create_openai_tools_agent(llm,tools,prompt)

## Agent Executer
from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)



import streamlit as st
st.title('Q&A chatbot using wikipedia, ArXiv & langsmith_search')
input_text=st.text_input("Search the topic u want")

if input_text:
    response=agent_executor.invoke({"input": input_text})
    st.write(response)


import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field

from typing import Type
from bs4 import BeautifulSoup

import requests
import json
import streamlit as st

from pages.database import ChatStore

load_dotenv('./key.env')

openai_key = os.getenv("OPENAI_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")

uri = f"mongodb+srv://{user}:{password}@qndb.fdshmnw.mongodb.net/?retryWrites=true&w=majority"
storage = ChatStore(uri, "qndb", "qna")


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text


def scrape_website(objective: str, url: str):
    print("Scraping website...")

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }

    data = {
        'url': url
    }

    data_json = json.dumps(data)

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, data=data_json, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("Content: ", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text

    else:
        print(f"HTTP failed with status: {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:\
    "{text}"\
    Summary: 
    """

    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)
    return output


class ScrapeWebsiteInput(BaseModel):
    objective: str = Field(description="The objective and task that users give to the agent.")
    url: str = Field(description="The URL of the website to be scraped.")


class ScrapeWebsiteTool(BaseTool):
    name = "scraped_website"
    description = "Useful when you need to get data from a website URL, passing both URL and objective to the function; DO NOT make up an URL"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError(":(")


tools = [
    Tool(
        name="Search",
        func=search,
        description="Useful for when you need to answer questions about current events, data. You should ask targeted questions."
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""
    You are a world class researcher, who can do detailed research on any topic and produce fact-based results;
    You do not make things up, you will try as hard as possible to gather facts and data to back up the research.
    
    Please make sure you complete the objective above with the following rules:
    1/ You should do enough research to gather as much information as possible about the objective.
    2/ If there are any URLs of relevant links and articles, you will scrape it to gather more information.
    3/ After scraping and searching, you should think: "Is there any new information I should search and scrape based on the data I collected to increase research quality?" If yes, continue; but don't do this for more than three iterations.
    4/ You should not make things up, you should only write facts and data that you have gathered.
    6/ In the final output, you should include all reference data and links to back up your research. You should include all reference data and links to back up your research.
    5/ In the final output, you should include all reference data and links to back up your research. You should include all reference data and links to back up your research.
    """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4")
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


def main():
    st.set_page_config(page_title="WebWeaver Chat", page_icon=":tangerine:")

    st.header("WebWeaver Chat :bird:")

    st.markdown("""
    WebWeaver may take some time to research your topic. Please be patient.
    
    You can ask it questions, or give it a topic to research. Be specific.
    You can also give it follow-up questions.
    """)

    st.warning(
        """If you click away from this page, the answer WebWeaver provides will disappear. You can find it in the database if you need.""",
        icon="⚠️")

    query = st.text_input("Research Goal")

    if query:
        st.write(f"Doing research for {query}...")
        result = agent({"input": query})
        storage.insert_question(query, result)
        st.info(result['output'])


if __name__ == '__main__':
    main()

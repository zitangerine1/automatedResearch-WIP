import os
from dotenv import load_dotenv
# These imports are used to load API keys that should be kept private.
# They do this by loading them as environment variables when the code is ran from a locally stored file.
# The passwords are never stored online and hence they are secure.


# LangChain is a orchestration framework for LLM-powered applications. It lets different LLMs interact with code and enables LLMs to be used in conjunction with other Python libraries. In my case, I combined the front-end library Streamlit with LangChain to build this application. 
from langchain.prompts import PromptTemplate
# PromptTemplate is a class that allows me to use prompts with placeholders that are dynamically filled as the code runs.
# For example, the prompt could be: "I have {num} apples.", and during runtime, it would get filled and the LLM would receive "I have 10 apples."
from langchain.agents import initialize_agent, Tool
# initialize_agent and Tool are functions for setting up an 'agent' - i.e, code that acts like a person would, able to take multi-step actions with their own decision-making. 
# The Tool class allows us to give the agent tools to complete a particular task. In this case, it'll allow the agent to search for information online and scrape websites.
from langchain.agents import AgentType
# Defines the type of agent - in our case, it allows us to use an agent type optimised for OpenAI's GPT models and improves overall performance (as in, decision-making accuracy, speed and response quality etc.)
from langchain.chat_models import ChatOpenAI
# This lets us interact with OpenAI's GPT models.
from langchain.prompts import MessagesPlaceholder
# MessagePlaceholder objects are used in conjunction with PromptTemplates to inject the entire conversation history to a new prompt.
# It contains a list of all the messages exchanged between the agent and the LLM.
# This allows the full context of the previous conversation to be used for further prompting - it's very useful when we're using an agent that needs to remember what has already been researched and scraped.
from langchain.memory import ConversationSummaryBufferMemory
# This slightly differs from MessagesPlaceholder, wherein it holds a summarised version of key points from the conversation.
# ConversationSummaryBufferMemory is a cheaper, faster, more concise alternative of MessagesPlaceholder, but we use MessagesPlaceholder for extracting particular 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# When we scrape a website, the output is often massive, but also filled with useless information such as whitespace, copyright info etc.
# We use RecursiveCharacterTextSplitter to break down the content for the language model.
from langchain.chains.summarize import load_summarize_chain
# A special tool for loading a 'chain of actions' for summarising. In our case, we scrape website content, use the TextSplitter to break it up and then summarise it using the LLM. 
from langchain.tools import BaseTool
# A blueprint (like a class in OOP) for providing the structure and methods to create tools. In our case, we inherit name, description, args_schema and __run() respectively.
# name and description tells the agent what the tool is and what it can do so that it can make decisions on how to use it.
# args_schema is a way to pass parameters into the function, and _run() is the code/function that is ran when the tool is used.
from langchain.schema import SystemMessage
# SystemMessagve is used to pass a message directly to the LLM, separate from the prompt. It provides contextual information on how prompts should be interpreted.
# For example, "You are world-class researcher" is a valid SystemMessage.
from pydantic import BaseModel, Field
# Pydantic is a library for data validation and modelling. 
# BaseModel represents a particular structure of data, in our case, Objective and URL.
# Field are the individual attributes for the model.

from typing import Type
# typing/Type is used for type-hinting. It helps the code be more readable, and is used in the LangChain tools.
from bs4 import BeautifulSoup
# Parses HTML/XML documents. We use this to scrape websites.

import requests
# Makes RESTful API calls for us. We use this to access our searching and webscraping API services by making POST and GET requests.
import json
# This allow us to work with JSON data in our code. We use this when we get data from the APIs, which is returned to us in the form of JSON.
import streamlit as st
# Highly abstracted and simplified front-end model for data applications. It makes the front-end process much less tedious than it has to be as responsiveness, layout are all pre-made for us.

from pages.database import ChatStore
# Imported from the database.py file to allow database R/W I/O processes.

load_dotenv('./key.env')
# .env files store environment variables which are retreived using the os library.
openai_key = os.getenv("OPENAI_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")
# Load the environment variables with all the API keys and passwords needed.

uri = f"mongodb+srv://{user}:{password}@qndb.bjan2b5.mongodb.net/?retryWrites=true&w=majority&appName=qndb"
# Target URL for the database containing all user queries, responses etc., used later in code.
storage = ChatStore(uri, "qndb", "qna")
# Initialises a ChatStore class from database.py, enabling various I/O methods to the database.
# Further documentation regarding ChatStore's methods are in database.py.


def search(query):
    """ Provides the ability for the agent to search the Internet.

    Args:
        query (str): What to search for.

    Returns:
        response (dict): Dictionary with website name, URL.
    """
    url = "https://google.serper.dev/search"
    # URL to make API calls to to search the Internet.

    payload = json.dumps({
        "q": query
    })
    # json.dumps() converts a Python dictionary to a JSON string, which is used to make the API call.
    # JSON makes exchanges between this app and the Serper service easier to understand, as there are standardised parameters that must be present when this app submits data to Serper.
    # In this case, only one parameter is used, "q", standing for query.

    headers = {
        # HTTP headers
        'X-API-KEY': serper_api_key,
        # API key used to access Serper's services - it ensures that the API call made to Serper has permission to use Serper.
        'Content-Type': 'application/json'
        # Tells the API that the format of the request is JSON.
    }

    response = requests.post(url, headers=headers, data=payload)
    # requests.request() sends the actual request to Serper.
    # "POST" specifies the operation - data is being 'posted' to the server.
    # url, headers and data builds the relevant components of the request.
    # The result of the request is stored in 'response'.
    print(response.text)
    # Outputs the results of the search.
    return response.text
    # Returns the output so the result can be used elsewhere.


def scrape_website(objective: str, url: str):
    """ Scrapes a website given a URL and something to look for.

    Args:
        objective (str): What to look for in the website.
        url (str): Website to be scraped.

    Returns:
        str: Content scraped.
    """    

    headers = {
        # HTTP headers
        'Cache-Control': 'no-cache',
        # This tells the browser that it should not scrape a cached version of the website.
        # This means that it will always scrape the live site and not a response from a CDN.
        # Though this increases load on the website's server, the app is guaranteed the most up-to-date information from the website.
        'Content-Type': 'application/json'
        # Specifies that the request is formatted in JSON.
    }
    

    data = {
        'url': url
        # The website to be scraped; this is sent to Browserless so it knows what to scrape.
    }

    data_json = json.dumps(data)
    # Converts the dictionary 'data' to JSON format.

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, data=data_json, headers=headers)
    # requests.post sends the actual request to scrape the website and stores the output in response.
    # 'objective' is not used here as Browserless doesn't actually care what we're looking for - it only scrapes the website.
    # 'objective' will be used later in the program.

    if response.status_code == 200:
        # 200 is a success code, meaning that the request was successfully made and a response was given.
        soup = BeautifulSoup(response.content, "html.parser")
        # BeautifulSoup is a Python package that parses HTML documents.
        # To 'parse' the HTML is to break down the HTML into its component parts, such as tags, attributes and text. It converts it to something called a Document Object Model (DOM), which represents the structure of the page as a tree.
        # A DOM tree looks something like this: https://www.w3schools.com/js/pic_htmltree.gif
        # Parsing the HTML allows us to access specific types of data, such as only returning bold text, or only <div> elements.
        # In our case, we only return the text; i.e, we only return the *content* of the website and ignore all the other HTML elements.
        text = soup.get_text()
        print("Content: ", text)

        if len(text) > 10000:
            # It costs money to give text to an LLM to process.
            # Sometimes, even after filtering for text only, it would still be quite costly to input the entire string to the LLM for it to learn information from it because it's too long.
            # Therefore, we use a cheaper model (60x cheaper) to summarise the data before we give it to the more expensive model.
            output = summary(objective, text)
            return output
        else:
            return text

    else:
        # This condition allows the code to re-run if a website blocks the scraping.
        print(f"HTTP failed with status: {response.status_code}")


def summary(objective, content):
    # The aforementioned cheaper model.
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        # This is a tool, provided by LangChain, and has a hefty explanation on how it works.
        # RecursiveCharacterTextSplitter will try to split text while keeping paragraphs, sentences and words instead of arbitrarily splitting every x characters.
        # 'separators' is how the Splitter tries to maintain paragraph structure. \n stands for a line break. When the TextSplitter encounters a \n\n, it will try to split the text into one chunk. If the size of that chunk is greater than 10000 characters, it will use the next separator, \n. That's what makes it recursive.
        # For example, this piece of text: "Lorem Ipsum \n ...[11000 characters] ... \n\n" is one paragraph extracted from a website. It will split at \n\n first, but realise that it is too long. It will then move to \n and create a chunk there.
        # By default, 'separators' will use ["\n\n", "\n", " ", ""], splitting down to the word (since " "  is a common way to split words). However, we try to maximise the number of sentences this can extract and remove the " " and "" delimiter. This creates natural splits instead of a split that abruptly ends mid-sentence.
        # chunk_overlap describes how much of the previous chunk can be in the next chunk, in characters.
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    # Splits the input text into chunks of text.
    
    map_prompt = """
    Write a summary of the following text for {objective}:\
    "{text}"\
    Summary: 
    """
    # A map_prompt is a special kind of instruction provided to the LLM, part of the LangChain map-reduce summarisation process. map_prompt defines the task that the LLM needs to do for each individual chunk of text it receives from the original content. 
    # The map_prompt is a string that has placeholders and instructions, the placeholders being {objective} and {text}.
    # We use this as there are several chunks, and each time the text and objective will vary, hence we need to use this instead of using parameters for the function.

    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
        # This uses map_prompt and formally creates the PromptTemplate object. The code will use this later and generate specific summarisation prompts for each chunk of text.
    )

    summary_chain = load_summarize_chain(
        # A function provided by LangChain to summarise long sets of documents.
        # In our case, we use it to summarise scraped website content.
        llm=llm,
        # Tells the LangChain orchestrator what LLM to use to summarise.
        chain_type='map_reduce',
        # map_reduce, as explained earlier, breaks the text to summarise into chunks, then creates a 'map' for each chunk. The maps are 'reduced' and the text is summarised into one big summary.
        # Technical details aside, it's also best not to touch this. LangChain only offers two options for a summary chain, being 'stuff' and 'map_reduce'. You may be able to tell that one is more sophisticated than the other, and that we're using the clearly better one. So maybe just refrain from changing this. 
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        # Tools for map_reduce to use, defined and explained above.
        verbose=True
        # Verbose output for... looking cool? Debugging? I don't know, I just like it.
        # Does nothing. Can remove with no consequence.
    )

    output = summary_chain.run(input_documents=docs, objective=objective)
    # Uses the code we wrote above, runs it, and summarises the content of the scraped website, docs.
    return output

# The below classes inherits from Pydantic models.
# Pydantic is a data validation library. From it, we have imported BaseModel and Field. BaseModel is the base 'template' from data model and Field defines the fields of said model.
# ScrapeWebsiteInput inherits from BaseModel. This means that it behaves like a Pydantic model and hence can be used to validate and parse data.
class ScrapeWebsiteInput(BaseModel):
    objective: str = Field(description="The objective and task that users give to the agent.")
    url: str = Field(description="The URL of the website to be scraped.")
    # Field is used to give extra metadata, in this case, a description of each field to provide to the LangChain orchestrator.
    # Each field is statically typed in accordance, and rather aptly, with general Pydantic best-practices.


# The reason we use Pydantic models is because they let us define structure to the data that the tool needs. This means that we define what fields each function will have and what their constraints are. When we're out scraping random websites, we don't really know what the data returned to us will look like. It could have unescaped characters that break the program by ending a string early, or some other formatting issues that stop our code from working. Having Pydantic models will let Python raise validation errors instead of breaking, which we can then catch using try-except statements to keep the code going and continue scraping new websites.


class ScrapeWebsiteTool(BaseTool):
    # This is another Pydantic data validation model, this time inheriting from BaseTool.
    # It has a similar purpose to the previous model, this time provide an args_schema alongside the description of the function.
    # The args_schema defines the er, schema for the arguments. It's kinda great when the library has descriptive names for their fields. args_schema uses the other Pydantic model defined above and tells the orchestrator how the input data is structured.
    name = "scraped_website"
    description = "Useful when you need to get data from a website URL, passing both URL and objective to the function; DO NOT make up an URL"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        # _run is a method that calls the function we defined above to scrape a website.
        return scrape_website(objective, url)

    def _arun(self, url: str):
        # Though I have not implemented a function to run if scrape_website does not run, I'm still required to declare the _arun condition. This does nothing except but raise an error, but also cannot be removed, else the code will not run.
        # This will basically never run, as scrape_website has it's own loop for handling failures and will never return a fail condition to THIS function.
        # https://twitter.com/TransgirlSource/status/1386542050233896970 (same energy, contains swear words.)
        raise NotImplementedError("How'd you get here??")


tools = [
    Tool(
        name="Search",
        func=search,
        description="Useful for when you need to answer questions about current events, data. You should ask targeted questions."
    ),
    ScrapeWebsiteTool(),
    # Here, we formally pass the Pydantic templates to LangChain to use.
    # This is dynamic, so if anyone ever wants to add new functionality, just make a new Pydantic model accordingly to your needs, write the function it needs to run, and add it here as a tool and it'll be automagically implemented.
]

system_message = SystemMessage(
    # SystemMessages are what we tell the LLM before we ask it any questions. Typically, this is used for giving a set of specific guidelines and constraints to the LLM. In our case, we use this to make the LLM think it's really smart and to make sure it gives factual responses.
    # This is a technique called prompt engineering.
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
    # kwargs is a term used in vanilla Python (not to be confused with **kwargs!! They are different things!) that accepted named keywords into a function. It works similarly here as you can add keyword arguments to be passed to the LangChain agent.
    # Keyword arguments are parameters that are passed into the function with a pre-defined value. In this case, the key-value pairs below are our keyword arguments.
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4")
# This initialises the LLM using the ChatOpenAI class from LangChain. 
# temperature as a parameter defines how random the answers will be. Ranging from 0-1, a temperature of 1 will yield very diverse and brave answers, whereas 0 will yield deterministic and consistent responses.
# model indicates which OpenAI model to use, in our case, gpt-4.
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)
# The above line will allow the LLM to use past conversations to improve its current answer. 
# memory_key lets the LLM access the memory from MessagesPlaceholder defined on/close to line 286, using the key "memory".
# return_messages indicate that the memory should return the actual message instead of the message content.


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    # Combines the power of all the code we've written before into a super-powerful research agent! YAY!
    # tools tells the agent what tools it can use, seen in the array defined earlier.
    # agent indicates that the orchestrator should use OpenAI Functions API, which is an API designed specifically for use by LangChain and OpenAI models.
    # agent_kwargs is previously defined and allows it to use the system message and the extra prompt messages.
)


def main():
    st.set_page_config(page_title="WebWeaver Chat", page_icon=":tangerine:")
    # Basic configuration components.
    # page_title is the equivalent to a <title> tag on an HTML page.
    # page_icon is the favicon of the site.

    st.header("WebWeaver Chat :bird: :turtle: ")

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
        # Detects whether the user has pressed 'enter' or otherwise submitted data into the input field. If they have, run the code. Otherwise, it will not run.
        st.write(f"Doing research for {query}...")
        result = agent({"input": query})
        # Store answer and write to db.
        storage.insert_question(query, result)
        st.info(result['output'])

if __name__ == '__main__':
    # This block 'guards' the main function. The function will ONLY run if it is intentionally executed as a script or application, and not as a module. Advisable to not touch, for obvious reasons.
    main()

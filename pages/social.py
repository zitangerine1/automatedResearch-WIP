import streamlit as st
import tweepy
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pages.database import ChatStore
from pages.chat import agent
from pages.chat import search, scrape_website, summary, ScrapeWebsiteInput, ScrapeWebsiteTool

load_dotenv("./key.env")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")

uri = f"mongodb+srv://{user}:{password}@qndb.fdshmnw.mongodb.net/?retryWrites=true&w=majority&appName=qndb"


def phrase_for_socials(query):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_prompt = """You are a social media influencer on Twitter. You cover cutting-edge tech topics such as AI.
    You will take scientifically accurate information and rephrase it to be attractive to social media.
    You will NOT alter the information given, you will only rephrase. The original information should be kept.
    
    Abide by these ruls while completeing the objective above:
    1/ You will make the post engaging to an audience on Twitter.
    2/ You will include all the sources originally provided.
    3/ You will NOT alter any of the content given to you.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), ("user", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    output = chain.invoke({"input": query})

    return output


class TwitterHandler:
    def __init__(self, key, secret_key, token):
        self.auth = tweepy.OAuthHandler(key, secret_key)
        self.auth.set_access_token(token)
        self.api = tweepy.API(self.auth)

    def connTwitter(self):
        try:
            self.api.verify_credentials()
            return True
        except:
            return False

    def post(self, message):
        if self.connTwitter():
            self.api.update_status(message)


def main():
    st.header("Post to Socials")
    storage = ChatStore(uri, "qndb", "qna")

    query = st.text_input("Research Goal")
    post_handler = TwitterHandler() 

    if query:
        st.write(f"Researching {query}...")
        result = agent({"input": query})
        result = phrase_for_socials(result)
        st.info(result)
        
        if post_handler.connTwitter():
            post_handler.post(result)
            st.success("Posted successfully!")
            
        else:
            st.error("Failed to connect to Twitter. Check your credentials.")
        
    option = st.selectbox(
    "Alternatively, fetch from the database of responses...",
    ([f"{i['question'].capitalize()}?" for i in storage.get_all_questions_responses()]),
    index=None,
    placeholder="Select query...",
    )

    if st.button("Use database entry"):
        result = agent({"input": option})
        result = phrase_for_socials(result)
        st.info(result)
        
        if post_handler.connTwitter():
            post_handler.post(result)
            st.success("Posted successfully!")
            
        else:
            st.error("Failed to connect to Twitter. Check your credentials.")
    
    # Open to implementing other social media sources.
    

if __name__ == '__main__':
    main()

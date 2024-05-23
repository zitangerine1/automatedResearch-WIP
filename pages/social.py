import streamlit as st
# Frontend library.
import tweepy
# This is the API that we use to access Twitter specifically. 
# More APIs can be added and I've tried to make the code modular enough so that new social medias can be added.
import os
from dotenv import load_dotenv
# These imports are used to load API keys that should be kept private.
# They do this by loading them as environment variables when the code is ran from a locally stored file.
# The passwords are never stored online and hence they are secure.

from langchain.chat_models import ChatOpenAI
# This lets us interact with OpenAI's GPT models.
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pages.database import ChatStore
# Imports the database management class so that we can access and mutate data as-needed from this file too. OOP is great, in the sense that I can just initialise the class here and not have to redefine every function again.
from pages.chat import agent
from pages.chat import search, scrape_website, summary, ScrapeWebsiteInput, ScrapeWebsiteTool
# Bulk import of functions. IDE may indicate that they are not being used but the agent needs them imported for it to run properly.

load_dotenv("./key.env")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")

tuser = os.getenv("TWITTER_KEY")
tsecret = os.getenv("TWITTER_SECRET")
atoken = os.getenv("TWITTER_TOKEN")
asecret = os.getenv("TWITTER_TOKEN_SECRET")
# Load the environment variables with all the API keys and passwords needed.

uri = f"mongodb+srv://{user}:{password}@qndb.bjan2b5.mongodb.net/?retryWrites=true&w=majority&appName=qndb"
# Target URL for the database containing all user queries, responses etc., used later in code.


def phrase_for_socials(query):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # This initialises the LLM using the ChatOpenAI class from LangChain. 
    # temperature as a parameter defines how random the answers will be. Ranging from 0-1, a temperature of 1 will yield very diverse and brave answers, whereas 0 will yield deterministic and consistent responses. 
    # Feel free to play around with temperature here, I feel like it'll be quite interesting to see what it does with a high temperature, especially when it's trying to be bubbly for social media.
    system_prompt = """You are a social media influencer on Twitter. You cover cutting-edge tech topics such as AI.
    You will take scientifically accurate information and rephrase it to be attractive to social media.
    You will NOT alter the information given, you will only rephrase. The original information should be kept.
    
    Abide by these rules while completeing the objective above:
    1/ You will make the post engaging to an audience on Twitter.
    2/ You will include all the sources originally provided.
    3/ You will NOT alter any of the content given to you.
    """
    # Here, I'm using a simpler version of a system message as compared to the one used in chat.py.
    # This simply stores it as a variable, as I don't have any agent kwargs to pass this to. The rephrasing for social media is a pipeline, passing the response from the agent to a LLM with a specific prompt and having it rephrase. It's not a dynamic process that requires live decision making, unlike the chat.py program.
    # Hence, a less complex system with no agency is used. The text to rephrase is simply passed through the LLM with this system prompt and spat out for us to use.

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), ("user", "{input}")
    ])
    # As seen, a much simpler version of piping input to an LLM. We just give it the input and the system message for prompt engineering.

    output_parser = StrOutputParser()
    # StrOutputParser parses the result from the LLM into the most likely string. 
    # Or in a more understandable way, it converts the output from the LLM to something that is more humanly understandable, and is also the best response that the LLM provided.

    chain = prompt | llm | output_parser
    # This is a unique statement written in LangChain Expression Language (LCEL). Normally, the | (called pipe) symbol is typically representative of an OR operator in vanilla Python, but in LCEL, it acts more like a pipe in Linux.
    # Pipes in Linux combines two commands, such that the output of the first command is passed into the second. For example, "ls | grep file.txt". ls lists all the files in a directory. The output of this is passed to grep, which finds a specific string among a larger set of characters. grep looks for 'file.txt' in the output that ls put out.
    # This is similar in that LLM is using the user prompt, and the output_parser is using the output from the LLM.
    # This variable only declares what the LLM should do and doesn't actually execute anything. 
    output = chain.invoke({"input": query})
    # Runs the chain defined above and passes in the input.

    return output

# Again, a class is used for modularity. In hindsight, it might've been better to make a general class for "SocialMediaHandler" and have its subclasses be for specific media and override its 'connect' and 'post' features.
# For now, this is absolutely not nessecary as the client only wants to post to Twitter, but this places a good stepping-stone to adding more social media.

class TwitterHandler:
    def __init__(self, key, secret_key, token, token_secret):
        self.auth = tweepy.OAuthHandler(key, secret_key)
        # Uses OAuth to authenticate access to Twitter.
        # Declared during initialisation so that it doesn't take time to connect when the user wants to post.
        self.auth.set_access_token(token, token_secret)
        self.api = tweepy.API(self.auth)
        # Setup to use the API and authenticate the user.

    def connTwitter(self):
        # try-except is used here as the next function, post, needs an explicit declaration of a failure condition so that re-run or non-break failures can be declared.
        try:
            self.api.verify_credentials()
            return True
        except:
            return False
        

    def post(self, message):
        # Post functionality. Add features using Tweepy documentation if needed.
        if self.connTwitter():
            self.api.update_status(message)


def main():
    st.header("Post to Socials")
    storage = ChatStore(uri, "qndb", "qna")

    query = st.text_input("Research Goal")
    post_handler = TwitterHandler(tuser, tsecret, atoken, asecret) 

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
    (f"{i['question'].capitalize()}{'?' if '?' not in i['question'] else ''}" for i in storage.get_all_questions_responses()),
    # This line is so condensed.
    # Firstly, this uses what's called a list comprehension. It creates a list using an iterable object, in this case the output from the database containing the question-answer pairs. The item being iterated over is 'storage.get_all_questions_responses(), the item is i and the expression is f"{i['question'].capitalize()}?".
    # Let's break this down into parts: f"{i['question].capitalize()}?", uses an f-string, which is a way to place variables in a string. The variable we place is 'i['question'], which references the value with the key 'question' from i. i is the raw output from the database, and by selecting the value associated with the key 'question', we fetch whatever question the user had asked. For example, {"question": "who am i"} will turn into "Who am i?" in final output.
    # The conditional expression '{'?' if '?' not in i['question'] else ''}' checks if the character '?' is not present in the 'question' string using the 'not in' operator. If '?' is not found in the 'question', the expression evaluates to '?', adding a question mark to the end of the sentence. If '?' is already present in the 'question', the expression evaluates to an empty string '', effectively adding nothing to the end of the sentence.
    # .capitalize is self explanatory, and the question mark at the end simply emphasises the fact that it was a user-asked question. These parts are for aesthetic only.
    index=None,
    placeholder="Select query...",
    )

    if st.button("Use database entry"):
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

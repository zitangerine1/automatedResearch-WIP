from pymongo.mongo_client import MongoClient
# This is a nessecary import to allow us to connect to our database.
from dotenv import load_dotenv
# This allows us to import our API keys and passwords from a 'dotenv', which holds them in the form of an environmental variable. The file is then omitted from any saves that are exposed to the public so that only I, the developer, have access to them.
import os
# Nessecary to access the passwords stored in the 'dotenv' file. The file stores the passwords in what are called environmental variables.
import re
# Regular Expression (Regex) expression for formatting/filtering the output from the database so that it displays properly on the webpage.
import streamlit as st 
# Frontend library.

load_dotenv("./key.env")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")
# Loads the dotenv file with all the passwords and API keys needed.

uri = f"mongodb+srv://{user}:{password}@qndb.fdshmnw.mongodb.net/?retryWrites=true&w=majority&appName=qndb"
# Target URL for the database containing all user queries, responses etc.


class ChatStore:
    # I chose to use a class to represent, access and mutate values in the database for several reasons.
    # OOP makes the database MODULAR - I can add or remove any number of methods to access, manipulate or otherwise use the database, and it will apply to all instances across my codebase. Here, I define a set of instance methods that provide basic access to the database, but I can choose to add more complex function later on and it will be universally usable.
    # It also makes my code REUSABLE. Instead of rewriting the functions every time I want to use them in a different file, I simply initialise a ChatStore class in the file. Granted, Python provides the ability to import functions from other files, but that is also object-oriented in nature, as Python treats each .py file as a module. By writing it this way, I make the class relation more clear and explicit.
    def __init__(self, constr, database_name, collection_name):
        self.client = MongoClient(constr)
        # Establishes the connection using the uri variable.
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]
        # Selects the database collection, using the parameter provided in the function.

        # Ensure collection exists.
        if collection_name not in self.database.list_collection_names():
            # If the collection doesn't exist, create it.
            print(f"Collection '{collection_name}' not found. Creating...")
            self.database.create_collection(collection_name)

    def insert_question(self, question, response):
        # Prepares the data in a format readable by the database, as key-value pairs.
        data = {"question": question, "response": f"{response['output']}"}
        self.collection.insert_one(data)
        print(f"Inserted: {data}")

    def query_response(self, question):
        # Searches for a result from the database.
        query = {"question": question}
        result = self.collection.find_one(query)
        # Returns None if no matching query is found. 
        # Reference None, not False if using this function.
        return result["response"] if result else None

    def get_all_questions_responses(self):
        # Fetches all the documetns in the collection.
        # It excludes the _id field, as it is not very useful to us.
        # Remove the parameter if further development requires using the id.
        cursor = self.collection.find({}, {"_id": 0})
        return list(cursor)

    def clear_all_data(self):
        # Pretty self explanatory. Wipes the database.
        self.collection.delete_many({})


def main():
    st.title("Past Queries")
    
    # Connects to the database.
    storage = ChatStore(uri, "qndb", "qna")

    # Retrieve all question/response pairs from MongoDB
    all_data = storage.get_all_questions_responses()
    # So that it is in chronological order, from most recently asked to last.
    all_data.reverse()
    st.write("---")

    # Display each question and response pair
    for data in all_data:
        st.write(f"**Question:** {data['question']}")
        output = data['response'].replace("\n", "<br>")
        output = re.sub(r'\[\^(\d+)\^\]', r'<sup>\1</sup>', output)
        # This one line of code was a lot ☹️. 
        # This regex expression will catch this particular string: "[^(number)^]" from any database output. We need to catch this as this is how citation notation is imported into the database. When it is output, it looks ugly as it will look something like "blah blah [^1^]".
        # \[ means that it is looking for the "[" character. The backslash 'escapes' it, meaning that it will not be taken as code, but rather as a literal string to look for.
        # The same applies to \^ and the \^\] at the end. All of them mean that it is looking for that exact combination of characters.
        # The \d character is regex shorthand for 'any digit'. The expression will catch any number 0-9 in that slot. The + quantifier means 'one or more' of that element.
        # re.sub replaces anything the regex statement catches with the string afterwards. We use <sup> to indicate that it should be shown as superscript instead of [^1^].
        
        st.markdown(f"**Response:** {output}", unsafe_allow_html=True)
        # Printing the final, regex processed output from the database. 
        # unsafe_allow_html permits the <sup> element to be used. It is called 'unsafe' as it allows potential HTML statement injection. For example, someone who has hacked into our mongoDB instance can replace <sup> with <script> and run JS code to infiltrate our website. Thankfully, me and my project are too irrelevant for that to matter, so I'm not too worried about it.
        st.write("---")  # Separator for better readability

    if st.button("Clear All Data"):
        storage.clear_all_data()
        st.success("Data cleared successfully. Reload the page to see changes.")
        

if __name__ == "__main__":
    # This block 'guards' the main function. The function will ONLY run if it is intentionally executed as a script or application, and not as a module. Advisable to not touch, for obvious reasons.
    main()

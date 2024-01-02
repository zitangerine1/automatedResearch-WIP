from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
import re
import streamlit as st

load_dotenv("./key.env")
password = os.getenv("MONGO_PWD")
user = os.getenv("MONGO_USER")

uri = f"mongodb+srv://{user}:{password}@qndb.fdshmnw.mongodb.net/?retryWrites=true&w=majority"


class ChatStore:
    def __init__(self, constr, database_name, collection_name):
        self.client = MongoClient(constr)
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    def insert_question(self, question, response):
        data = {"question": question, "response": f"{response['output']}"}
        self.collection.insert_one(data)
        print(f"Inserted: {data}")

    def query_response(self, question):
        query = {"question": question}
        result = self.collection.find_one(query)
        return result["response"] if result else None

    def get_all_questions_responses(self):
        cursor = self.collection.find({}, {"_id": 0})
        return list(cursor)

    def clear_all_data(self):
        self.collection.delete_many({})


def main():
    st.title("Past Queries")

    storage = ChatStore(uri, "qndb", "qna")

    # Retrieve all question/response pairs from MongoDB
    all_data = storage.get_all_questions_responses()
    all_data.reverse()
    st.write("---")

    # Display each question and response pair
    for data in all_data:
        st.write(f"**Question:** {data['question']}")
        output = data['response'].replace("\n", "<br>")
        output = re.sub(r'\[\^(\d+)\^\]', r'<sup>\1</sup>', output)
        st.markdown(f"**Response:** {output}", unsafe_allow_html=True)
        st.write("---")  # Separator for better readability

    if st.button("Clear All Data"):
        storage.clear_all_data()
        st.success("Data cleared successfully. Reload the page to see changes.")


if __name__ == "__main__":
    main()

## Installation and Usage
#### To install
1. Clone this repository to your local machine.
2. Make sure you have Python 3.x installed, as well as pip.
3. `pip install -r requirements.txt`.
4. Follow the usage guide below, then `streamlit run landing.py`.
---

#### Usage Guide
To use this code, you will need several things first.
- I am absolutely not able to financially fund many people using this project, as it makes many OpenAI API calls. You need your **own OpenAI API Key**.
- You will also need an API key for **Browserless** and **Serper** respectively. As long as this application is not used extensively, it'll be fully free to use and non-rate-limiting.
- You need your own MongoDB instance. Alternatively, use another database solution by editing the `ChatStore` class in `database.py`.

Once you have all these, create a file called `key.env` in the main folder  and add the variables in accordance with the names in `chat.py`.




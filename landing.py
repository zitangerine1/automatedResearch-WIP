import streamlit as st


def main():
    st.set_page_config(page_title="WebWeaver Home", page_icon=":tangerine:", layout="wide")
    st.title("Welcome to WebWeaver.")

    st.markdown(
        "**WebWeaver** is an AI-powered research tool. What differentiates it from the rest is two main features: searching and webscraping.")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Why was WebWeaver made?"):
            st.markdown("""
        ### Background
Typically, LLMs are constrained by their training data and hence sometimes *hallucinate*, which is when they provide a grammatically correct answer that contains fabricated or incorrect information.

For example, a hallucination:
```
Q: Who was the president of the United States in 2020?
A: John F. Kennedy
```

These happen for various reasons, but predominantly due to flaws in the training data or outdated information. WebWeaver mitigates hallucinations using **LangChain**.""")

        with st.expander("Why use WebWeaver over ChatGPT?"):
            st.markdown("""### Improvements
WebWeaver, as aforementioned, uses a scraping and searching tool. This circumvents risk of hallucination as we use LangChain to ensure that the answer the LLM provides is fully based on sources it found online, and that it lists its sources for human verification.""")

    with col2:
        with st.expander("What powers WebWeaver?"):
            st.markdown("""### LangChain
LangChain is an orchestration framework for building LLM powered applications. This means that it coordinates many components and presents it as one whole, much like a computer; abstracting the process of using multiple AI services.

That explanation may be a little hard to understand. Let's say we're baking a cake. The user could prompt as such:
```
Q1: How can I bake a cake?
A1: Prepare ingredients, mix dry and wet ingredients, then bake.

Q2: What ingredients do I need for a chocolate cake?
A2: Flour, Chocolate, Sugar ...

Q3: How do I bake it?
A3: Preheat oven, set a timer, take out and rest when done.
```

LangChain simplifies this process by allowing the developer to create 'chains' that perform tasks intelligently based on the user prompt. For instance:
```
Q: How can I bake a cake?
Chain: Components of cake-making process -> Ingredients -> Baking -> Summarise

A: You need to prepare flour, chocolate, sugar (...), preheat the oven and rest it after the timer expires.
```

Of course, this is a gross oversimplification, and any self-respecting AI developer is probably plotting to murder me, but hopefully you understand.""")


if __name__ == '__main__':
    main()

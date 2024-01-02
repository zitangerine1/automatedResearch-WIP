**WebWeaver** is an AI-powered research tool. What differentiates it from the rest is two main features: searching and webscraping.
### Background
Typically, LLMs are constrained by their training data and hence sometimes *hallucinate*, which is when they provide a grammatically correct answer that contains fabricated or incorrect information.

For example, a hallucination:
```
Q: Who was the president of the United States in 2020?
A: John F. Kennedy
```

These happen for various reasons, but predominantly due to flaws in the training data or outdated information. WebWeaver mitigates hallucinations using **LangChain**.
### LangChain
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

LangChain simplifies this process by allowing the developer to create 'chains' that 
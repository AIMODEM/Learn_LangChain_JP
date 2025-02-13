# [Build a simple LLM application with chat models and prompt templates:](https://python.langchain.com/docs/tutorials/llm_chain/)

Build a simple **LLM application** with LangChain to **translate text from English into another language**.

After this tutorial, we'll have a high level overview of:
- Using [language models](https://python.langchain.com/docs/concepts/chat_models/)
- Using [prompt templates](https://python.langchain.com/docs/concepts/prompt_templates/)
- Debugging and tracing your application using [LangSmith](https://docs.smith.langchain.com/)

To install LangChain:
```bash
pip install langchain
```

## LangSmith:
Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com/).

Set your environment variables to start logging traces:
```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```
or
```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Using Language Models:
```bash
pip install -qU "langchain[groq]"
pip install -qU "langchain[openai]"
pip install -qU "langchain[anthropic]"
pip install -qU "langchain[google-vertexai]"
pip install -qU "langchain[aws]"
pip install -qU "langchain[mistralai]"
pip install -qU "databricks-langchain"
```

GROQ:
```python
import getpass
import os

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model
model = init_chat_model("llama3-8b-8192", model_provider="groq")
```

OpenAI:
```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")
```

AWS:
```python
# Ensure your AWS credentials are configured
from langchain.chat_models import init_chat_model
model = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")
```

Google VertexAI:
```python
# Ensure your VertexAI credentials are configured
from langchain.chat_models import init_chat_model
model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
```

To simply call the model, we can pass in a list of messages to the .invoke method.
```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)
```

Messages:
- https://python.langchain.com/docs/concepts/messages/
- Messages are the unit of communication in chat models.
- They are used to represent the input and output of a chat model, as well as any additional context or metadata that may be associated with a conversation.

HumanMessage:
- Message from a human.
- HumanMessages are messages that are passed in from a human to the model.
- https://python.langchain.com/api_reference/core/messages/langchain_core.messages.human.HumanMessage.html

SystemMessage:
- Message for priming AI behavior.
- The system message is usually passed in as the first of a sequence of input messages.
- https://python.langchain.com/api_reference/core/messages/langchain_core.messages.system.SystemMessage.html

mode.invoke(messages) returns:
```python
AIMessage(content='Ciao!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0705bf87c0', 'finish_reason': 'stop', 'logprobs': None}, id='run-32654a56-627c-40e1-a141-ad9350bbfd3e-0', usage_metadata={'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```
ChatModels receive message objects as input and generate message objects as output.


**Streaming**:
```python
for token in model.stream(messages):
    print(token.content, end="|")
```


## Prompt Templates:
Right now we are passing a list of messages directly into the language model. Where does this list of messages come from? Usually, it is constructed from a combination of user input and application logic. This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language model. Common transformations include adding a system message or formatting a template with the user input.

**Prompt template take in raw user input and return data (a prompt) that is ready to pass into a language model.**

Let's create a prompt template here. It will take in two user variables:
- language: The language to translate text into
- text: The text to translate

```python
from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English language into {language} language"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
```

The input to this prompt template is a dictionary.
```python
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

prompt
```
We can see that it returns a ChatPromptValue that consists of two messages. If we want to access the messages directly we do:
```python
prompt.to_messages()
```

Finally, we can invoke the chat model on the formatted prompt:
```python
response = model.invoke(prompt)
print(response.content)
```

**LLM application to translate the text from ENGLISH to any other language:**
```bash
pip install langchain
pip install -qU "langchain[groq]"
```

```python
import getpass
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "Hi! How are you?"})

response = model.invoke(prompt)
print(response.content)
```


## Conclusion:
We have created our first simple LLM application.
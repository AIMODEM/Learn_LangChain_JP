[LangChain](https://python.langchain.com/docs/introduction/)

# Introduction:
LangChain is a framework for developing applications powered by large language models (LLMs).

LangChain simplifies every stage of the LLM application lifecycle:

- Development: Build your applications using LangChain's open-source [components](https://python.langchain.com/docs/concepts/) and [third-party integrations](https://python.langchain.com/docs/integrations/providers/). Use [LangGraph](https://python.langchain.com/docs/concepts/architecture/#langgraph) to build stateful agents with first-class streaming and human-in-the-loop support.

- Productionization: Use [LangSmith](https://docs.smith.langchain.com/) to inspect, monitor and evaluate your applications, so that you can continuously optimize and deploy with confidence.

- Deployment: Turn your LangGraph applications into production-ready APIs and Assistants with [LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/).

```bash
pip install -qU "langchain[groq]"
pip install -qU "langchain[openai]"
pip install -qU "langchain[anthropic]"
pip install -qU "langchain[google-vertexai]"
pip install -qU "langchain[aws]"
pip install -qU "langchain[cohere]"
pip install -qU "langchain-nvidia-ai-endpoints"
pip install -qU "langchain[fireworks]"
pip install -qU "langchain[mistralai]"
pip install -qU "langchain[together]"
pip install -qU "databricks-langchain"
```

## Chat models:
- [Chat models](https://python.langchain.com/docs/concepts/chat_models/) are language models that use a sequence of [messages](https://python.langchain.com/docs/concepts/messages/) as inputs and return messages as outputs (as opposed to using plain text). These are generally newer models.
- Messages are the unit of communication in chat models. They are used to represent the input and output of a chat model, as well as any additional context or metadata that may be associated with a conversation.
- What is inside a message?
  - https://python.langchain.com/docs/concepts/messages/
  - A message typically consists of the following pieces of information:
    - Role: The role of the message (e.g., "user", "assistant").
    - Content: The content of the message (e.g., text, multimodal data).
    - Additional metadata: id, name, token usage and other model-specific metadata.
- SystemMessage -- for content which should be passed to direct the conversation
- HumanMessage -- for content in the input from the user.
- AIMessage -- for content in the response from the model.
- Multimodality -- for more information on multimodal content.

### GROQ:
```python
import getpass, os

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
from langchain.chat_models import init_chat_model

model = init_chat_model("llama3-8b-8192", model_provider="groq")
model.invoke("Hello, world!")
```

### OpenAI:
```python
import getpass, os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
model.invoke("Hello, world!")
```

### Anthropic:
```python
import getpass, os

if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
model.invoke("Hello, world!")
```

### Google VerterAI:
```python
# Ensure your VertexAI credentials are configured
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
model.invoke("Hello, world!")
```

### AWS:
```python
# Ensure your AWS credentials are configured
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")
model.invoke("Hello, world!")
```

### Mistral AI:
```python
import getpass, os

if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")
from langchain.chat_models import init_chat_model

model = init_chat_model("mistral-large-latest", model_provider="mistralai")
model.invoke("Hello, world!")
```


# Architecture:
The LangChain framework consists of multiple open-source libraries. 
1) langchain-core:
   - Base abstractions for chat models and other components.
2) Integration packages:
   - e.g. langchain-openai, langchain-anthropic, etc.
3) langchain:
   - Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
4) langchain-community:
   - Third-party integrations that are community maintained.
5) langgraph:
   - Orchestration framework for combining LangChain components into production-ready applications with persistence, streaming, and other key features.











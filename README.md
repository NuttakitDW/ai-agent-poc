### The Complete Guide to Building Your First AI Agent with LangGraph. (It’s Easier Than You Think)
https://medium.com/data-science-collective/the-complete-guide-to-building-your-first-ai-agent-its-easier-than-you-think-c87f376c84b2

Three months into building my first commercial AI agent, everything collapsed during the client demo.

What should have been a seamless autonomous workflow turned into an embarrassing loop of repeated clarification requests and inconsistent decisions. The client remained polite but visibly disappointed.

After they left, I spent hours analyzing the failure and discovered I had fundamentally misunderstood agent architecture — I’d built an overcomplicated system with poor decision boundaries and no clear reasoning paths.

That failure transformed my approach and became the foundation for how I explain these systems. Once you understand the core principles, building effective agents becomes surprisingly simple.


Introducing AI Agents
I hate when people overcomplicate this stuff, so let’s keep it simple.

AI agents are systems that: (1) think through problems step-by-step, (2) connect to external tools when needed and, (3) learn from their actions to improve over time.

Unlike chatbots that just respond to prompts, agents take initiative and work through tasks on their own. They’re the difference between having someone answer your questions about data and having someone actually analyze that data for you.

From Models to Agents
Before agents, we built AI solutions as separate, disconnected components — one model for understanding text, another for generating code, and yet another for processing images.

This fragmented approach (1) forced users to manage workflows manually, (2) caused context to vanish when moving between different systems, and (3) required to build custom integrations for each process step.

Agents transform this paradigm.

Unlike traditional models handling isolated task, an agent manages various capabilities while maintaining an overall understanding of the entire task.

Agents don’t just follow instructions — they adapt and makes intelligent decisions about next steps based on what it learns during the process, similar to how we human operate.

The Core Advantage of AI Agents
Let’s understand agents’ capabilities by looking at a specific task: analyzing articles on Medium.

Traditional AI breaks this into isolated steps — summarizing, extracting key terms, categorizing content, and generating insights — each step requiring explicit human coordination.

The limitation isn’t just that models work in isolation, but that you must manually sequence the entire process, explicitly manage knowledge transfer between steps, and independently determine what additional operations is needed based on intermediate results.

Agent-based approaches, in contrast, autonomously perform each step without loosing side of the broader goal.

The Building Blocks of Agent Inteligence
AI agents rest on three fundamental principles:

State Management: The agent’s working memory tracking context about what it’s learned and aims to accomplish
Decision-Making: The agent determining which approach make sense based on current knowledge
Tool Use: The agent knowing which tool solves each specific problem
Building AI Agents with LangGraph
Now that you understand what AI agents are and why they matter, let’s build one using LangGraph — a LangChain’s framework for building robust AI agents.

What I really like about LangGraph is that it lets you map your agent’s thinking and actions as a graph. Each node represent a capability (like searching the web or writing code), and connections between nodes (edges) control information flows.

When I started building agents, this approached clicked for me because I could actually visualize my agent thinking process.

Your First Agent: Medium Articles Analyzer
Let’s see how we can create a text analysis agent with LangGraph.

This agent will read articles, figure out what they’re about, extract important elements, and deliver clean summaries — essentially your personal research assistant.

Setting Up The Environment
First, you need to set up your development environment.

Step 1 — Create a project directory:

mkdir ai_agent_project cd ai_agent_project
Step 2— Create and activate a virtual environment:

# On Windows 
python -m venv agent_env agent_env\Scripts\activate 

# On macOS/Linux
python3 -m venv agent_env source agent_env/bin/activate
Step 3— Install necessary packages:

pip install langgraph langchain langchain-openai python-dotenv
Step 4— Set up your OpenAI API:

I’m using GPT-4o mini as our agent’s brain, but you can swap it for any LLM you prefer. If you don’t have an API key:

Create an account with OpenAI
Navigate to the API Keys section
Click “Create new secret key”
Copy your API key
Step 5— Create a .env file:

# On Windows
echo OPENAI_API_KEY=your-api-key-here > .env 

# On macOS/Linux
echo "OPENAI_API_KEY=your-api-key-here" > .env
Replace ‘your-api-key-here’ with your OpenAI API key.

Step 6 — Create a test file named test_setup.py

python

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv() 

# Initialize the ChatOpenAI instance 
llm = ChatOpenAI(model="gpt-4o-mini") 

# Test the setup 
response = llm.invoke("Hello! Are you working?") print(response.content)
Step 7— Run the test:

python test_setup.py
If you receive a response, congrats, your environment is ready for agent building!

Creating Our First Agent
First, import the necessary libraries:

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
StateGraphmanage information flow between agent components. PromptTemplate creates consistent instructions, and ChatOpenAI connects to OpenAI’s char models to power agent’s thinking.

Creating Our Agent’s Memory
Our agent needs memory to track it’s progress, we can create this with a TypedDict:

# Importing necessary types from the typing module
from typing import TypedDict, List  

# Define a TypedDict named 'State' to represent a structured dictionary
class State(TypedDict):

    text: str  # Stores the original input text
    classification: str  # Represents the classification result (e.g., category label)
    entities: List[str]  # Holds a list of extracted entities (e.g., named entities)
    summary: str  # Stores a summarized version of the text
Now that our agent has memory, let’s give it some thinking capabilities!

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
Setting Temperature=0 ensures our agent consistently chooses the most likely response — this is critical for agents following specific reasoning patterns. As a refresher, temperature functions as the “creativity knob” for LLMs:

Temperature=0: Focused, deterministic responses
Temperature=1: More varied, creative outputs
Temperature=2: Wild, sometimes incoherent ideas
If your agent makes strange decisions, check your temperature setting first!

Adding Agent’s Capabilities
Now we’ll build specialized tools for our agent, each handling a specific task type.

First, our classification capability:

def classification_node(state: State):
   """
   Classify the text into one of predefined categories.
   
   Parameters:
       state (State): The current state dictionary containing the text to classify
       
   Returns:
       dict: A dictionary with the "classification" key containing the category result
       
   Categories:
       - News: Factual reporting of current events
       - Blog: Personal or informal web writing
       - Research: Academic or scientific content
       - Other: Content that doesn't fit the above categories
   """

   # Define a prompt template that asks the model to classify the given text
   prompt = PromptTemplate(
       input_variables=["text"],
       template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:"
   )

   # Format the prompt with the input text from the state
   message = HumanMessage(content=prompt.format(text=state["text"]))

   # Invoke the language model to classify the text based on the prompt
   classification = llm.invoke([message]).content.strip()

   # Return the classification result in a dictionary
   return {"classification": classification}
This function uses a prompt template to give clear instructions to our AI model. The function takes our current state (containing the text we’re analyzing) and returns its classification.

Next, our entity extraction capability:

def entity_extraction_node(state: State):
  # Function to identify and extract named entities from text
  # Organized by category (Person, Organization, Location)
  
  # Create template for entity extraction prompt
  # Specifies what entities to look for and format (comma-separated)
  prompt = PromptTemplate(
      input_variables=["text"],
      template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
  )
  
  # Format the prompt with text from state and wrap in HumanMessage
  message = HumanMessage(content=prompt.format(text=state["text"]))
  
  # Send to language model, get response, clean whitespace, split into list
  entities = llm.invoke([message]).content.strip().split(", ")
  
  # Return dictionary with entities list to be merged into agent state
  return {"entities": entities}
This function processes the document and returns a list of key entities like important names, organizations, and places.

Finally, our summarization capability:

def summarize_node(state: State):
    # Create a template for the summarization prompt
    # This tells the model to summarize the input text in one sentence
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following text in one short sentence.
        
        Text: {text}
        
        Summary:"""
    )
    
    # Create a chain by connecting the prompt template to the language model
    # The "|" operator pipes the output of the prompt into the model
    chain = summarization_prompt | llm
    
    # Execute the chain with the input text from the state dictionary
    # This passes the text to be summarized to the model
    response = chain.invoke({"text": state["text"]})
    
    # Return a dictionary with the summary extracted from the model's response
    # This will be merged into the agent's state
    return {"summary": response.content}
This function distills the document into a concise summary of its main points.

Combined, these skills enable our agent to understand content types, identify key information, and create digestible summaries — each function following the same pattern of taking current state, processing it, and returning useful information to the next function.

Finalizing the Agent Structure
Now we’ll connect these capabilities into a coordinated workflow:

workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()
Congratulations!

You’ve built an agent that flows from classification to entity extraction to summarization in a coordinated sequence, allowing it to understand text types, identify important entities, create summaries, and then complete the process.


Overview of LangGraph Agent’s Architecture
The Agent in Action
Now let’s test our agent with a sample text:

# Define a sample text about Anthropic's MCP to test our agent
sample_text = """
Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
"""

# Create the initial state with our sample text
state_input = {"text": sample_text}

# Run the agent's full workflow on our sample text
result = app.invoke(state_input)

# Print each component of the result:
# - The classification category (News, Blog, Research, or Other)
print("Classification:", result["classification"])

# - The extracted entities (People, Organizations, Locations)
print("\nEntities:", result["entities"])

# - The generated summary of the text
print("\nSummary:", result["summary"])
Running this code processes the text through each capabilities:

Classification: Technology
Entities: [‘Anthropic’, ‘MCP’, ‘Model Context Protocol’]
Summary: Anthropic’s MCP is an open-source protocol enabling seamless application interaction with various API systems.
What’s impressive isn’t just the final results but how each stage builds upon the previous one. This mirrors our own reading process: we first determine content type, then identify important names and concepts, and finally create mental summary connecting everything.

This agent-building approach extends far beyond our tech example. You can use a similar set-up for:

Personal development articles — categorize growth areas, extract actionable advice, and summarize key insights
Startup founder stories — recognize business models, funding patterns, and growth strategies
Product reviews — identify features, brands, and recommendations
The Limits of AI Agents
Our agent works within the rigid framework of nodes and connections we designed.

This predictable limits its adaptability. Unlike humans, agents follow fixed pathways and can’t pivot when when facing unexpected situations.

Contextual understanding is another limitation. This agent can process text but lacks the broader knowledge and cultural nuances that humans naturally grasp. The agent operates within the scope of the text provided, though adding internet search can help supplement its knowledge.

The black box problem exists in agentic systems too. We see inputs and outputs but not internal decision-making. Reasoning models like GPT-o1 or DeepSeek R1 offer more transparency by showing their thinking process, though we still can’t fully control what happens inside.

Finally, these systems aren’t fully autonomous and require human supervision, especially for validating outputs and ensuring accuracy. As with any other AI system, the best results come from combining AI capabilities with human oversight.

Understanding these limits helps us build better systems and know exactly when humans needs to step in. The best results come from combining AI capabilities with human expertise.

The Painful Lesson From My Agent Demo Disaster
Looking back at my embarrassing client demo failure, I now recognize that understanding agents’ limitations is essential for success. My overcomplicated system crashed because I ignored the fundamental constraints of agent architecture.

By accepting that agents: (1) need clear frameworks, (2) operate within defined pathways, (3) operate as partial black boxes and, (4) require human oversight, I’ve built systems that actually deliver results instead of endless clarification loops.

That painful demo taught me the most valuable lesson in AI development: sometimes the path to building something remarkable starts with understanding what AI can’t do.

Understanding these limitations doesn’t diminish agent technology — it makes it genuinely useful. And that’s the difference between a demo that crashes and a solution that deliver results.
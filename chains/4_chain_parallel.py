from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatOpenAI(model="gpt-4o")

main_summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie cretic."),
        ("human", "Give me a summary of the {movie_name}.")
    ]
)

# def analyze_plot(plot):
#     analyze_plot_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a movie cretic."),
#             ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?")
#         ]
#     )
#     return analyze_plot_template.format_prompt(plot=plot)

analyze_plot_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie cretic."),
        ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?")
    ]
)


# def analyze_character(character):
#     analyze_character_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a movie cretic."),
#             ("human", "Analyze the chracter of the {character}. What are its strengths and weaknesses?")
#         ]
#     )
#     return analyze_character_template.format_prompt(character=character)

analyze_character_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie cretic."),
        ("human", "Analyze the chracter of the {character}. What are its strengths and weaknesses?")
    ]
)

prepare_main_summary_template = RunnableLambda(lambda x: {"movie_name": "John Wick 1"})
# prepare_plot_summary_template = RunnableLambda(lambda x: analyze_plot(x))
prepare_plot_summary_template = RunnableLambda(lambda x: analyze_plot_template.format_prompt(plot=x))
# prepare_character_summary_template = RunnableLambda(lambda x: analyze_character(x))
prepare_character_summary_template = RunnableLambda(lambda x: analyze_character_template.format_prompt(character=x))

chain = (
    prepare_main_summary_template
    | main_summary_template
    | llm_model
    | StrOutputParser()
    | RunnableParallel(branches={
            "plot": prepare_plot_summary_template | llm_model | StrOutputParser(),
            "character": prepare_character_summary_template | llm_model | StrOutputParser()
        })
    | RunnableLambda(lambda x: {
        f"Plot: {x['branches']['plot']}\nCharacter: {x['branches']['character']}"
        })
)
print("")
print(chain.invoke({}))
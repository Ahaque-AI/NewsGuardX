from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from src.causalitynet.tools import custom_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class CausalCrew:
    """Causal Crew"""

    agents_config = "config/causal_agents.yaml"
    tasks_config = "config/causal_tasks.yaml"

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["retriever_agent"],
            llm=LLM(
            model="groq/llama-3.3-70b-specdec",
            temperature=0.8,
            ),
            tools=[custom_tool.hybrid_search],
            allow_delegation=False,
        )
    
    @agent
    def tagger_agent(self) -> Agent:
        return(
            Agent(
                config=self.agents_config["tagger_agent"],
                llm=LLM(
                model="nvidia_nim/meta/llama-3.1-405b-instruct",
                ),
                tools=[],
                verbose=True,
                memory=True,  
            )
        )
    
    @task
    def use_retriever(self) -> Task:
        return Task(
            config=self.tasks_config["use_retriever"],
        )
    
    @task 
    def use_tagger(self) -> Task:
        return Task(
            config=self.tasks_config["use_tagger"],
            context=[self.use_retriever()],  
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Causal Crew"""

        return Crew(
            agents=self.agents,  
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True
        )

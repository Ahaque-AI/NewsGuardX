from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from causalitynet.tools import custom_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class FactCheckingCrew:
    """FactChecking Crew"""

    agents_config = "config/facts_agents.yaml"
    tasks_config = "config/facts_tasks.yaml"

    @agent
    def fact_checker_agent_v1(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_checker_agent_v1"],
            llm=LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.8,
            ),
            tools=[custom_tool.search_internet],
            allow_delegation=False,
            max_iter=1,
        )
    
    @agent
    def fact_checker_agent_v2(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_checker_agent_v2"],
            llm=LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.8,
            ),
            tools=[custom_tool.search_tavily],
            allow_delegation=False,
            max_iter=1,
        )
    
    @agent
    def source_credibility_agent(self) -> Agent:
        return(
            Agent(
                config=self.agents_config["source_credibility_agent"],
                llm=LLM(
                model="gemini/gemini-2.0-flash",
                ),
                tools=[],
                verbose=True,
                memory=True,  
            )
        )
    
    @task
    def use_fact_checker_v1(self) -> Task:
        return Task(
            config=self.tasks_config["use_fact_checker_v1"],
        )
    
    @task
    def use_fact_checker_v2(self) -> Task:
        return Task(
            config=self.tasks_config["use_fact_checker_v2"],
        )
    
    @task
    def use_source_credibility(self) -> Task:
        return Task(
            config=self.tasks_config["use_source_credibility"],
            context=[self.use_fact_checker_v1(), self.use_fact_checker_v2()],  
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FactChecking Crew"""

        return Crew(
            agents=self.agents,  
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True
        )

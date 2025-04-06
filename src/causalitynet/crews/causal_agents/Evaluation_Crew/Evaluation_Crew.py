from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from causalitynet.tools import custom_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class EvaluationCrew:
    """Evaluation Crew"""

    agents_config = "config/evaluation_agents.yaml"
    tasks_config = "config/evaluation_tasks.yaml"

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["evaluation_agent"],
            llm=LLM(
                model="nvidia_nim/deepseek-ai/deepseek-r1",
                temperature=0.0,
            ),
            verbose=True,
            memory=True,
        )
    
    @task
    def evaluate_checking_overall(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_checking_overall"],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Evaluation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
    
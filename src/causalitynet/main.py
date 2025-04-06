#!/usr/bin/env python

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, and_, router

from .crews.causal_agents.FactChecking_Crew.FactChecking_Crew import FactCheckingCrew
from .crews.causal_agents.Causal_Crew.Causal_Crew import CausalCrew
from .crews.causal_agents.Evaluation_Crew.Evaluation_Crew import EvaluationCrew


class CausalityNetState(BaseModel):
    headline: str = ""
    verification: str = ""
    causal: str = ""
    evaluation: dict = {}
    result: str = ""


class CausalFlow(Flow[CausalityNetState]):

    @start()
    def Headline_Input(self):
        self.state.headline = input("Enter the headline: ")

    @listen(Headline_Input)
    def News_Detector_Crew(self):
        print("Fake News Detector:")
        result = (
            FactCheckingCrew()
            .crew()
            .kickoff(inputs={"headline": self.state.headline})
        )
        self.state.verification = result.raw

    @listen(Headline_Input)
    def Causal_Tagger_Crew(self):
        print("Causal Tagger:")
        result = (
            CausalCrew()
            .crew()
            .kickoff(inputs={"headline": self.state.headline})
        )
        self.state.causal = result.raw

    @listen(and_(News_Detector_Crew, Causal_Tagger_Crew))
    def Evaluation_Crew(self):
        print("Evaluation:")
        eval_result = (
            EvaluationCrew()
            .crew()
            .kickoff(inputs={
                "verification": self.state.verification,
                "causal": self.state.causal,
                "headline": self.state.headline
            })
        )
        self.state.evaluation = eval_result.raw

    @router(Evaluation_Crew)  
    def Route_Back_If_Scores_Low(self):
        
        # Check if the evaluation output indicates failure for fact checking or causal tagging.
        eval_data = self.state.evaluation
        recheck_fact = "fact_checker_failed" in str(eval_data)
        recheck_causal = "causal_tagger_failed" in str(eval_data)
        
        if recheck_fact:
            print("Re-running Fact Checker due to low evaluation score...")
            self.Fact_Checker()
        if recheck_causal:
            print("Re-running Causal Tagger due to low evaluation score...")
            self.Causal_Tagger()
        
        # After re-running if necessary, combine outputs into final result.
        self.state.result = f"Final outputs:\nFact Checking: {self.state.verification}\nCausal Tagging: {self.state.causal}"
        print("Final Evaluation Complete.")

    @listen(Route_Back_If_Scores_Low)
    def Final_Output(self):
        print("Final Output:")
        print(f"Verification: {self.state.verification}")
        print(f"Causal: {self.state.causal}")
        print(f"Evaluation: {self.state.evaluation}")
        print(f"Result: {self.state.result}")

def kickoff():
    causal_flow = CausalFlow()
    causal_flow.kickoff({"input": "Hello, this is a test run of CausalityNet."})

def plot():
    causal_flow = CausalFlow()
    causal_flow.plot()
    print("Plot saved")


if __name__ == "__main__":
    plot()
    kickoff()

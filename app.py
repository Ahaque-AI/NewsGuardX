#!/usr/bin/env python

import asyncio
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Generator

from crewai.flow import Flow, listen, start, and_, router
from src.causalitynet.crews.causal_agents.FactChecking_Crew.FactChecking_Crew import FactCheckingCrew
from src.causalitynet.crews.causal_agents.Causal_Crew.Causal_Crew import CausalCrew
from src.causalitynet.crews.causal_agents.Evaluation_Crew.Evaluation_Crew import EvaluationCrew

app = FastAPI()

class CausalityNetState(BaseModel):
    headline: str = ""
    verification: str = ""
    causal: str = ""
    evaluation: dict = {}
    result: str = ""

class CausalFlow(Flow[CausalityNetState]):
    def __init__(self, streamer):
        super().__init__()
        self.streamer = streamer

    @start()
    async def Headline_Input(self):
        # Headline is already set externally.
        pass

    @listen(Headline_Input)
    async def News_Detector_Crew(self):
        self.streamer("Running Fake News Detector...\n")
        await asyncio.sleep(0.1)
        # Offload the synchronous call to a separate thread.
        result = await asyncio.to_thread(
            lambda: FactCheckingCrew().crew().kickoff(inputs={"headline": self.state.headline})
        )
        self.state.verification = result.raw
        self.streamer(f"Fake News Detection Result:\n{self.state.verification}\n")
        await asyncio.sleep(0.1)

    @listen(Headline_Input)
    async def Causal_Tagger_Crew(self):
        self.streamer("Running Causal Tagger...\n")
        await asyncio.sleep(0.3)
        result = await asyncio.to_thread(
            lambda: CausalCrew().crew().kickoff(inputs={"headline": self.state.headline})
        )
        self.state.causal = result.raw
        self.streamer(f"Causal Tagging Result:\n{self.state.causal}\n")
        await asyncio.sleep(0.3)

    @listen(and_(News_Detector_Crew, Causal_Tagger_Crew))
    async def Evaluation_Crew(self):
        self.streamer("Running Evaluation Crew...\n")
        await asyncio.sleep(0.1)
        eval_result = await asyncio.to_thread(
            lambda: EvaluationCrew().crew().kickoff(inputs={
                "verification": self.state.verification,
                "causal": self.state.causal,
                "headline": self.state.headline,
                "sentence": self.state.headline
            })
        )
        self.state.evaluation = eval_result.raw

    @router(Evaluation_Crew)
    async def Route_Back_If_Scores_Low(self):
        eval_data = self.state.evaluation
        recheck_fact = "fact_checker_failed" in str(eval_data)
        recheck_causal = "causal_tagger_failed" in str(eval_data)

        if recheck_fact:
            self.streamer("Re-running Fact Checker due to low evaluation score...\n")
            await asyncio.sleep(0.1)
            result = await asyncio.to_thread(
                lambda: FactCheckingCrew().crew().kickoff(inputs={"headline": self.state.headline})
            )
            self.state.verification = result.raw
            self.streamer(f"Updated Fact Checking:\n{self.state.verification}\n")
            await asyncio.sleep(0.1)

        if recheck_causal:
            self.streamer("Re-running Causal Tagger due to low evaluation score...\n")
            await asyncio.sleep(0.1)
            result = await asyncio.to_thread(
                lambda: CausalCrew().crew().kickoff(inputs={"headline": self.state.headline})
            )
            self.state.causal = result.raw
            self.streamer(f"Updated Causal Tagging:\n{self.state.causal}\n")
            await asyncio.sleep(0.1)

        self.state.result = (
            f"Final outputs:\nFact Checking: {self.state.verification}\n"
            f"Causal Tagging: {self.state.causal}\nEvaluation: {self.state.evaluation}"
        )
        self.streamer("Final Evaluation Complete.\n")
        await asyncio.sleep(0.1)

    @listen(Route_Back_If_Scores_Low)
    async def Final_Output(self):
        self.streamer("Returning Final Output...\n")
        await asyncio.sleep(0.1)
        self.streamer(self.state.result)
        await asyncio.sleep(0.1)

@app.post("/run/{headline}")
async def run_flow(headline: str):
    async def stream_generator() -> Generator[str, None, None]:
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Thread-safe streamer that schedules messages on the event loop.
        def streamer(msg: str):
            loop.call_soon_threadsafe(queue.put_nowait, msg)

        flow = CausalFlow(streamer)
        flow.state.headline = headline

        async def run_flow_logic():
            # Run kickoff in a separate thread to avoid asyncio.run() conflicts.
            await asyncio.to_thread(flow.kickoff)
            await queue.put(None)  # Signal end of stream

        asyncio.create_task(run_flow_logic())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(stream_generator(), media_type="text/plain")

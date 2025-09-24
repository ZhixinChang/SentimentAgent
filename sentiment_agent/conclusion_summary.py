from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


class ConclusionSummaryAgent:
    def __init__(self, base_url, api_key, model, domain, question_type):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.domain = domain
        self.question_type = question_type
        self.model_client = OpenAIChatCompletionClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=0,
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": False,
                "family": ModelFamily.R1,
                "structured_output": True,
            },
        )
        self.conclusion_summary_agent = AssistantAgent(
            name="conclusion_summary_agent",
            model_client=self.model_client,
            system_message="""你是一个{domain}领域{question_type}问题总结的专家，给定{question_type}问题和文本内容，请总结用户在{domain}领域主要存在的{question_type}问题，确保按{question_type}问题的占比进行重要性降序排列，输出结果需要包含三类信息：{question_type}问题、数据占比和具体问题描述。
                                    """.format(domain=self.domain, question_type=self.question_type),
        )

    async def batch_run(self, classified_summary):
        self.conclusion_summary = classified_summary.copy()

        task_message = "\n".join((
                "{question_type}问题为：".format(question_type=self.question_type)
                + self.conclusion_summary["class"]
                + "此问题的数据占比"
                + (self.conclusion_summary["percentage"].map(lambda x: round(x * 100, 2))).astype(
            "str"
        )
                + "%，"
                + "{question_type}问题的具体描述为：".format(question_type=self.question_type)
                + self.conclusion_summary["推理原因"]
        ))

        result = await self.conclusion_summary_agent.run(task=task_message)
        self.messages = result.messages

        return self.messages[-1].content

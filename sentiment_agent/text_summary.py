import re

import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tqdm.asyncio import tqdm_asyncio

from .utils import get_classified_summary


class TextSummaryAgent:
    def __init__(self, base_url, api_key, model):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
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
        self.text_summary_agent = AssistantAgent(
            name="text_summary_agent",
            model_client=self.model_client,
            system_message="""你是一个体验问题分析的专家，给定体验问题和文本内容，请推理用户在生活服务方面存在该体验问题背后的原因，输出结果为推理原因，其中推理原因为分条列点回答对应的判断依据。
                                    回答模板格式如下：推理原因：1.yy,2.yy,3.yy,...,N.yy
                                    其中yy表示对应的判断依据。""",
        )

    async def batch_run(self, df, input_col):
        self.classified_df, self.classified_summary = get_classified_summary(df, input_col)

        start_tag = self.classified_summary[self.classified_summary['推理原因'] == ''].index.min()
        end_tag = self.classified_summary[self.classified_summary['推理原因'] == ''].index.max() + 1
        if not np.isnan(start_tag):
            index_list = list(range(start_tag, end_tag))

        pbar = tqdm_asyncio(total=len(self.classified_summary[self.classified_summary['推理原因'] == '']),
                            desc="补全进度")
        for i in index_list:
            await self.run(self.classified_df, self.classified_summary, i, input_col)
            pbar.update(1)

        pbar.close()
        print("所有项目处理完成")

        return self.classified_summary

    async def run(self, df, df_summary, i, input_col):
        self.classified_df = df.copy()
        self.classified_summary = df_summary.copy()
        error_answer = True
        omit_prompt = ''

        while error_answer:
            if (
                    len(
                        self.classified_df[
                            self.classified_df[input_col]
                            == self.classified_summary.loc[i, input_col]
                        ]
                    )
                    > 500
            ):
                task_message = (
                        "体验问题为："
                        + self.classified_summary.loc[i, input_col]
                        + "。"
                        + "\n相关文本内容为："
                        + "\n".join(
                    self.classified_df[
                        self.classified_df[input_col]
                        == self.classified_summary.loc[i, input_col]
                        ]["prompt"].sample(n=500, random_state=1, replace=False)
                )
                )
                print("第{}类分析，文本量过大，已抽样500条".format(i))
            else:
                task_message = (
                        "体验问题为："
                        + self.classified_summary.loc[i, input_col]
                        + "。"
                        + "\n相关文本内容为："
                        + "\n".join(
                    self.classified_df[
                        self.classified_df[input_col]
                        == self.classified_summary.loc[i, input_col]
                        ]["prompt"]
                )
                )
            result = await self.text_summary_agent.run(task='\n\n'.join([omit_prompt, task_message]))

            response = result.messages[-1].content

            if "推理原因：" in response:
                error_answer = False

            elif "推理原因：" not in response:
                omit_prompt = '''输出结果相比回答模板格式存在遗漏，请检查确保回答内容完整。
                        回答模板格式如下：推理原因：1.yy,2.yy,3.yy,...,N.yy
                        其中yy表示对应的判断依据。"""
                        请重新输出结果，确保输出完整内容，且结果格式与模板格式一致。'''
                print("第{}类分析存在回答遗漏错误，将重新进行分析".format(i + 1))
                continue

        answer = re.findall(r"推理原因：(.*)", response, re.S)
        self.classified_summary.loc[i, "推理原因"] = answer[0].strip()

        return self.classified_summary

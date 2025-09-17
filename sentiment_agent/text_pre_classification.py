import re

import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tqdm.asyncio import tqdm_asyncio

from .utils import pre_classified_fit


class TextPreClassificationAgent:
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
        self.text_pre_classification_agent = AssistantAgent(
            name="text_pre_classified_agent",
            model_client=self.model_client,
            system_message="""你是一个{domain}领域{question_type}问题分析的专家，给定文本内容，请推理用户在{domain}领域可能存在哪些{question_type}问题，输出结果为{question_type}问题和推理原因，其中{question_type}问题为分条列点的简要词汇概括，推理原因为分条列点回答对应的判断依据。
                            回答模板格式如下：{question_type}问题：1.xx<sep0>2.xx<sep0>3.xx<sep0>...<sep0>N.xx<sep1>推理原因：1.yy<sep0>2.yy<sep0>3.yy<sep0>...<sep0>N.yy
                            其中xx表示简要词汇概括，yy表示对应的判断依据，<sep0>和<sep1>为分隔符。""".format(domain=self.domain, question_type=self.question_type),
        )

    async def batch_run(self, df, input_col, pre_cluster_num=20, score_col=None):

        self.pre_classified_df, self.pre_classified_summary = pre_classified_fit(df, input_col, pre_cluster_num, score_col)
        start_tag = self.pre_classified_summary[self.pre_classified_summary['体验问题'] == ''].index.min()
        end_tag = self.pre_classified_summary[self.pre_classified_summary['体验问题'] == ''].index.max() + 1
        if not np.isnan(start_tag):
            index_list = list(range(start_tag, end_tag))

        pbar = tqdm_asyncio(total=len(self.pre_classified_summary[self.pre_classified_summary['体验问题'] == '']),
                            desc="补全进度")
        for cluster in index_list:
            self.pre_classified_summary = await self.run(self.pre_classified_df, self.pre_classified_summary, cluster,
                                                         input_col)
            pbar.update(1)

        pbar.close()
        print("所有项目处理完成")

        return self.pre_classified_summary

    async def run(self, df, df_summary, cluster, input_col):

        self.pre_classified_df = df.copy()
        self.pre_classified_summary = df_summary.copy()

        error_answer = True
        omit_prompt = ''
        sep_prompt = ''
        num_prompt = ''

        while error_answer:

            if len(self.pre_classified_df[self.pre_classified_df['cluster'] == cluster][input_col]) > 100:
                print("第{}类分析，文本量过大，已抽样100条".format(cluster))
                task_message = "\n".join(
                    self.pre_classified_df[self.pre_classified_df['cluster'] == cluster][input_col].sample(n=100,
                                                                                                           random_state=1,
                                                                                                           replace=False))
            else:
                task_message = "\n".join(
                    self.pre_classified_df[self.pre_classified_df['cluster'] == cluster][input_col])

            result = await self.text_pre_classification_agent.run(
                task='\n\n'.join([omit_prompt, sep_prompt, num_prompt, task_message]))

            response = result.messages[-1].content

            if '体验问题' not in response or '推理原因' not in response:
                omit_prompt = '''输出结果相比回答模板格式存在遗漏，请检查确保回答内容完整。
                            回答模板格式如下：体验问题：1.xx<sep0>2.xx<sep0>3.xx<sep0>...<sep0>N.xx<sep1>推理原因：1.yy<sep0>2.yy<sep0>3.yy<sep0>...<sep0>N.yy
                            其中xx表示简要词汇概括，yy表示对应的判断依据，<sep0>和<sep1>为分隔符。
                            请重新输出结果，确保输出完整内容，且结果格式与模板格式一致。'''
                print("第{}类分析存在回答遗漏错误，将重新进行分析".format(cluster))
                continue

            if len(response.strip().strip('<sep1>').split('<sep1>')) != 2:
                sep_prompt = '''输出结果的分隔符与回答模板格式不一致，请检查确保分隔符<sep2>正确。
                            回答模板格式如下：体验问题：1.xx<sep0>2.xx<sep0>3.xx<sep0>...<sep0>N.xx<sep1>推理原因：1.yy<sep0>2.yy<sep0>3.yy<sep0>...<sep0>N.yy
                            其中xx表示简要词汇概括，yy表示对应的判断依据，<sep0>和<sep1>为分隔符。
                            请重新输出结果，确保输出结果的分隔符与模板格式一致。'''
                print("第{}类分析存在<sep1>分隔符错误，将重新进行分析".format(cluster))
                continue

            if len(response.strip().strip('<sep1>').split('<sep1>')[0].split('<sep0>')) != len(
                    response.strip().strip('<sep1>').split('<sep1>')[1].split('<sep0>')):
                num_prompt = '''输出结果的体验问题和推理原因的分条列点数量不一致，请检查确保分隔符<sep1>正确，
                            回答模板格式如下：体验问题：1.xx<sep0>2.xx<sep0>3.xx<sep0>...<sep0>N.xx<sep1>推理原因：1.yy<sep0>2.yy<sep0>3.yy<sep0>...<sep0>N.yy
                            其中xx表示简要词汇概括，yy表示对应的判断依据，<sep0>和<sep1>为分隔符。
                            请重新输出结果，确保输出体验问题和推理原因的分条列点数量一致。'''
                print("第{}类分析存在数据量不一致或<sep0>分隔符错误，将重新进行分析".format(cluster))
                continue

            else:
                error_answer = False

        answer = re.findall(r'体验问题：(.*)<sep1>.*推理原因：(.*)', response, re.S)
        self.pre_classified_summary.loc[self.pre_classified_summary['cluster'] == cluster, '体验问题'] = ','.join(
            answer[0][0].split('<sep0>'))
        self.pre_classified_summary.loc[self.pre_classified_summary['cluster'] == cluster, '推理原因'] = ','.join(
            answer[0][1].split('<sep0>'))

        return self.pre_classified_summary

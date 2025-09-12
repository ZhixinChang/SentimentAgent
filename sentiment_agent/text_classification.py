import re

import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tqdm.asyncio import tqdm_asyncio


class TextClassificationAgent:
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
        self.summary2label_agent = AssistantAgent(
            name="summary2label_agent",
            model_client=self.model_client,
            system_message="""你是一个体验问题总结的专家，给定文本内容，请推理用户在生活服务方面可能存在哪些体验问题，请将相似问题进行合并分类，避免重复回答，输出结果为体验问题，请分条列点并用简要词汇概括。
                                                            回答模板格式如下：体验问题：1.xx<sep>2.xx<sep>3.xx<sep>...<sep>N.xx""",
        )


    async def summary2label(self, df_summary):

        self.pre_classified_summary = df_summary.copy()
        task_message = "\n".join((
                "体验问题为："
                + self.pre_classified_summary["体验问题"]
                + "推理原因为："
                + self.pre_classified_summary["推理原因"]
        ))
        result = await self.summary2label_agent.run(task=task_message)

        response = result.messages[-1].content
        print(','.join(response.split('<sep>')))
        self.class_labels = re.findall(r'\d{1,2}.(\w+)', ','.join(response.split('<sep>')), re.S) + ['其他问题']

        self.text_classified_agent = AssistantAgent(
            name="text_classified_agent",
            model_client=self.model_client,
            system_message="""你是一个体验问题分类的专家，给定多条文本内容，每条文本内容之间以符号<sep>作为分割符，已知体验问题标签集合为{}，请对每条文本内容分别判断所属的体验问题标签，且每条文本内容只能属于集合中的一个标签，输出结果为体验问题分类，请确保体验问题分类属于已知体验类别集合。
                                回答模板格式如下：xx<sep>xx<sep>xx<sep>...<sep>xx
                                其中xx表示体验问题标签，<sep>为分隔符。""".format(set(self.class_labels)),
        )

    async def batch_run(self, df, batch_size=30, input_col: str = 'content',
                        output_col: str = 'class'):

        self.classified_df = df.copy()
        start_tag = self.classified_df[self.classified_df[output_col] == ''].index.min()
        end_tag = self.classified_df[self.classified_df[output_col] == ''].index.max() + 1

        if not np.isnan(start_tag):
            index_list = list(range(start_tag, end_tag, batch_size))
            index_list.append(end_tag)

        pbar = tqdm_asyncio(total=len(self.classified_df[self.classified_df[output_col] == '']),
                            desc="补全进度")
        for i in range(len(index_list) - 1):
            start_index = index_list[i]
            end_index = index_list[i + 1]
            await self.run(self.classified_df, start_index, end_index, input_col,
                           output_col)
            pbar.update(end_index - start_index)

        pbar.close()
        print("所有项目处理完成")

        return self.classified_df

    async def run(self, df, start_index, end_index, input_col,
                  output_col):
        self.classified_df = df.copy()
        error_answer = True
        num_prompt = ''
        label_prompt = ''

        while error_answer:

            task_message = "<sep>".join(self.classified_df[input_col].iloc[start_index: end_index])

            result = await self.text_classified_agent.run(
                task='\n\n'.join([num_prompt, label_prompt, task_message]))

            response = result.messages[-1].content

            if len(response.strip().strip('<sep>').split('<sep>')) != end_index - start_index:
                num_prompt = '''输出结果数量与给定文本内容数量不一致，请检查确保分割符划分正确。
                    每条文本内容之间的分隔符为<sep>，输出结果的分隔符为<sep>，
                    请重新输出体验问题标签，确保输出结果数量与给定文本内容数量一致。'''
                print("第{}-{}行分析存在输入与输出数量不一致错误，将重新进行分析".format(start_index, end_index - 1))
                continue

            elif len(response.strip().strip('<sep>').split('<sep>')) == end_index - start_index:
                for label in response.strip().strip('<sep>').split('<sep>'):
                    if label not in self.class_labels:
                        error_answer = True
                        label_prompt = '''输出结果标签不属于给定标签集合，请检查确保标签属于给定标签集合。
                    每条文本内容之间的分隔符为<sep>，输出结果的分隔符为<sep>，
                    请重新输出体验问题标签，确保标签属于给定的标签集合。'''
                        print("第{}-{}行分析存在标签不属于给定的标签集合，将重新进行分析".format(start_index,
                                                                                                end_index - 1))
                        print("标签为：{}，整体返回结果为{}".format(label, response))
                        break
                    else:
                        error_answer = False

        self.classified_df.loc[start_index: end_index - 1, output_col] = response.strip().strip(
            '<sep>').split(
            '<sep>')

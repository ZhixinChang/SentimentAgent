import re

import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tqdm.asyncio import tqdm_asyncio


class SentimentAnalysisAgent:
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
        self.sentiment_analysis_agent = AssistantAgent(
            name="sentiment_analysis_agent",
            model_client=self.model_client,
            system_message="""你是一个{domain}领域的情感分析的专家，给定多条文本内容，每条文本内容之间以符号<sep>作为分割符，请对每条文本内容分别判断用户在{domain}领域对{question_type}问题的情感倾向，输出结果为情感分，取值范围为[0,10]，其中[0,3]为负向、(3,6]为中性、(6,10]为正向。
                                                                    回答模板格式如下：xx<sep>xx<sep>xx<sep>...<sep>xx
                                                                    其中xx表示取值范围为[0,10]的情感分，<sep>为分隔符。""".format(domain=self.domain, question_type=self.question_type), )

    async def batch_run(self, sentiment_df, batch_size=30, content_col: str = 'content',
                        sentiment_col: str = 'score'):

        self.sentiment_df = sentiment_df.copy()
        start_tag = self.sentiment_df[self.sentiment_df[sentiment_col] == ''].index.min()
        end_tag = self.sentiment_df[self.sentiment_df[sentiment_col] == ''].index.max() + 1

        if not np.isnan(start_tag):
            index_list = list(range(start_tag, end_tag, batch_size))
            index_list.append(end_tag)

        pbar = tqdm_asyncio(total=len(self.sentiment_df[self.sentiment_df[sentiment_col] == '']),
                            desc="sentiment analysis complete progress")
        for i in range(len(index_list) - 1):
            start_index = index_list[i]
            end_index = index_list[i + 1]
            self.sentiment_df = await self.run(self.sentiment_df, start_index, end_index, content_col,
                                               sentiment_col)
            pbar.update(end_index - start_index)

        pbar.close()
        print("sentiment analysis completion completed")

        return self.sentiment_df

    async def run(self, sentiment_df, start_index, end_index, content_col: str = 'content',
                  sentiment_col: str = 'score'):
        self.sentiment_df = sentiment_df.copy()
        error_answer = True
        num_prompt = ''
        dtype_prompt = ''

        task_message = "<sep>".join(self.sentiment_df[content_col].iloc[start_index: end_index])
        while error_answer:

            result = await self.sentiment_analysis_agent.run(task='\n\n'.join([num_prompt, dtype_prompt, task_message]))

            response = result.messages[-1].content

            input_num = end_index - start_index
            output_num = len(response.strip().strip('<sep>').split('<sep>'))
            if output_num != input_num:
                num_prompt = '''输出结果数量为{}，与输入文本内容数量{}不一致，请检查确保分割符划分正确。
                                    每条文本内容之间的分隔符为<sep>，
                                    请重新输出情感分，确保输出结果数量等于给定文本内容数量。'''.format(output_num,
                                                                                                     input_num)
                print(
                    "第{}-{}行的分析存在输出数量{}与输入数量{}不一致错误，将重新进行分析，若多次失败，建议减少batch_size的值以提升模型性能。".format(
                        start_index, end_index, output_num, input_num))
                print(response)
                continue


            elif not bool(re.fullmatch(r'^([0-9]<sep>|10<sep>){' + r'{}'.format(input_num - 1) + r'}([0-9]|10)$',
                                       response.strip().strip('<sep>'))):
                dtype_prompt = '''输出结果数据类型与给定回答模板格式不一致，请检查确保数据类型正确，
                                    回答模板格式如下：xx<sep>xx<sep>xx<sep>...<sep>xx
                                    其中xx表示取值范围为[0,10]的情感分。
                                    请重新输出情感分，确保输出数据类型与模板格式一致。'''
                print(
                    "第{}-{}行的分析存在数据格式错误，将重新进行分析，若多次失败，建议减少batch_size的值以提升模型性能。".format(
                        start_index, end_index))
                print(response)
                continue

            else:
                error_answer = False

        self.sentiment_df.loc[start_index: end_index - 1, sentiment_col] = response.strip().strip('<sep>').split(
            '<sep>')

        return self.sentiment_df
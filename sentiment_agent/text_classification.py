import re

import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tqdm.asyncio import tqdm_asyncio


class TextClassificationAgent:
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
        self.summary2label_agent = AssistantAgent(
            name="summary2label_agent",
            model_client=self.model_client,

            system_message="""你是一个{question_type}问题总结的专家，给定文本内容，请推理用户在{domain}领域可能存在哪些{question_type}问题，请将相似问题进行合并分类，避免重复回答，输出结果为{question_type}问题，请分条列点并用简要词汇概括。
                                                            回答模板格式如下：{question_type}问题：1.xx<sep>2.xx<sep>3.xx<sep>...<sep>N.xx
                                                            其中xx表示{question_type}问题，<sep>为分隔符。""".format(
                domain=self.domain, question_type=self.question_type),
        )

    async def summary2label(self, pre_classified_summary):

        self.pre_classified_summary = pre_classified_summary.copy()
        error_answer = True
        human_prompt = ''

        while error_answer:
            task_message = "\n".join((
                    "体验问题为："
                    + self.pre_classified_summary["体验问题"]
                    + "推理原因为："
                    + self.pre_classified_summary["推理原因"]
            ))
            result = await self.summary2label_agent.run(task='\n\n'.join([human_prompt, task_message]))

            response = result.messages[-1].content
            print('Chain of thought:', result.messages[1].content)
            print(f'{self.question_type}问题:', ','.join(response.split('<sep>')))
            human_feedback = input(
                '''请对大模型总结的问题标签提供反馈，\n如果赞同请回复[y]。\n如果不赞同并需要大模型重新总结请回复[n]并提供提示词，请以如下格式回复：\nn<sep>xxxx\n其中xxxx表示提示词，<sep>为分隔符。\n如果您只需要修改大模型总结的部分结论，请以如下格式回复：\n1.xx<sep>2.xx<sep>3.xx<sep>...<sep>N.xx\n其中xx表示{question_type}问题，<sep>为分隔符。\n请输入你的反馈：'''.format(
                    question_type=self.question_type))
            if human_feedback == 'y':
                self.class_labels = re.findall(r'\d{1,}.(\w+)',
                                               ','.join(response.strip().strip('<sep>').split('<sep>')), re.S) + [
                                        '其他问题']
                error_answer = False


            elif human_feedback.startswith('n'):
                human_prompt = human_feedback.split('<sep>')[-1]

            elif bool(re.fullmatch(r'(\d{1,}.\w+,){' + r'{}'.format(
                    len(human_feedback.strip().strip('<sep>').split('<sep>')) - 1) + r'}(\d{1,}.\w+)',
                                   ','.join(human_feedback.strip().strip('<sep>').split('<sep>')))):
                custom_labels = re.findall(r'\d{1,}.(\w+)',
                                           ','.join(human_feedback.strip().strip('<sep>').split('<sep>')), re.S)
                self.class_labels = custom_labels + ['其他问题']
                error_answer = False

            else:
                print('回答内容无效，将重新进行问题总结。')

        self.text_classification_agent = AssistantAgent(
            name="text_classified_agent",
            model_client=self.model_client,
            system_message="""你是一个{question_type}问题分类的专家，给定多条文本内容，每条文本内容之间以符号<sep>作为分割符，已知{question_type}问题标签集合为{class_labels}，请对每条文本内容分别判断所属的问题标签，输出结果为问题标签，且每条文本内容只能属于集合中的一个标签。
                                回答模板格式如下：xx<sep>xx<sep>xx<sep>...<sep>xx
                                其中xx表示体验问题标签，<sep>为分隔符。""".format(question_type=self.question_type,
                                                                                class_labels=set(self.class_labels)),
        )

    async def batch_run(self, pre_classified_df, batch_size=30, content_col: str = 'content',
                        classified_col: str = 'class'):

        self.classified_df = pre_classified_df.copy()
        start_tag = self.classified_df[self.classified_df[classified_col] == ''].index.min()
        end_tag = self.classified_df[self.classified_df[classified_col] == ''].index.max() + 1

        if not np.isnan(start_tag):
            index_list = list(range(start_tag, end_tag, batch_size))
            index_list.append(end_tag)

        pbar = tqdm_asyncio(total=len(self.classified_df[self.classified_df[classified_col] == '']),
                            desc="text classification complete progress")
        for i in range(len(index_list) - 1):
            start_index = index_list[i]
            end_index = index_list[i + 1]
            await self.run(self.classified_df, start_index, end_index, content_col,
                           classified_col)
            pbar.update(end_index - start_index)

        pbar.close()
        print("text classification completion completed")

        return self.classified_df

    async def run(self, classified_df, start_index, end_index, content_col,
                  classified_col):
        self.classified_df = classified_df.copy()
        error_answer = True
        num_prompt = ''
        label_prompt = ''

        while error_answer:

            task_message = "<sep>".join(self.classified_df[content_col].iloc[start_index: end_index])

            result = await self.text_classification_agent.run(
                task='\n\n'.join([num_prompt, label_prompt, task_message]))

            response = result.messages[-1].content

            if len(response.strip().strip('<sep>').split('<sep>')) != end_index - start_index:
                num_prompt = '''输出结果数量与给定文本内容数量不一致，请检查确保分割符划分正确。
                    每条文本内容之间的分隔符为<sep>，输出结果的分隔符为<sep>，
                    请重新输出体验问题标签，确保输出结果数量与给定文本内容数量一致。'''
                print("第{}-{}行分析存在输入与输出数量不一致错误，将重新进行分析".format(start_index, end_index - 1))
                print("整体返回结果为{}".format(response))
                continue

            elif len(response.strip().strip('<sep>').split('<sep>')) == end_index - start_index:
                error_answer_num = 0
                for label in response.strip().strip('<sep>').split('<sep>'):
                    if error_answer_num > 0:
                        break
                    else:
                        if label not in self.class_labels:
                            error_answer_num += 1
                            label_prompt = '''输出结果标签不属于给定标签集合，请检查确保标签属于给定标签集合。
                        每条文本内容之间的分隔符为<sep>，输出结果的分隔符为<sep>，
                        请重新输出体验问题标签，确保标签属于给定的标签集合。'''
                            print("第{}-{}行分析存在标签不属于给定的标签集合，将重新进行分析".format(start_index,
                                                                                                    end_index - 1))
                            print("标签为：{}，整体返回结果为{}".format(label, response))
                            break
                        else:
                            pass
                if error_answer_num == 0:
                    error_answer = False

        self.classified_df.loc[start_index: end_index - 1, classified_col] = response.strip().strip(
            '<sep>').split(
            '<sep>')

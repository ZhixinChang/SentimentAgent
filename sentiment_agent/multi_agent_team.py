from .conclusion_summary import ConclusionSummaryAgent
from .sentiment_analysis import SentimentAnalysisAgent
from .text_classification import TextClassificationAgent
from .text_pre_classification import TextPreClassificationAgent
from .text_summary import TextSummaryAgent


class SentimentMultiAgentTeam:
    def __init__(self, base_url, api_key, model, domain, question_type):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.domain = domain
        self.question_type = question_type
        self.multi_agent_team = {}
        self.multi_agent_team['SentimentAnalysisAgent'] = SentimentAnalysisAgent(base_url=self.base_url,
                                                                                 api_key=self.api_key,
                                                                                 model=self.model, domain=self.domain,
                                                                                 question_type=self.question_type)

        self.multi_agent_team['TextPreClassificationAgent'] = TextPreClassificationAgent(base_url=self.base_url,
                                                                                         api_key=self.api_key,
                                                                                         model=self.model,
                                                                                         domain=self.domain,
                                                                                         question_type=self.question_type)

        self.multi_agent_team['TextClassificationAgent'] = TextClassificationAgent(base_url=self.base_url,
                                                                                   api_key=self.api_key,
                                                                                   model=self.model, domain=self.domain,
                                                                                   question_type=self.question_type)

        self.multi_agent_team['TextSummaryAgent'] = TextSummaryAgent(base_url=self.base_url, api_key=self.api_key,
                                                                     model=self.model,
                                                                     domain=self.domain,
                                                                     question_type=self.question_type)
        self.multi_agent_team['ConclusionSummaryAgent'] = ConclusionSummaryAgent(base_url=self.base_url,
                                                                                 api_key=self.api_key,
                                                                                 model=self.model,
                                                                                 domain=self.domain,
                                                                                 question_type=self.question_type)

    async def batch_run(self, df, batch_size=30, content_col='content', sentiment_col='score', pre_cluster_num=20,
                        classified_col='class'):
        self.sentiment_df = df.copy()
        self.sentiment_df[sentiment_col] = ''

        print('SentimentAnalysisAgent starts working...')
        self.sentiment_df = await self.multi_agent_team['SentimentAnalysisAgent'].batch_run(
            sentiment_df=self.sentiment_df,
            batch_size=batch_size,
            content_col=content_col,
            sentiment_col=sentiment_col)
        self.negative_sentiment_df = self.sentiment_df[
            self.sentiment_df[sentiment_col].astype('float64') <= 3].reset_index().copy()

        print('TextPreClassificationAgent starts working...')
        self.pre_classified_summary = await self.multi_agent_team['TextPreClassificationAgent'].batch_run(
            negative_sentiment_df=self.negative_sentiment_df, content_col=content_col, pre_cluster_num=pre_cluster_num)
        self.pre_classified_df = self.multi_agent_team['TextPreClassificationAgent'].pre_classified_df
        self.pre_classified_df[classified_col] = ''

        print('TextClassificationAgent starts working...')
        await self.multi_agent_team['TextClassificationAgent'].summary2label(self.pre_classified_summary)
        self.classified_df = await self.multi_agent_team['TextClassificationAgent'].batch_run(self.pre_classified_df,
                                                                                              batch_size=batch_size,
                                                                                              content_col=content_col,
                                                                                              classified_col=classified_col)

        print('TextSummary starts working...')
        self.classified_summary = await self.multi_agent_team['TextSummaryAgent'].batch_run(self.classified_df,
                                                                                            content_col, sentiment_col,
                                                                                            classified_col)
        print('ConclusionSummaryAgent starts working...')
        self.conclusion_summary = await self.multi_agent_team['ConclusionSummaryAgent'].batch_run(
            self.classified_summary)

        return [self.pre_classified_summary, self.classified_df, self.classified_summary, self.conclusion_summary]

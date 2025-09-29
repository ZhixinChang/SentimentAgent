# SentimentAgent
[![GitHub license][license-badge]][license-url]
[![Python Versions][python-badge]][pypi-url]

[license-badge]: https://img.shields.io/badge/license-Apache%202.0-green
[license-url]: https://github.com/ZhixinChang/SentimentAgent/blob/main/LICENSE

[python-badge]: https://img.shields.io/badge/python-3.13-blue
[pypi-url]: https://pypi.org/project/sentiment_agent/

A sentiment analysis agent based on user text, achieving quantitative summarization from user text to specific problems.

## What can this do for you?
SentimentAgent is specifically designed for data analysis, focusing on processing structured text datasets to achieve localization and quantification from user text to specific problems, providing data support for exploring users' actual needs.

### Features
- **Quantification of user emotions**. SentimentAgent has the ability of sentiment analysis, which can quantify user emotions in the form of data. The range of sentiment scores is [0,10], where [0,3] is negative, (3,6] is neutral, and (6,10] is positive.
  
- **Negative text problem tagging**. SentimentAgent is capable of summarizing specific problems in negative text and supporting humans to provide optimization feedback to LLM until specific problem labels are identified.

- **Quantify problem labels**. SentimentAgent can quantify the proportion of each problem in all negative texts for a given set of problem labels, and provide reasoning reasons for each problem.

- **Conclusion summary**. SentimentAgent can summarize all problem names, data proportions, and specific problem descriptions, and prioritize them according to their proportional sizes, simplifying the information processing cost of data analysis.

### Workflows
- **SentimentAnalysisAgent**: Specifically responsible for sentiment analysis of text and providing data quantification of sentiment scores.

- **TextPreClassificationAgent**: Specifically responsible for pre classifying text to reduce costs for formal text classification.

- **TextClassificationAgent**: Specifically responsible for formally classifying text, completing problem tagging and text annotation work.

- **TextSummaryAgent**: Specifically responsible for quantifying problem labels and providing reasoning reasons for each problem.

- **ConclusionSummaryAgent**: Specifically responsible for summarizing conclusions on issues and prioritizing them based on their proportion, providing issues, data proportions, and specific problem descriptions.

## Quick Start
Make sure you have followed the installation instructions.

### Example: texts to problems by SentimentMultiAgentTeam
Firstly, import the relevant libraries and methods

```python
from sentiment_agent import SentimentMultiAgentTeam
from sentiment_agent.utils import get_simulated_data_by_llm

base_url = "<your_base_url>"
api_key = "<your_api_key>"
model = "<your_model>"
```

To avoid issues with data privacy, we pre simulated and generated user feedback text data using LLM. And we can use a simple method to load the data for this example:

```python
domain='酒店'
question_type='居住质量'
content_col = 'content'

df = get_simulated_data_by_llm()
df.head()
```

Simple observation and understanding of textual data information.

```
	content
0	预订时显示的价格与到店后实际收费不一致
1	提前预订的房型到店后被告知无房需加价升级
2	线上承诺的优惠活动到店后无法兑现
3	预订时备注的特殊需求完全未被处理
4	酒店官网信息更新不及时导致行程受影响
```

Only two lines of code are needed to complete the analysis from text to problem!

```python
team = SentimentMultiAgentTeam(base_url=base_url,
                              api_key=api_key,
                              model=model,
                              domain=domain,
                              question_type=question_type)
result_list = await team.batch_run(df=df)
```

In the stage of clarifying problem labels, we can combine LLM's thought chain and the set of problem labels returned by LLM to provide human feedback until the results meet expectations. As an example of a quick start, we directly choose y to quickly pass through this stage:

```
Chain of thought: 
用户让我总结酒店领域的居住质量问题，需要把相似的问题合并分类，避免重复。首先，我需要仔细看看用户提供的所有体验问题和推理原因，然后找出共同点进行归类。

首先，用户分了几个部分，每个部分有不同的问题。比如第一部分提到儿童游乐区安全隐患、无障碍设施不完善、周边设施匮乏。这些属于安全设施和周边配套的问题。接下来是价格透明度、预订不符、特殊需求未处理等，这些属于服务预订和价格方面的问题。

然后有关于前台服务效率、态度，押金退还等，属于前台服务质量。房间隔音、卫生差、设备故障这些属于客房设施与卫生。餐饮方面有早餐单一、卫生问题、热菜品质，属于餐饮质量问题。还有客服响应慢、投诉处理低效，属于客户服务问题。

安全方面有消防通道堵塞、设施缺失，隐私侵犯，属于安全与隐私问题。无障碍设施、电梯无障碍缺失，属于无障碍设施问题。周边交通不便、治安隐患，属于周边环境问题。公共区域清洁差、电梯故障，属于公共区域维护。收费不透明的问题，比如停车、娱乐设施，属于收费透明度。

还有设备老化的问题，比如健身房、游戏室，属于设施设备老化。强制消费、服务效率低，属于服务规范性。噪音问题，比如外部施工、设备噪音，属于噪音干扰。床上用品、浴袍清洁，属于布草与客用品卫生。官网信息滞后、客服质量差，属于信息与客服服务。最后，像会议室设施故障、婚庆服务混乱，属于特殊服务问题。

需要确保每个类别不重复，用简要词汇概括。比如把儿童游乐区和泳池标识归到安全隐患，无障碍设施相关的合并，价格和收费相关的合并，前台服务相关的合并，客房卫生和设备合并，餐饮问题合并，客服和投诉处理合并，安全和隐私合并，周边环境包括交通和治安，公共区域维护包括清洁和电梯，设施老化单独一类，服务规范性包括强制消费和效率，噪音问题单独一类，布草卫生，信息客服，特殊服务如会议婚庆等。这样分条列点，应该覆盖所有问题，没有重复。

居住质量问题: 
居住质量问题：1.安全隐患,2.无障碍设施不完善,3.周边配套设施匮乏,4.价格与收费问题,5.预订与房型问题,6.特殊需求与服务问题,7.前台服务质量,8.客房设施与卫生,9.餐饮质量,10.客户服务,11.安全与隐私问题,12.周边环境,13.公共区域维护,14.设施设备老化,15.服务规范性,16.噪音干扰,17.布草与客用品卫生,18.信息与客服服务,19.特殊服务问题
请对大模型总结的问题标签提供反馈，
如果赞同请回复[y]。
如果不赞同并需要大模型重新总结请回复[n]并提供提示词，请以如下格式回复：
n<sep>xxxx
其中xxxx表示提示词，<sep>为分隔符。
如果您只需要修改大模型总结的部分结论，请以如下格式回复：
1.xx<sep>2.xx<sep>3.xx<sep>...<sep>N.xx
其中xx表示居住质量问题，<sep>为分隔符。
请输入你的反馈： y
```

We can directly obtain the conclusion summary compiled by LLM:

```python
print('\n'.join(result_list[-1].strip().split('\n')[: 3]))
```

```
1. 服务规范性，数据占比12.5%，具体问题描述：服务承诺管理机制失效；服务流程存在强制消费导向；财务流程设计不规范；餐饮服务流程执行混乱；产品宣传与实物标准脱节；财务合规管理意识淡薄；定制化服务执行体系缺失；跨部门协作机制不健全；运营信息透明度不足；服务时效与操作规范失控；洗衣服务流程管理粗放；会员体系运营混乱；续住服务沟通机制断裂；餐饮后勤保障效率低下。
2. 设施设备老化，数据占比11.67%，具体问题描述：设备维护保养制度缺失；设施设备采购决策短视；专项预算分配不合理；供应商管理机制失效；环境控制系统老化；维修响应流程低效；技术更新意识滞后；材质选用标准低下；管理责任划分模糊；重大活动保障机制缺位。
3. 客房设施与卫生，数据占比10.0%，具体问题描述：卫生间清洁流程不规范；淋浴设备老化未及时检修；马桶配件质量差或安装不规范；客房装修设计未调研现代用电需求；电视遥控器电池更换不及时；客房清洁质量管控失效；加床采购未遵循人体工学标准；浴室玻璃清洁工具配备不全；灯光设计未区分功能区域；窗帘材质遮光率低；热水供应系统容量匹配不合理；网络电视服务商合作协议内容受限。
```

We can also obtain intermediate results of agent reasoning to help us better understand the reasons behind the problem:

```python
result_list[2].head(3)
```

```
	class	count	percentage	score	推理原因
0	价格与收费问题	8	0.06666666666666667	1.625	1.预订定价系统或策略存在问题，存在用低价吸引客人、到店加价的隐性收费情况,2.停车收费未明确公示标准，利用地理位置优势抬高价格,3.加床服务未合理定价，可能因房间资源紧张或成本高导致费用高,4.熨烫服务未根据成本合理定价，且员工培训不足或设备不当致服务质量差,5.外币兑换与外部合作机构协议不合理，想通过附加服务赚高额利润,6.干洗服务选择的供应商成本高，且服务流程不严格、质量控制不到位,7.宠物寄存可能因酒店不鼓励携带宠物而故意设高费用，或设施投入不足,8.娱乐设施未在显眼位置公示收费标准，存在临时随意定价情况
1	信息与客服服务	1	0.008333333333333333	2.0	1.酒店内部信息管理流程不完善，未建立官网信息及时更新的责任机制与跨部门协作流程,2.官网技术维护不足，内容管理系统操作复杂或技术团队响应滞后导致信息更新延迟,3.客服团队未同步获取官网最新信息或培训不到位，无法对不准确信息进行提前预警与纠正,4.缺乏官网信息定期巡检制度，对信息时效性与准确性的监管力度不足
2	公共区域维护	11	0.09166666666666666	1.6363636363636365	1.清洁管理体系不完善，未制定合理清洁计划、明确清洁频率标准且清洁人员配置不足,2.网络基础设施建设与维护不到位，Wi-Fi设备老化或部署点位不合理且未定期检测信号,3.泳池维护专业度不足，水处理系统未及时清洗更换、消毒剂投放比例失衡且缺乏水质监测机制,4.员工行为规范管理松散，未划定清洁工具专属存放区域且未开展职业素养与操作流程培训,5.桑拿房环境控制系统缺陷，通风除湿设备故障或运行不足且装修材料防潮处理不合格,6.功能区域规划设计不科学，吸烟区与非吸烟区空间隔离及通风换气系统设置不合理,7.空调系统养护制度缺失，未建立定期清洗滤网与出风口的维护计划且监管检查不到位,8.公共设施采购决策失误，选用低成本劣质地毯、座椅材质且未评估实际使用耐久性,9.游戏厅环境管理机制失效，机械通风设备功率不足或未严格执行室内禁烟规定,10.公共区域功能需求调研滞后，图书馆等区域插座数量未匹配现代电子设备充电需求
```

If necessary, we also support obtaining the original dataset after problem annotation:

```python
result_list[1][['content', 'score', 'class']].head(3)
```

```
	content	score	class
0	预订时显示的价格与到店后实际收费不一致	1	价格与收费问题
1	提前预订的房型到店后被告知无房需加价升级	1	预订与房型问题
2	线上承诺的优惠活动到店后无法兑现	1	服务规范性

```

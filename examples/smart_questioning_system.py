"""基于 deepagents 的多智能体智能问数系统示例。

该示例遵循《智能问数系统说明》中描述的六阶段流程，展示如何使用
`create_deep_agent` 配置意图识别、上下文收集、SQL/接口执行等协同能力。
所有提示与工具说明均为中文，便于直接在中文业务场景中落地。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepagents import create_deep_agent
from langchain_core.tools import tool


@dataclass
class TableField:
    """数据库字段元数据。"""

    name: str
    dtype: str
    zh_name: str
    meaning: str
    example: str


@dataclass
class TableSchema:
    """数据库表结构信息。"""

    name: str
    zh_name: str
    usage: str
    primary_key: str
    indexes: List[str]
    fields: List[TableField] = field(default_factory=list)

    def summarize(self) -> str:
        lines = [f"表名: {self.name} ({self.zh_name})", f"用途: {self.usage}"]
        lines.append(f"主键: {self.primary_key}")
        if self.indexes:
            lines.append(f"常用索引: {', '.join(self.indexes)}")
        if self.fields:
            lines.append("字段列表:")
            for field_ in self.fields:
                lines.append(
                    f"- {field_.name} ({field_.zh_name}, {field_.dtype}): {field_.meaning} 示例: {field_.example}"
                )
        return "\n".join(lines)


@dataclass
class BusinessVocabulary:
    """业务术语词典，支持同义词检索。"""

    term: str
    canonical: str
    synonyms: List[str]
    field_name: Optional[str] = None
    note: Optional[str] = None


@dataclass
class QuestionTemplate:
    """常见业务问题模板。"""

    intent: str
    pattern: str
    sql_template: str
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    example: str | None = None


class SmartDataEngine:
    """极简数据引擎，用于支撑演示级别的工具调用。

    - 提供表结构、术语词典、问题模板
    - 模拟 SQL 执行与 API 调用，以便在无真实数据时也能运行
    """

    def __init__(
        self,
        schemas: List[TableSchema],
        vocabulary: List[BusinessVocabulary],
        templates: List[QuestionTemplate],
    ) -> None:
        self._schemas = {schema.name: schema for schema in schemas}
        self._vocabulary = vocabulary
        self._templates = templates

    # === 查询结构化配置 ===
    def list_tables(self) -> str:
        return "\n".join(schema.summarize() for schema in self._schemas.values())

    def get_table(self, name: str) -> str:
        schema = self._schemas.get(name)
        if not schema:
            return f"未找到表 {name}，请确认表名是否正确。"
        return schema.summarize()

    def search_terms(self, keyword: str) -> str:
        keyword_lower = keyword.lower()
        hits = [
            vocab
            for vocab in self._vocabulary
            if keyword_lower in vocab.term.lower()
            or keyword_lower in vocab.canonical.lower()
            or any(keyword_lower in syn.lower() for syn in vocab.synonyms)
        ]
        if not hits:
            return "未匹配到相关术语，可尝试直接描述业务含义。"
        return "\n".join(
            f"术语: {hit.term} | 标准表达: {hit.canonical} | 同义词: {', '.join(hit.synonyms)} | 对应字段: {hit.field_name or '未知'}"
            + (f" | 说明: {hit.note}" if hit.note else "")
            for hit in hits
        )

    def list_question_templates(self) -> str:
        lines = []
        for template in self._templates:
            lines.append(
                "\n".join(
                    [
                        f"意图: {template.intent}",
                        f"匹配模式: {template.pattern}",
                        f"SQL 模板: {template.sql_template}",
                        f"必填参数: {', '.join(template.required_params) if template.required_params else '无'}",
                        f"可选参数: {', '.join(template.optional_params) if template.optional_params else '无'}",
                        f"示例: {template.example or '无'}",
                    ]
                )
            )
        return "\n\n".join(lines)

    # === 模拟执行能力 ===
    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """模拟 SQL 执行，返回可预期的结构化结果。"""

        return {
            "status": "success",
            "sql": sql,
            "rows": [
                {"示例字段": "value", "count": 42},
            ],
            "note": "已模拟执行，真实环境请替换为数据库连接。",
        }

    def call_api(self, api_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """模拟业务 API 调用。"""

        return {
            "status": "success",
            "api": api_name,
            "payload": payload,
            "message": "已模拟调用业务 API，请在真实环境中接入实际接口。",
        }


# === 工具定义 ===

def build_tools(engine: SmartDataEngine):
    @tool
    def list_business_tables() -> str:
        """列出已注册的业务表及字段信息，用于上下文收集与 SQL 生成。"""

        return engine.list_tables()

    @tool
    def inspect_table(table_name: str) -> str:
        """查看指定表的字段、主键与索引信息。"""

        return engine.get_table(table_name)

    @tool
    def lookup_business_term(keyword: str) -> str:
        """查询业务术语词典，返回标准表达及对应字段。"""

        return engine.search_terms(keyword)

    @tool
    def show_question_templates() -> str:
        """展示已配置的业务问题模板，便于比对意图与缺失参数。"""

        return engine.list_question_templates()

    @tool
    def run_business_sql(sql: str) -> Dict[str, Any]:
        """执行 SQL（演示环境为模拟执行，真实环境需替换为实际数据库查询）。"""

        return engine.execute_sql(sql)

    @tool
    def call_business_api(api_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """调用业务 API（演示环境为模拟调用，确保在无外部依赖时也能运行）。"""

        return engine.call_api(api_name, payload)

    return [
        list_business_tables,
        inspect_table,
        lookup_business_term,
        show_question_templates,
        run_business_sql,
        call_business_api,
    ]


# === 智能体配置 ===

def build_subagents() -> list[Dict[str, Any]]:
    """构建多智能体配置，映射到《智能问数系统说明》中的关键阶段。"""

    return [
        {
            "name": "intent_router",
            "description": "意图识别与信息补全，判断 small_talk / data_query / api_call / hybrid 并补齐缺失参数。",
            "system_prompt": (
                "你负责阶段1：意图识别与信息补全。\n"
                "- 输出 JSON，总结意图、缺失信息、引导性追问。\n"
                "- 若能匹配常见问题模板，需要列出使用的模板与参数。\n"
                "- 保持中文回复。"
            ),
            "tools": [],
        },
        {
            "name": "context_builder",
            "description": "上下文收集与语义匹配，整理表结构、术语词典、时间范围等关键信息。",
            "system_prompt": (
                "你负责阶段2：上下文收集。\n"
                "- 主动使用工具获取表结构、术语词典和问题模板。\n"
                "- 结合用户意图，推断需要的字段、时间范围与过滤条件。\n"
                "- 产出结构化摘要，供后续 SQL 生成或 API 调用使用。"
            ),
            "tools": [
                "list_business_tables",
                "inspect_table",
                "lookup_business_term",
                "show_question_templates",
            ],
        },
        {
            "name": "sql_planner",
            "description": "负责 SQL 生成与 Engine 转译，确保过滤条件完备且可执行。",
            "system_prompt": (
                "你负责阶段3：SQL 生成与转译。\n"
                "- 基于上下文确定查询目标、字段与过滤条件，生成标准 SQL。\n"
                "- 若 SQL 涉及时间范围，需明确起止时间。\n"
                "- 完成后调用 run_business_sql 工具执行，返回简洁结果概览。"
            ),
            "tools": [
                "list_business_tables",
                "inspect_table",
                "run_business_sql",
            ],
        },
        {
            "name": "api_coordinator",
            "description": "处理 api_call 或 hybrid 场景，提取参数并调用业务 API。",
            "system_prompt": (
                "你负责阶段4：API 调用与混合流程。\n"
                "- 提取所需参数，缺失时给出中文追问。\n"
                "- 使用 call_business_api 执行，并返回稳健的结果说明。"
            ),
            "tools": ["call_business_api"],
        },
    ]


def build_system_prompt() -> str:
    """主代理提示词，串联多阶段协作并要求中文输出。"""

    return (
        "你是多智能体智能问数编排器，必须使用中文回答并对异常场景具备容错能力。\n"
        "按照《智能问数系统说明》的六个阶段工作：\n"
        "1) 意图识别与信息补全（intent_router）\n"
        "2) 上下文收集（context_builder）\n"
        "3) SQL 生成与 Engine 转译（sql_planner）\n"
        "4) 查询执行或 API 调用（根据意图选择 run_business_sql 或 call_business_api）\n"
        "5) 结果优化与可视化建议\n"
        "6) 经验沉淀与知识库更新建议\n"
        "使用 task 工具将工作委派给对应子智能体，并在最终回复中合并关键结论。\n"
        "无论数据或接口是否可用，都要给出替代方案或下一步指引。"
    )


def create_smart_questioning_agent() -> Any:
    """创建可直接运行的多智能体智能问数系统。"""

    schemas = [
        TableSchema(
            name="pingshan_stat_info",
            zh_name="坪山区事件统计信息表",
            usage="记录所有上报事件的详细信息和处理状态",
            primary_key="EVENT_ID",
            indexes=["CREATE_TIME", "first_unit_name", "SUB_TYPE_NAME"],
            fields=[
                TableField(
                    name="EVENT_ID",
                    dtype="VARCHAR(20)",
                    zh_name="事件编号",
                    meaning="唯一标识事件的编号",
                    example="PS20250304001",
                ),
                TableField(
                    name="EVENT_SRC_NAME",
                    dtype="VARCHAR(50)",
                    zh_name="事件来源",
                    meaning="事件的上报渠道或来源系统",
                    example="12345热线",
                ),
                TableField(
                    name="first_unit_name",
                    dtype="VARCHAR(100)",
                    zh_name="处理单位",
                    meaning="首次负责处理该事件的单位名称",
                    example="坪山街道办事处",
                ),
                TableField(
                    name="SUB_TYPE_NAME",
                    dtype="VARCHAR(50)",
                    zh_name="事件子类型",
                    meaning="事件的具体分类",
                    example="暴露垃圾",
                ),
                TableField(
                    name="CREATE_TIME",
                    dtype="TIMESTAMP",
                    zh_name="创建时间",
                    meaning="事件在系统中的创建时间",
                    example="2025-03-04 14:30:00",
                ),
                TableField(
                    name="EVENT_DESC",
                    dtype="TEXT",
                    zh_name="事件描述",
                    meaning="事件的详细描述信息",
                    example="小区门口垃圾堆积...",
                ),
            ],
        )
    ]

    vocabulary = [
        BusinessVocabulary(
            term="处理单位",
            canonical="first_unit_name",
            synonyms=["负责单位", "责任单位", "处理部门"],
            field_name="first_unit_name",
            note="首次处理事件的单位",
        ),
        BusinessVocabulary(
            term="事件类型",
            canonical="SUB_TYPE_NAME",
            synonyms=["案件类型", "问题类型", "事件分类"],
            field_name="SUB_TYPE_NAME",
        ),
        BusinessVocabulary(
            term="垃圾事件",
            canonical="垃圾相关事件",
            synonyms=["暴露垃圾", "垃圾堆积", "垃圾清理"],
            field_name=None,
            note="需在描述和类型字段做模糊匹配",
        ),
        BusinessVocabulary(
            term="本月",
            canonical="当前自然月",
            synonyms=["这个月", "当月"],
            field_name="CREATE_TIME",
            note="从本月1日到下月1日",
        ),
        BusinessVocabulary(
            term="最近",
            canonical="最近7天",
            synonyms=["近期", "最近几天"],
            field_name="CREATE_TIME",
            note="默认7天内",
        ),
    ]

    templates = [
        QuestionTemplate(
            intent="data_query",
            pattern=".*街道.*处理.*多少.*案件",
            sql_template=(
                "SELECT COUNT(*) FROM pingshan_stat_info "
                "WHERE first_unit_name LIKE '%{unit_name}%' AND {time_filter}"
            ),
            required_params=["unit_name"],
            optional_params=["time_filter"],
            example="这个月坪山街道办事处处理了多少案件",
        ),
        QuestionTemplate(
            intent="data_query",
            pattern=".*事件.*主要来源",
            sql_template=(
                "SELECT EVENT_SRC_NAME, COUNT(*) FROM pingshan_stat_info "
                "WHERE {event_filter} GROUP BY EVENT_SRC_NAME ORDER BY COUNT(*) DESC"
            ),
            required_params=["event_filter"],
            example="暴露垃圾事件的主要来源是什么",
        ),
        QuestionTemplate(
            intent="data_query",
            pattern="统计.*事件.*数量",
            sql_template=(
                "SELECT SUB_TYPE_NAME, COUNT(*) FROM pingshan_stat_info "
                "WHERE SUB_TYPE_NAME IS NOT NULL GROUP BY SUB_TYPE_NAME ORDER BY COUNT(*) DESC"
            ),
            required_params=[],
            example="统计各类型事件的数量",
        ),
    ]

    engine = SmartDataEngine(schemas=schemas, vocabulary=vocabulary, templates=templates)
    tools = build_tools(engine)

    agent = create_deep_agent(
        tools=tools,
        subagents=build_subagents(),
        system_prompt=build_system_prompt(),
    )
    return agent


if __name__ == "__main__":
    # 示例：创建智能问数代理。运行时需要配置可用的 LLM 密钥。
    agent = create_smart_questioning_agent()
    print("智能问数系统已就绪，可通过 agent.invoke({\"messages\": [...]}) 调用。")

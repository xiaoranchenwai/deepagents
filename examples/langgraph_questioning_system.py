"""基于 LangGraph 的多智能体智能问数系统示例。

该示例沿用《智能问数系统说明》中描述的六阶段流程，
但使用 LangGraph 显式编排节点（意图识别→上下文补全→SQL 生成→执行→结果生成→反思）。
数据库配置与 `smart_questioning_system.py` 保持一致，
可在具备 mysql-connector-python 且能访问数据库时直连，
否则自动回退到模拟结果，保证示例可运行。
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph


@dataclass
class TableField:
    name: str
    dtype: str
    zh_name: str
    meaning: str
    example: str


@dataclass
class TableSchema:
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
    term: str
    canonical: str
    synonyms: List[str]
    field_name: Optional[str] = None
    note: Optional[str] = None


@dataclass
class QuestionTemplate:
    intent: str
    pattern: str
    sql_template: str
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    example: str | None = None


DB_CONFIG = {
    "host": "10.250.2.19",
    "port": 3306,
    "user": "root",
    "password": "hwits888",
    "database": "pingshan",
}


class SmartDataEngine:
    """极简数据引擎，兼容真实查询与模拟返回。"""

    def __init__(
        self,
        schemas: List[TableSchema],
        vocabulary: List[BusinessVocabulary],
        templates: List[QuestionTemplate],
        db_config: Optional[Dict[str, Any]] = None,
        connect_timeout: int = 5,
    ) -> None:
        self._schemas = {schema.name: schema for schema in schemas}
        self._vocabulary = vocabulary
        self._templates = templates
        self._db_config = db_config
        self._connect_timeout = connect_timeout

    def list_tables(self) -> str:
        return "\n".join(schema.summarize() for schema in self._schemas.values())

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
            f"术语: {hit.term} | 标准: {hit.canonical} | 同义词: {', '.join(hit.synonyms)}"
            + (f" | 字段: {hit.field_name}" if hit.field_name else "")
            + (f" | 说明: {hit.note}" if hit.note else "")
            for hit in hits
        )

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        if not self._db_config:
            return {
                "status": "mock",
                "sql": sql,
                "rows": [{"示例字段": "value", "count": 42}],
                "note": "未提供数据库配置，已返回模拟结果。",
            }

        connector_spec = importlib.util.find_spec("mysql.connector")
        if connector_spec is None:
            return {
                "status": "mock",
                "sql": sql,
                "rows": [{"示例字段": "value", "count": 42}],
                "note": "未检测到 mysql-connector-python，已返回模拟结果。",
            }

        import mysql.connector  # type: ignore

        connection = None
        try:
            connection = mysql.connector.connect(
                **self._db_config, connection_timeout=self._connect_timeout
            )
            cursor = connection.cursor(dictionary=True)
            cursor.execute(sql)
            rows = cursor.fetchall()
            return {"status": "success", "sql": sql, "rows": rows, "note": "已连接 MySQL 返回真实结果。"}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "sql": sql,
                "rows": [],
                "note": f"查询失败，回退到模拟模式: {exc}",
            }
        finally:
            if connection is not None:
                connection.close()


class QAState(TypedDict, total=False):
    """LangGraph 状态定义。"""

    question: str
    intent: str
    template: QuestionTemplate
    params: Dict[str, str]
    sql: str
    result: Dict[str, Any]
    logs: List[str]


def _default_time_filter() -> str:
    today = _dt.date.today()
    first_day = today.replace(day=1).strftime("%Y-%m-01")
    next_month = (first_day[:7] + "-01")
    next_month_dt = _dt.datetime.strptime(next_month, "%Y-%m-%d") + _dt.timedelta(days=32)
    next_first = next_month_dt.replace(day=1).strftime("%Y-%m-%d")
    return f"CREATE_TIME >= '{first_day}' AND CREATE_TIME < '{next_first}'"


def _build_assets() -> tuple[List[TableSchema], List[BusinessVocabulary], List[QuestionTemplate]]:
    fields = [
        TableField("REC_ID", "bigint", "记录ID", "主键，唯一标识", "1782698"),
        TableField("TASK_NUM", "varchar(40)", "工单号", "外部任务编号", "SZPS202502252032300001"),
        TableField("CREATE_TIME", "timestamp", "创建时间", "案件创建时间", "2025-02-25 00:44:18"),
        TableField("ADDRESS", "text", "地址", "事件发生地址", "坪山区环盛路万樾府"),
        TableField("STREET_NAME", "varchar(40)", "街道", "所属街道", "坪山街道"),
        TableField("COMMUNITY_NAME", "varchar(40)", "社区", "所属社区", "坪环社区"),
        TableField("SUB_TYPE_NAME", "varchar(100)", "事件小类", "案件细分类型", "暴露垃圾"),
        TableField("THIRD_TYPE_NAME", "varchar(100)", "事件三级分类", "更细分类", "无照经营游商"),
        TableField("MAX_EVENT_TYPE_NAME", "varchar(200)", "最大事件类型", "归一化类型", "暴露垃圾"),
        TableField("EVENT_DESC", "text", "事件描述", "市民描述", "市民反映用水问题"),
        TableField("EVENT_SRC_NAME", "varchar(40)", "事件来源", "上报渠道", "i深圳app"),
        TableField("REC_TYPE_NAME", "varchar(40)", "录入渠道", "工单渠道", "机动中队"),
        TableField("first_unit_name", "varchar(200)", "首办单位", "首次处置单位", "坪山街道办事处"),
    ]

    schemas = [
        TableSchema(
            name="pingshan_stat_info",
            zh_name="坪山区事件统计表",
            usage="统一存储民生事件及工单信息，用于治理与分析。",
            primary_key="REC_ID",
            indexes=["CREATE_TIME", "STREET_NAME", "SUB_TYPE_NAME", "EVENT_SRC_NAME", "REC_TYPE_NAME", "first_unit_name"],
            fields=fields,
        )
    ]

    vocabulary = [
        BusinessVocabulary("暴露垃圾", "垃圾相关事件", ["垃圾堆积", "街面垃圾", "垃圾"], None, "需在小类、三级类、最大类或描述中模糊匹配"),
        BusinessVocabulary("处理单位", "first_unit_name", ["责任单位", "处置单位"], "first_unit_name", "首办单位"),
        BusinessVocabulary("街道", "STREET_NAME", ["街道办", "街道办事处"], "STREET_NAME", None),
        BusinessVocabulary("创建时间", "CREATE_TIME", ["时间", "日期"], "CREATE_TIME", "支持日、月过滤"),
        BusinessVocabulary("来源渠道", "EVENT_SRC_NAME", ["来源", "上报渠道"], "EVENT_SRC_NAME", None),
    ]

    templates = [
        QuestionTemplate(
            intent="data_query",
            pattern=".*街道.*处理.*多少.*案件",
            sql_template=(
                "SELECT COUNT(*) AS 案件数量 FROM pingshan_stat_info "
                "WHERE first_unit_name LIKE '%{unit_name}%' AND {time_filter}"
            ),
            required_params=["unit_name"],
            optional_params=["time_filter"],
            example="这个月坪山街道办事处处理了多少案件",
        ),
        QuestionTemplate(
            intent="data_query",
            pattern=".*暴露垃圾.*主要来源",
            sql_template=(
                "SELECT EVENT_SRC_NAME AS 事件来源, COUNT(*) AS 数量 FROM pingshan_stat_info "
                "WHERE {event_filter} GROUP BY EVENT_SRC_NAME ORDER BY 数量 DESC"
            ),
            required_params=["event_filter"],
            example="暴露垃圾事件的主要来源是什么",
        ),
        QuestionTemplate(
            intent="data_query",
            pattern="统计.*事件.*数量",
            sql_template=(
                "SELECT SUB_TYPE_NAME AS 事件类型, COUNT(*) AS 事件数量 FROM pingshan_stat_info "
                "WHERE SUB_TYPE_NAME IS NOT NULL AND SUB_TYPE_NAME <> '' "
                "GROUP BY SUB_TYPE_NAME ORDER BY 事件数量 DESC"
            ),
            required_params=[],
            example="统计各类型事件的数量",
        ),
    ]

    return schemas, vocabulary, templates


def _detect_template(question: str, templates: List[QuestionTemplate]) -> Optional[QuestionTemplate]:
    for tpl in templates:
        if re.search(tpl.pattern, question):
            return tpl
    return None


def _derive_params(question: str, template: QuestionTemplate) -> Dict[str, str]:
    params: Dict[str, str] = {}

    if "unit_name" in template.required_params or "unit_name" in template.optional_params:
        match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]+街道办事处)", question)
        if match:
            params["unit_name"] = match.group(1)

    if "event_filter" in template.required_params:
        params["event_filter"] = (
            "(SUB_TYPE_NAME LIKE '%暴露垃圾%' OR THIRD_TYPE_NAME LIKE '%暴露垃圾%' "
            "OR MAX_EVENT_TYPE_NAME LIKE '%暴露垃圾%' OR EVENT_DESC LIKE '%暴露垃圾%')"
        )

    if "time_filter" in template.optional_params:
        params["time_filter"] = _default_time_filter()

    return params


def _render_sql(template: QuestionTemplate, params: Dict[str, str]) -> str:
    filled = {**{key: "" for key in template.required_params + template.optional_params}, **params}
    return template.sql_template.format(**filled)


def build_langgraph_agent():
    schemas, vocabulary, templates = _build_assets()
    engine = SmartDataEngine(schemas=schemas, vocabulary=vocabulary, templates=templates, db_config=DB_CONFIG)

    def detect_intent(state: QAState) -> QAState:
        question = state["question"]
        tpl = _detect_template(question, templates)
        logs = state.get("logs", []) + ["完成意图识别"]
        if not tpl:
            return {**state, "logs": logs + ["未匹配到模板，默认统计事件数量。"], "intent": "fallback"}
        params = _derive_params(question, tpl)
        return {**state, "intent": tpl.intent, "template": tpl, "params": params, "logs": logs}

    def enrich_context(state: QAState) -> QAState:
        logs = state.get("logs", [])
        tpl = state.get("template")
        params = state.get("params", {})
        if tpl:
            missing = [p for p in tpl.required_params if p not in params or not params[p]]
            if missing:
                logs.append(f"缺失参数 {missing}，尝试默认填充")
                if "time_filter" in missing:
                    params["time_filter"] = _default_time_filter()
        logs.append("完成上下文补全")
        return {**state, "params": params, "logs": logs}

    def build_sql(state: QAState) -> QAState:
        logs = state.get("logs", [])
        tpl = state.get("template")
        if not tpl:
            sql = "SELECT COUNT(*) AS 事件数量 FROM pingshan_stat_info"
            return {**state, "sql": sql, "logs": logs + ["使用兜底 SQL"]}
        sql = _render_sql(tpl, state.get("params", {}))
        return {**state, "sql": sql, "logs": logs + ["完成 SQL 生成"]}

    def execute(state: QAState) -> QAState:
        logs = state.get("logs", [])
        sql = state.get("sql", "")
        result = engine.execute_sql(sql)
        return {**state, "result": result, "logs": logs + ["完成 SQL 执行"]}

    def respond(state: QAState) -> QAState:
        logs = state.get("logs", [])
        result = state.get("result", {})
        status = result.get("status", "mock")
        rows = result.get("rows", [])
        summary = f"执行状态: {status}。返回 {len(rows)} 行。"
        return {**state, "logs": logs + [summary]}

    graph = StateGraph(QAState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("enrich_context", enrich_context)
    graph.add_node("build_sql", build_sql)
    graph.add_node("execute", execute)
    graph.add_node("respond", respond)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "enrich_context")
    graph.add_edge("enrich_context", "build_sql")
    graph.add_edge("build_sql", "execute")
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_langgraph_agent()
    demo_state = app.invoke({"question": "暴露垃圾事件的主要来源是什么", "logs": []})
    print("运行完成，状态如下：")
    print(demo_state)

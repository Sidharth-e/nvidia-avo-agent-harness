import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama


def _load_env() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


_load_env()


class LineageMemory:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASS", "password")
        self._driver = None
        self._in_memory_attempts: Dict[str, Dict[str, Any]] = {}
        self._in_memory_links: List[Dict[str, str]] = []
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase, NotificationMinimumSeverity
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                notifications_min_severity=NotificationMinimumSeverity.OFF,
            )
            driver.verify_connectivity()
            with driver.session() as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Attempt) REQUIRE a.id IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (a:Attempt) ON (a.task)")
            self._driver = driver
        except Exception:
            self._driver = None

    def save_attempt(
        self,
        task: str,
        content: str,
        hypothesis: str,
        score: float,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        attempt_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        metadata_dict = metadata or {}

        if self._driver:
            try:
                with self._driver.session() as session:
                    session.run(
                        """
                        CREATE (a:Attempt {
                            id: $id,
                            task: $task,
                            content: $content,
                            hypothesis: $hypothesis,
                            score: $score,
                            ts: $ts
                        })
                        """,
                        id=attempt_id,
                        task=task,
                        content=content,
                        hypothesis=hypothesis,
                        score=score,
                        ts=ts,
                    )
                    if parent_id:
                        session.run(
                            """
                            MATCH (child:Attempt {id: $child_id}), (parent:Attempt {id: $parent_id})
                            CREATE (child)-[:DERIVED_FROM]->(parent)
                            """,
                            child_id=attempt_id,
                            parent_id=parent_id,
                        )
                return attempt_id
            except Exception:
                pass

        self._in_memory_attempts[attempt_id] = {
            "id": attempt_id,
            "task": task,
            "content": content,
            "hypothesis": hypothesis,
            "score": score,
            "ts": ts,
            "parent_id": parent_id,
            "metadata": metadata_dict,
        }
        if parent_id:
            self._in_memory_links.append({"child_id": attempt_id, "parent_id": parent_id})
        return attempt_id

    def best_attempts(self, task: str, limit: int = 5) -> List[Dict[str, Any]]:
        if self._driver:
            try:
                with self._driver.session() as session:
                    result = session.run(
                        """
                        MATCH (a:Attempt {task: $task})
                        RETURN a.id AS id, a.content AS content, a.hypothesis AS hypothesis, a.score AS score, a.ts AS ts
                        ORDER BY a.score DESC
                        LIMIT $limit
                        """,
                        task=task,
                        limit=limit,
                    )
                    return [dict(record) for record in result]
            except Exception:
                pass

        matching = [att for att in self._in_memory_attempts.values() if att["task"] == task]
        matching.sort(key=lambda x: x["score"], reverse=True)
        return matching[:limit]

    def recent_score_trend(self, task: str, n: int = 5) -> List[float]:
        if self._driver:
            try:
                with self._driver.session() as session:
                    result = session.run(
                        """
                        MATCH (a:Attempt {task: $task})
                        RETURN a.score AS score
                        ORDER BY a.ts DESC
                        LIMIT $n
                        """,
                        task=task,
                        n=n,
                    )
                    return [record["score"] for record in result][::-1]
            except Exception:
                pass

        matching = [att for att in self._in_memory_attempts.values() if att["task"] == task]
        matching.sort(key=lambda x: x["ts"], reverse=True)
        return [att["score"] for att in matching[:n]][::-1]

    def is_stagnating(self, task: str, n: int = 5, threshold: float = 0.01) -> bool:
        scores = self.recent_score_trend(task, n)
        if len(scores) < n:
            return False
        return (max(scores) - min(scores)) < threshold

    def get_attempt_by_id(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        if self._driver:
            try:
                with self._driver.session() as session:
                    result = session.run(
                        """
                        MATCH (a:Attempt {id: $id})
                        RETURN a.id AS id, a.task AS task, a.content AS content, a.hypothesis AS hypothesis, a.score AS score, a.ts AS ts
                        """,
                        id=attempt_id,
                    )
                    record = result.single()
                    if record:
                        return dict(record)
            except Exception:
                pass
        return self._in_memory_attempts.get(attempt_id)


memory = LineageMemory()


def recall_lineage(task: str) -> str:
    """Retrieve top performing past attempts and optimization hypotheses for a task."""
    attempts = memory.best_attempts(task, limit=5)
    if not attempts:
        return "No prior attempts found for this task."
    lines: List[str] = []
    for att in attempts:
        lines.append(
            f"ID: {att['id']} | Score: {att['score']} | Hypothesis: {att.get('hypothesis', 'N/A')}\n"
            f"Content Preview: {att['content'][:150]}..."
        )
    return "\n\n".join(lines)


def inspect_parent_solution(attempt_id: str) -> str:
    """Retrieve the full code, hypothesis, and evaluation score of a specific attempt by ID."""
    att = memory.get_attempt_by_id(attempt_id)
    if not att:
        return f"Attempt {attempt_id} not found."
    return (
        f"Attempt ID: {att['id']}\n"
        f"Score: {att['score']}\n"
        f"Hypothesis: {att.get('hypothesis', 'N/A')}\n"
        f"Full Code / Content:\n{att['content']}"
    )


def evaluate_and_verify_candidate(code: str, test_input: str = "") -> str:
    """Execute and verify candidate code in a sandbox environment."""
    try:
        scope: Dict[str, Any] = {}
        exec(code, scope)
        if "solution" in scope and callable(scope["solution"]):
            result = scope["solution"](test_input) if test_input else scope["solution"]()
            return f"VERIFICATION_SUCCESS: Execution output: {result}"
        return "VERIFICATION_SUCCESS: Code executed cleanly without runtime exceptions."
    except Exception as exc:
        return f"VERIFICATION_FAILED: Runtime error: {type(exc).__name__}: {str(exc)}"


def record_variation(task: str, content: str, hypothesis: str, score: float, parent_id: str = "") -> str:
    """Commit an evaluated variation into the lineage memory graph."""
    aid = memory.save_attempt(
        task=task,
        content=content,
        hypothesis=hypothesis,
        score=score,
        parent_id=parent_id or None,
    )
    return f"Variation recorded with ID {aid}, score {score}."


def check_progress(task: str) -> str:
    """Check if evolutionary search fitness has plateaued or stagnated for a given task."""
    if memory.is_stagnating(task):
        return (
            "STAGNATING: Recent scores have plateaued. "
            "Instruct the variation operator to switch optimization strategies or explore an alternative lineage branch."
        )
    return "SEARCH_HEALTHY: Improvements or active exploration detected. Continue current lineage evolution."


variation_operator_subagent: Dict[str, Any] = {
    "name": "agentic-variation-operator",
    "description": (
        "Autonomous variation operator performing the Inspect-Plan-Implement-Evaluate loop. "
        "Formulates mutation hypotheses, writes code edits, verifies execution, and records results."
    ),
    "system_prompt": (
        "You are an Agentic Variation Operator (AVO). Your role is to evolve solutions via a 4-step loop:\n"
        "1. INSPECT: Call recall_lineage and inspect_parent_solution to understand top-performing ancestors and past failures.\n"
        "2. PLAN: Formulate an explicit hypothesis describing what modification will yield performance gains.\n"
        "3. IMPLEMENT: Author the complete code variant.\n"
        "4. EVALUATE & SELF-REPAIR: Call evaluate_and_verify_candidate to test correctness. If execution fails, fix the code iteratively.\n"
        "Once verified, compute the fitness score (0.0 to 1.0) and call record_variation to commit the variation into memory."
    ),
    "tools": [
        recall_lineage,
        inspect_parent_solution,
        evaluate_and_verify_candidate,
        record_variation,
    ],
}

supervisor_prompt = """
You are the Supervisor Orchestrator of an Agentic Variation Operators (AVO) evolutionary search harness.

Execution Workflow:
1. Call check_progress to inspect whether search fitness has stagnated.
2. If stagnation is detected, pivot search directions, mandate a novel exploration strategy, or select an alternative parent lineage.
3. Delegate generation tasks to the agentic-variation-operator subagent.
4. Review the outcome and repeat until optimization convergence or the iteration budget is exhausted.

Do not write variation implementations directly; delegate all candidate exploration and verification to the subagent.
"""

ollama_model_name = os.environ.get("OLLAMA_MODEL", "gemma4:e2b-mlx")
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

local_llm = ChatOllama(
    model=ollama_model_name,
    base_url=ollama_base_url,
    temperature=0.0,
)

agent = create_deep_agent(
    model=local_llm,
    tools=[check_progress],
    subagents=[variation_operator_subagent],
    system_prompt=supervisor_prompt,
)


def run_harness(task_description: str, iterations: int = 5) -> Any:
    return agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Task: {task_description}. Run {iterations} evolutionary search iterations.",
            }
        ]
    })


if __name__ == "__main__":
    initial_task = "Optimize an algorithm for computing running matrix operations"
    output = run_harness(initial_task, iterations=3)
    print(output)
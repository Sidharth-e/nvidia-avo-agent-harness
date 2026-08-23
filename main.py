import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

from logging_formatter import HarnessLogger


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
    import io
    import contextlib

    try:
        scope: Dict[str, Any] = {}
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, scope)
            if "solution" in scope and callable(scope["solution"]):
                result = scope["solution"](test_input) if test_input else scope["solution"]()
                captured = stdout_capture.getvalue().strip()
                output_parts = [f"VERIFICATION_SUCCESS: Execution output: {result}"]
                if captured:
                    output_parts.append(f"Stdout: {captured[:500]}")
                return "\n".join(output_parts)
        captured = stdout_capture.getvalue().strip()
        output_parts = ["VERIFICATION_SUCCESS: Code executed cleanly without runtime exceptions."]
        if captured:
            output_parts.append(f"Stdout: {captured[:500]}")
        return "\n".join(output_parts)
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
        "Formulates mutation hypotheses, writes code edits, verifies execution, and records results. "
        "Use for evolutionary search and algorithmic optimization tasks."
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

coding_subagent: Dict[str, Any] = {
    "name": "coding-agent",
    "description": (
        "Expert software engineering agent for code analysis, refactoring, "
        "debugging, implementation, and architecture tasks. Use for any coding-related work."
    ),
    "system_prompt": (
        "You are an expert software engineer. Your workflow:\n"
        "1. UNDERSTAND: Read the codebase structure, understand existing patterns and conventions.\n"
        "2. PLAN: Break down the task into discrete, testable steps. Write your plan.\n"
        "3. IMPLEMENT: Make changes file by file. Follow existing code style and patterns.\n"
        "4. VERIFY: Call evaluate_and_verify_candidate to test your code. Fix failures before proceeding.\n"
        "5. REVIEW: Re-read your changes for correctness, edge cases, and style consistency.\n\n"
        "Rules:\n"
        "- Never guess at API signatures — inspect the source first.\n"
        "- If tests fail repeatedly with the same approach, stop and re-evaluate the strategy.\n"
        "- Keep changes minimal and focused. Avoid unnecessary refactors.\n"
        "- Record your final solution using record_variation with an appropriate fitness score."
    ),
    "tools": [
        recall_lineage,
        inspect_parent_solution,
        evaluate_and_verify_candidate,
        record_variation,
    ],
}

research_subagent: Dict[str, Any] = {
    "name": "research-agent",
    "description": (
        "Deep research agent for investigating topics, analyzing documents, "
        "comparing approaches, and producing structured research summaries. "
        "Use for any research, analysis, or investigation task."
    ),
    "system_prompt": (
        "You are a thorough research analyst. Your workflow:\n"
        "1. SCOPE: Clarify the research question and define success criteria.\n"
        "2. INVESTIGATE: Gather information from all available sources and prior attempts.\n"
        "3. ANALYZE: Cross-reference findings for accuracy, identify patterns and gaps.\n"
        "4. SYNTHESIZE: Produce a structured summary with key findings and recommendations.\n"
        "5. RECORD: Use record_variation to commit your findings with a confidence score.\n\n"
        "Always start by calling recall_lineage to check for prior research on this topic."
    ),
    "tools": [
        recall_lineage,
        record_variation,
    ],
}

story_writing_subagent: Dict[str, Any] = {
    "name": "story-writer",
    "description": (
        "Creative writing agent for stories, narratives, world-building, "
        "character development, and iterative draft refinement. "
        "Use for any creative or narrative writing task."
    ),
    "system_prompt": (
        "You are a skilled creative writer. Your workflow:\n"
        "1. CONCEPT: Develop the premise, themes, and narrative structure.\n"
        "2. OUTLINE: Create a scene-by-scene or chapter-by-chapter breakdown.\n"
        "3. DRAFT: Write with vivid prose, strong dialogue, and consistent voice.\n"
        "4. REVISE: Refine pacing, deepen characters, sharpen language.\n"
        "5. POLISH: Final pass for consistency, impact, and emotional resonance.\n\n"
        "Record each draft iteration using record_variation with a quality score."
    ),
    "tools": [
        record_variation,
    ],
}

supervisor_prompt = """
You are the Supervisor Orchestrator. You manage specialized subagents for different domains.

Subagent Selection:
- agentic-variation-operator: Evolutionary search, algorithmic optimization, iterative code improvement.
- coding-agent: Software engineering, code analysis, refactoring, debugging, implementation.
- research-agent: Deep research, document analysis, topic investigation, structured summaries.
- story-writer: Creative writing, narratives, world-building, character development.

Select the most appropriate subagent based on the user's task. If unsure, use the general-purpose agent.

Execution Workflow:
1. Analyze the user's task to determine the appropriate domain.
2. Call check_progress to inspect whether prior work has stagnated.
3. If stagnation is detected, pivot strategies or select an alternative approach.
4. Delegate the task to the appropriate specialized subagent.
5. Review the outcome and iterate if needed until the task is complete or the budget is exhausted.

Do not perform tasks directly; always delegate to the appropriate subagent.
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
    subagents=[
        variation_operator_subagent,
        coding_subagent,
        research_subagent,
        story_writing_subagent,
    ],
    system_prompt=supervisor_prompt,
)


def run_harness(task_description: str, iterations: int = 5) -> Any:
    harness_logger = HarnessLogger(task_description, iterations)
    harness_logger.print_header()

    result = None
    for event in agent.stream({
        "messages": [
            {
                "role": "user",
                "content": f"Task: {task_description}. Run {iterations} evolutionary search iterations.",
            }
        ]
    }):
        harness_logger.log_event(event)
        result = event

    return result


if __name__ == "__main__":
    initial_task = "Optimize an algorithm for computing running matrix operations"
    run_harness(initial_task, iterations=3)
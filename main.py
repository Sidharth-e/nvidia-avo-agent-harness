import os
import re
import time
import uuid
import io
import contextlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from logging_formatter import HarnessLogger


def _load_env() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


_load_env()


# =========================================================================
# LINEAGE MEMORY — now auto-links parent_id to last attempt for the task,
# so DERIVED_FROM edges actually form without the model remembering to pass one.
# =========================================================================

class LineageMemory:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASS", "password")
        self._driver = None
        self._in_memory_attempts: Dict[str, Dict[str, Any]] = {}
        self._last_attempt_id: Dict[str, str] = {}  # task -> most recent attempt id
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase, NotificationMinimumSeverity
            driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password),
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
        self, task: str, content: str, hypothesis: str, score: float,
        parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        attempt_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        # Auto-link to the last attempt on this task if caller didn't specify one.
        effective_parent = parent_id or self._last_attempt_id.get(task)

        if self._driver:
            try:
                with self._driver.session() as session:
                    session.run(
                        """
                        CREATE (a:Attempt {id: $id, task: $task, content: $content,
                                            hypothesis: $hypothesis, score: $score, ts: $ts})
                        """,
                        id=attempt_id, task=task, content=content,
                        hypothesis=hypothesis, score=score, ts=ts,
                    )
                    if effective_parent:
                        session.run(
                            """
                            MATCH (child:Attempt {id: $child_id}), (parent:Attempt {id: $parent_id})
                            CREATE (child)-[:DERIVED_FROM]->(parent)
                            """,
                            child_id=attempt_id, parent_id=effective_parent,
                        )
                self._last_attempt_id[task] = attempt_id
                return attempt_id
            except Exception:
                pass

        self._in_memory_attempts[attempt_id] = {
            "id": attempt_id, "task": task, "content": content, "hypothesis": hypothesis,
            "score": score, "ts": ts, "parent_id": effective_parent, "metadata": metadata or {},
        }
        self._last_attempt_id[task] = attempt_id
        return attempt_id

    def best_attempts(self, task: str, limit: int = 5) -> List[Dict[str, Any]]:
        if self._driver:
            try:
                with self._driver.session() as session:
                    result = session.run(
                        """
                        MATCH (a:Attempt {task: $task})
                        RETURN a.id AS id, a.content AS content, a.hypothesis AS hypothesis,
                               a.score AS score, a.ts AS ts
                        ORDER BY a.score DESC LIMIT $limit
                        """,
                        task=task, limit=limit,
                    )
                    return [dict(r) for r in result]
            except Exception:
                pass
        matching = [a for a in self._in_memory_attempts.values() if a["task"] == task]
        matching.sort(key=lambda x: x["score"], reverse=True)
        return matching[:limit]

    def get_all_attempts(self, task: str) -> List[Dict[str, Any]]:
        """Used by the harness to verify iteration count actually grew — never trust the model's claim alone."""
        if self._driver:
            try:
                with self._driver.session() as session:
                    result = session.run(
                        "MATCH (a:Attempt {task: $task}) RETURN a.id AS id, a.score AS score, a.ts AS ts ORDER BY a.ts",
                        task=task,
                    )
                    return [dict(r) for r in result]
            except Exception:
                pass
        return [a for a in self._in_memory_attempts.values() if a["task"] == task]

    def recent_score_trend(self, task: str, n: int = 5) -> List[float]:
        attempts = self.get_all_attempts(task)
        return [a["score"] for a in attempts[-n:]]

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
                        """MATCH (a:Attempt {id: $id}) RETURN a.id AS id, a.task AS task, a.content AS content,
                           a.hypothesis AS hypothesis, a.score AS score, a.ts AS ts""",
                        id=attempt_id,
                    )
                    record = result.single()
                    if record:
                        return dict(record)
            except Exception:
                pass
        return self._in_memory_attempts.get(attempt_id)


memory = LineageMemory()


# =========================================================================
# EVALUATORS — task-specific, pluggable. Fixes bug #1: a single global
# matrix-only benchmark was silently zero-scoring every non-matrix task.
# =========================================================================

def _extract_code(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def evaluate_matrix_code(code: str) -> Optional[float]:
    """Benchmark for the running-matrix-product task specifically."""
    scope: Dict[str, Any] = {}
    try:
        exec(code, scope)
    except Exception:
        return None
    func = scope.get("calculate_running_matrix_product")
    if not callable(func):
        return None

    random.seed(42)
    def make_matrix(r, c):
        return [[random.random() for _ in range(c)] for _ in range(r)]
    matrices = [make_matrix(20, 20) for _ in range(10)]

    try:
        res = func([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        if len(res) != 2:
            return None
        start = time.perf_counter()
        for _ in range(50):
            func(matrices)
        elapsed = time.perf_counter() - start
        return 1.0 / (1.0 + elapsed)
    except Exception:
        return None


def evaluate_generic_code(code: str) -> Optional[float]:
    """Fallback for arbitrary code: reward clean execution + a lightweight
    complexity/correctness proxy. Not a substitute for a real task evaluator —
    use this only when no task-specific evaluator matches."""
    try:
        scope: Dict[str, Any] = {}
        stdout = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(stdout):
            exec(code, scope)
        elapsed = time.perf_counter() - start
    except Exception:
        return 0.0  # executes-cleanly gate failed — explicit 0, not silent

    # Executed without error: baseline credit, small bonus for fast execution.
    return round(0.5 + 0.5 / (1.0 + elapsed), 4)


def evaluate_text_output(content: str) -> float:
    """Heuristic scorer for non-code deliverables (research, creative writing).
    Not a quality judgment — just prevents unfairly zeroing every attempt
    the way the old global benchmark did. Structure/length as weak proxies."""
    word_count = len(content.split())
    has_structure = bool(re.search(r"(?m)^(#|\d+\.|-)\s", content))
    score = min(word_count / 400, 1.0) * 0.7 + (0.3 if has_structure else 0.0)
    return round(score, 4)


EVALUATOR_REGISTRY: Dict[str, Callable[[str], Optional[float]]] = {
    "matrix_ops": evaluate_matrix_code,
}


def score_variation(task: str, task_kind: str, content: str) -> tuple[float, Optional[float]]:
    """Returns (score, measured_benchmark_time_or_None)."""
    clean_code = _extract_code(content)

    evaluator = EVALUATOR_REGISTRY.get(task_kind)
    if evaluator:
        result = evaluator(clean_code)
        if result is not None:
            return result, result  # for code evaluators score doubles as reported metric

    if task_kind == "code":
        score = evaluate_generic_code(clean_code)
        return score, None

    if task_kind == "text":
        return evaluate_text_output(content), None

    # Unknown kind: don't silently zero it — fall back to generic code check,
    # then text heuristic, whichever looks more applicable.
    looks_like_code = "def " in content or "import " in content
    if looks_like_code:
        return evaluate_generic_code(clean_code), None
    return evaluate_text_output(content), None


# =========================================================================
# TOOLS
# =========================================================================

def recall_lineage(task: str) -> str:
    """Retrieve top performing past attempts and optimization hypotheses for a task."""
    attempts = memory.best_attempts(task, limit=5)
    if not attempts:
        return "No prior attempts found for this task."
    return "\n\n".join(
        f"ID: {a['id']} | Score: {a['score']} | Hypothesis: {a.get('hypothesis', 'N/A')}\n"
        f"Content Preview: {a['content'][:150]}..."
        for a in attempts
    )


def inspect_parent_solution(attempt_id: str) -> str:
    """Retrieve the full code, hypothesis, and evaluation score of a specific attempt by ID."""
    a = memory.get_attempt_by_id(attempt_id)
    if not a:
        return f"Attempt {attempt_id} not found."
    return (
        f"Attempt ID: {a['id']}\nScore: {a['score']}\nHypothesis: {a.get('hypothesis', 'N/A')}\n"
        f"Full Code / Content:\n{a['content']}"
    )


def evaluate_and_verify_candidate(code: str, test_input: str = "") -> str:
    """Execute and verify candidate code in a sandbox before recording it.
    This does NOT compute the final fitness score — record_variation does that,
    using a real measured benchmark, never a self-reported number."""
    clean_code = _extract_code(code)
    try:
        scope: Dict[str, Any] = {}
        stdout = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(stdout):
            exec(clean_code, scope)
            if "solution" in scope and callable(scope["solution"]):
                result = scope["solution"](test_input) if test_input else scope["solution"]()
                duration = time.perf_counter() - start
                out = [f"VERIFICATION_SUCCESS: Execution output: {result}", f"Execution time: {duration:.5f}s"]
                captured = stdout.getvalue().strip()
                if captured:
                    out.append(f"Stdout: {captured[:500]}")
                return "\n".join(out)
        duration = time.perf_counter() - start
        out = [f"VERIFICATION_SUCCESS: Code executed cleanly. Setup time: {duration:.5f}s"]
        captured = stdout.getvalue().strip()
        if captured:
            out.append(f"Stdout: {captured[:500]}")
        return "\n".join(out)
    except Exception as exc:
        return f"VERIFICATION_FAILED: Runtime error: {type(exc).__name__}: {str(exc)}"


def record_variation(task: str, task_kind: str, content: str, hypothesis: str, parent_id: str = "") -> str:
    """Commit an evaluated variation into the lineage memory graph.
    task_kind must be one of: 'matrix_ops', 'code', 'text'.
    Score is ALWAYS computed here from a real evaluator — never accepted from the model."""
    score, bench_time = score_variation(task, task_kind, content)
    aid = memory.save_attempt(task=task, content=content, hypothesis=hypothesis, score=score, parent_id=parent_id or None)
    if bench_time is not None:
        return f"Variation recorded with ID {aid}. Measured benchmark time {bench_time:.5f}s -> fitness score {score:.4f}."
    return f"Variation recorded with ID {aid}. Fitness score {score:.4f} (task_kind={task_kind})."


def check_progress(task: str) -> str:
    """Check if evolutionary search fitness has plateaued or stagnated for a given task."""
    if memory.is_stagnating(task):
        return (
            "STAGNATING: Recent scores have plateaued. "
            "Instruct the variation operator to switch optimization strategies or explore an alternative lineage branch."
        )
    return "SEARCH_HEALTHY: Improvements or active exploration detected. Continue current lineage evolution."


# =========================================================================
# SUBAGENTS
# =========================================================================

variation_operator_subagent: Dict[str, Any] = {
    "name": "agentic-variation-operator",
    "description": (
        "Autonomous variation operator performing the Inspect-Plan-Implement-Evaluate loop. "
        "Formulates mutation hypotheses, writes code edits, verifies execution, and records results. "
        "Use for evolutionary search and algorithmic optimization tasks."
    ),
    "system_prompt": (
        "You are an Agentic Variation Operator (AVO). Your role is to evolve solutions via a 4-step loop, "
        "ONE variation per invocation — you will be called again for further iterations, do not simulate them yourself:\n"
        "1. INSPECT: Call recall_lineage and inspect_parent_solution to understand top-performing ancestors.\n"
        "2. PLAN: Formulate an explicit hypothesis describing what modification will yield performance gains.\n"
        "3. IMPLEMENT: Author the complete, real, runnable code variant. Never describe a change you didn't actually write.\n"
        "4. EVALUATE & SELF-REPAIR: Call evaluate_and_verify_candidate to test correctness. If execution fails, fix the code.\n"
        "Once verified, call record_variation with task_kind='matrix_ops' (or 'code' if not the matrix task). "
        "The system computes the real fitness score by executing your code — you cannot set or guess the score yourself. "
        "Your final text response must only report what record_variation actually returned, not an invented summary."
    ),
    "tools": [recall_lineage, inspect_parent_solution, evaluate_and_verify_candidate, record_variation],
}

coding_subagent: Dict[str, Any] = {
    "name": "coding-agent",
    "description": (
        "Expert software engineering agent for code analysis, refactoring, debugging, "
        "implementation, and architecture tasks. Use for any coding-related work."
    ),
    "system_prompt": (
        "You are an expert software engineer.\n"
        "1. UNDERSTAND the task. 2. PLAN discrete steps. 3. IMPLEMENT real code. "
        "4. VERIFY with evaluate_and_verify_candidate — fix failures before proceeding. 5. REVIEW.\n"
        "Record your final solution using record_variation with task_kind='code'. "
        "The system computes the fitness score from real execution — never state a score yourself."
    ),
    "tools": [recall_lineage, inspect_parent_solution, evaluate_and_verify_candidate, record_variation],
}

research_subagent: Dict[str, Any] = {
    "name": "research-agent",
    "description": (
        "Deep research agent for investigating topics, analyzing documents, comparing approaches, "
        "and producing structured research summaries. Use for any research, analysis, or investigation task."
    ),
    "system_prompt": (
        "You are a thorough research analyst. SCOPE the question, INVESTIGATE, ANALYZE, SYNTHESIZE a structured summary. "
        "Start by calling recall_lineage to check prior research on this topic. "
        "Record your findings with record_variation using task_kind='text'."
    ),
    "tools": [recall_lineage, record_variation],
}

story_writing_subagent: Dict[str, Any] = {
    "name": "story-writer",
    "description": (
        "Creative writing agent for stories, narratives, world-building, character development, "
        "and iterative draft refinement. Use for any creative or narrative writing task."
    ),
    "system_prompt": (
        "You are a skilled creative writer. CONCEPT, OUTLINE, DRAFT, REVISE, POLISH. "
        "Record each draft iteration using record_variation with task_kind='text'."
    ),
    "tools": [record_variation],
}

supervisor_prompt = """
You are the Supervisor Orchestrator. You manage specialized subagents for different domains.

Subagent Selection:
- agentic-variation-operator: Evolutionary search, algorithmic optimization, iterative code improvement.
- coding-agent: Software engineering, code analysis, refactoring, debugging, implementation.
- research-agent: Deep research, document analysis, topic investigation, structured summaries.
- story-writer: Creative writing, narratives, world-building, character development.

Execution Workflow for EACH call you receive (this is always exactly ONE iteration):
1. Call check_progress EXACTLY ONCE to inspect whether prior work has stagnated.
2. If stagnation is detected, instruct the subagent to pivot strategy in your delegation message.
3. Delegate to exactly ONE subagent, exactly ONCE, for ONE variation.
4. Immediately report back what the subagent's record_variation tool actually returned, then STOP.

Hard limits — violating these wastes compute and is treated as an error:
- Never call check_progress more than once per call.
- Never delegate to more than one subagent per call.
- Never call any tool again after the subagent has returned its result — just report it.

Do not perform tasks directly; always delegate to the appropriate subagent.
"""

ollama_model_name = os.environ.get("OLLAMA_MODEL", "gemma4:e2b-mlx")
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

local_llm = ChatOllama(model=ollama_model_name, base_url=ollama_base_url, temperature=0.0)

agent = create_deep_agent(
    model=local_llm,
    tools=[check_progress],
    subagents=[variation_operator_subagent, coding_subagent, research_subagent, story_writing_subagent],
    system_prompt=supervisor_prompt,
    checkpointer=InMemorySaver(),
)


# =========================================================================
# HARNESS — fixes bug #3: iterations are now externally enforced. The
# harness checks the attempt count in Neo4j after each call; if the model
# claims "done" without actually recording a new attempt, it is re-prompted
# (up to max_retries) instead of being trusted.
# =========================================================================

def run_harness(task_description: str, task_kind: str = "matrix_ops", iterations: int = 3, max_retries_per_iter: int = 2) -> Dict[str, Any]:
    harness_logger = HarnessLogger(task_description, iterations)
    harness_logger.print_header()

    # NOTE: no single shared thread_id here anymore. Each attempt below gets its
    # own fresh thread, so context never compounds across iterations. Long-term
    # memory lives in Neo4j (via LineageMemory), not in LangGraph checkpoint state.

    completed = 0
    attempt_before = len(memory.get_all_attempts(task_description))

    while completed < iterations:
        retries = 0
        recorded_this_round = False

        while retries <= max_retries_per_iter and not recorded_this_round:
            count_before = len(memory.get_all_attempts(task_description))

            instruction = (
                f"Task: {task_description}. Task kind: {task_kind}. "
                f"Run exactly ONE evolutionary search iteration (iteration {completed + 1} of {iterations}). "
                f"You MUST call record_variation with task_kind='{task_kind}' exactly once, then stop. "
                f"Do not call check_progress or delegate more than once. "
                f"Do not describe a result without having actually called record_variation."
            )
            if retries > 0:
                instruction += (
                    " REMINDER: your previous attempt did not call record_variation with real code. "
                    "Actually implement and record a variation this time — a description alone is not sufficient."
                )

            # Fresh thread per attempt: prevents the full prior transcript from being
            # replayed and re-processed on every call (the cause of the 9k -> 65k
            # token blowup and the model self-chaining extra delegations/checks).
            attempt_thread_id = str(uuid.uuid4())
            attempt_config = {
                "configurable": {"thread_id": attempt_thread_id},
                "recursion_limit": 12,  # hard cap: model cannot loop indefinitely inside one call
            }

            result = None
            try:
                for event in agent.stream(
                    {"messages": [{"role": "user", "content": instruction}]},
                    config=attempt_config,
                ):
                    harness_logger.log_event(event)
                    result = event
            except Exception as exc:
                harness_logger.log_event({"warning": f"Attempt errored (likely hit recursion_limit): {exc}"})

            count_after = len(memory.get_all_attempts(task_description))
            if count_after > count_before:
                recorded_this_round = True
            else:
                retries += 1

        completed += 1
        if not recorded_this_round:
            harness_logger.log_event({"warning": f"Iteration {completed} failed to record a real variation after {max_retries_per_iter} retries."})

    # Ground truth summary — read directly from memory, not from the model's narration.
    final_attempts = memory.get_all_attempts(task_description)
    new_attempts = final_attempts[attempt_before:]
    best = memory.best_attempts(task_description, limit=1)

    summary = {
        "task": task_description,
        "iterations_requested": iterations,
        "attempts_recorded_this_run": len(new_attempts),
        "best_score": best[0]["score"] if best else None,
        "best_attempt_id": best[0]["id"] if best else None,
    }
    harness_logger.log_event({"final_summary": summary})
    return summary


if __name__ == "__main__":
    initial_task = "Optimize an algorithm for computing running matrix operations"
    summary = run_harness(initial_task, task_kind="matrix_ops", iterations=3)
    print("\n--- Ground-truth summary (read from Neo4j, not model narration) ---")
    print(summary)
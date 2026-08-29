import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


console = Console()


def _to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content) if content is not None else ""


def _progress_bar(score: float, width: int = 10) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def _extract_token_counts(msg: AIMessage) -> Dict[str, int]:
    usage = msg.usage_metadata or {}
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }


def _extract_duration(msg: AIMessage) -> Optional[float]:
    metadata = msg.response_metadata or {}
    total_ns = metadata.get("total_duration")
    if total_ns and isinstance(total_ns, (int, float)):
        return total_ns / 1_000_000_000
    return None


def _extract_model_name(msg: AIMessage) -> str:
    metadata = msg.response_metadata or {}
    model = metadata.get("model_name") or metadata.get("model") or "unknown"
    return str(model)


def _short_id(full_id: str) -> str:
    if not full_id:
        return ""
    return full_id[:8]


class HarnessLogger:
    def __init__(self, task: str, iterations: int) -> None:
        self.task = task
        self.iterations = iterations
        self.start_time = time.time()
        self.total_tokens = 0

    def print_header(self) -> None:
        header = Text()
        header.append("AVO Harness", style="bold cyan")
        header.append(" — Evolutionary Search\n", style="dim")
        header.append(f"Task: {self.task}\n", style="white")
        header.append(f"Iterations: {self.iterations}", style="white")
        console.print(Panel(header, border_style="cyan", padding=(0, 2)))
        console.print()

    def log_event(self, event: Dict[str, Any]) -> None:
        if "warning" in event:
            console.print(f"  ⚠️  [bold red]{event['warning']}[/bold red]\n")
            return
        for key, value in event.items():
            if not isinstance(value, dict):
                continue
            messages = value.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    self._log_ai_message(msg)
                elif isinstance(msg, ToolMessage):
                    self._log_tool_message(msg)

    def _log_ai_message(self, msg: AIMessage) -> None:
        model = _extract_model_name(msg)
        duration = _extract_duration(msg)
        tokens = _extract_token_counts(msg)
        self.total_tokens += tokens["total"]

        if msg.tool_calls:
            for tc in msg.tool_calls:
                self._log_tool_call(tc, model, duration, tokens)
        else:
            content_str = _to_str(msg.content)
            if content_str:
                self._log_final_response(content_str, model, duration, tokens)

    def _log_tool_call(
        self,
        tool_call: Union[Dict[str, Any], ToolCall],
        model: str,
        duration: Optional[float],
        tokens: Dict[str, int],
    ) -> None:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})

        if name == "task":
            subagent_type = args.get("subagent_type", "unknown")
            description = args.get("description", "")

            icon = self._subagent_icon(subagent_type)
            console.print(f"  🧠 [bold cyan]Supervisor[/bold cyan]  [dim]({model})[/dim]")
            console.print(f"     → Delegating to: [bold yellow]{icon} {subagent_type}[/bold yellow]")

            if len(description) > 120:
                description = description[:117] + "..."
            console.print(f"     → Task: [italic]\"{description}\"[/italic]")

            if duration:
                console.print(f"     [dim]⏱ {_format_duration(duration)} │ {tokens['total']:,} tokens[/dim]")
            console.print()

        elif name == "check_progress":
            task_name = args.get("task", "")
            console.print(f"  📊 [bold]Progress Check[/bold] [dim]task=\"{task_name}\"[/dim]")

        else:
            console.print(f"  🔧 [bold]Tool Call:[/bold] {name}")
            if args:
                for k, v in args.items():
                    val_str = str(v)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    console.print(f"     {k}: {val_str}")
            console.print()

    def _log_tool_message(self, msg: ToolMessage) -> None:
        content = _to_str(msg.content)
        name = getattr(msg, "name", None) or ""

        if name == "task":
            self._log_subagent_result(content)
        elif name == "check_progress":
            if "STAGNATING" in content:
                console.print(f"     ⚠️  [yellow]{content}[/yellow]")
            else:
                console.print(f"     ✅ [green]{content}[/green]")
            console.print()
        else:
            if len(content) > 200:
                content = content[:197] + "..."
            console.print(f"  📨 [dim]{name}:[/dim] {content}")
            console.print()

    def _log_subagent_result(self, content: str) -> None:
        lines = content.strip().split("\n")

        table_rows = self._parse_table_rows(lines)

        if table_rows:
            for row in table_rows:
                iteration = row.get("iteration", "?")
                hypothesis = row.get("hypothesis", "N/A")
                strategy = row.get("strategy", "N/A")
                score_str = row.get("score", "0")
                variation = row.get("variation", "N/A")

                try:
                    score = float(score_str)
                except (ValueError, TypeError):
                    score = 0.0

                bar = _progress_bar(score)
                console.print(f"     🔧 [bold]Iteration {iteration}[/bold]")
                console.print(f"        Hypothesis: [italic]{hypothesis}[/italic]")
                console.print(f"        Strategy:   {strategy}")
                console.print(f"        Score:      {score:.2f} {bar}")
                console.print(f"        Variation:  [dim]{_short_id(variation)}[/dim]")
                console.print()
        else:
            conclusion_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("|") and not stripped.startswith("---"):
                    conclusion_lines.append(stripped)

            if conclusion_lines:
                summary = "\n".join(conclusion_lines[:10])
                console.print(Panel(summary, title="Subagent Result", border_style="green", padding=(0, 1)))
                console.print()

    def _parse_table_rows(self, lines: List[str]) -> List[Dict[str, str]]:
        table_lines = [l for l in lines if l.strip().startswith("|") and "---" not in l]

        if len(table_lines) < 2:
            return []

        headers_raw = [h.strip() for h in table_lines[0].split("|") if h.strip()]
        header_map: Dict[int, str] = {}
        for i, h in enumerate(headers_raw):
            h_lower = h.lower()
            if "iteration" in h_lower:
                header_map[i] = "iteration"
            elif "hypothesis" in h_lower:
                header_map[i] = "hypothesis"
            elif "strategy" in h_lower or "implementation" in h_lower:
                header_map[i] = "strategy"
            elif "score" in h_lower or "fitness" in h_lower:
                header_map[i] = "score"
            elif "variation" in h_lower or "id" in h_lower:
                header_map[i] = "variation"

        rows = []
        for line in table_lines[1:]:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            row: Dict[str, str] = {}
            for i, cell in enumerate(cells):
                if i in header_map:
                    clean = cell.strip("* ")
                    row[header_map[i]] = clean
            if row:
                rows.append(row)

        return rows

    def _log_final_response(
        self,
        content: str,
        model: str,
        duration: Optional[float],
        tokens: Dict[str, int],
    ) -> None:
        self.total_tokens += tokens["total"]
        elapsed = time.time() - self.start_time

        summary = Text()
        summary.append("✅ Search Complete\n", style="bold green")

        content_lines = content.strip().split("\n")
        for line in content_lines:
            stripped = line.strip()
            if stripped.startswith("###") or stripped.startswith("**"):
                summary.append(f"{stripped}\n", style="bold")
            elif stripped.startswith("|"):
                summary.append(f"{stripped}\n", style="white")
            elif stripped:
                summary.append(f"{stripped}\n", style="white")

        summary.append(f"\nModel: {model}", style="dim")
        if duration:
            summary.append(f" │ Step: {_format_duration(duration)}", style="dim")
        summary.append(f" │ Total: {_format_duration(elapsed)}", style="dim")
        summary.append(f" │ Tokens: {self.total_tokens:,}", style="dim")

        console.print(Panel(summary, border_style="green", padding=(0, 2)))

    def _subagent_icon(self, subagent_type: str) -> str:
        icons = {
            "agentic-variation-operator": "🧬",
            "coding-agent": "💻",
            "research-agent": "🔍",
            "story-writer": "✍️",
            "general-purpose": "🔧",
        }
        return icons.get(subagent_type, "🤖")


def format_result(output: Dict[str, Any], task: str = "", iterations: int = 0) -> None:
    logger = HarnessLogger(task, iterations)

    messages = output.get("messages", [])
    if not messages:
        console.print("[yellow]No messages in output.[/yellow]")
        return

    for msg in messages:
        if isinstance(msg, AIMessage):
            logger._log_ai_message(msg)
        elif isinstance(msg, ToolMessage):
            logger._log_tool_message(msg)

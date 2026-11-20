"""Markdown Parser and Specification Extractors for AGENTS.md and SKILL.md.

Extracts structured instructions, personas, rules, and skill definitions
from standard agent markdown files, preparing them for Ahead-of-Time (AOT)
tensor cartridge compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union


@dataclass
class AgentSpec:
    """Structured representation of an agent specification from AGENTS.md / CLAUDE.md."""
    name: str
    instruction_text: str
    description: str = ""
    rules: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillSpec:
    """Structured representation of an agent skill from SKILL.md."""
    name: str
    description: str
    procedure_text: str
    tools_required: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_yaml_frontmatter(content: str) -> tuple[Dict[str, str], str]:
    """Extracts simple YAML frontmatter (key: value) from the head of markdown content."""
    frontmatter: Dict[str, str] = {}
    body = content

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        fm_block = match.group(1)
        body = content[match.end() :].strip()
        for line in fm_block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                frontmatter[key.strip()] = val.strip().strip("'\"")

    return frontmatter, body


def parse_agent_markdown(source: Union[str, Path], name: Optional[str] = None) -> AgentSpec:
    """Parses an AGENTS.md or CLAUDE.md file into an AgentSpec.
    
    Supports:
    - Standard markdown headings (# Instructions, ## Rules, etc.)
    - Optional frontmatter
    - Automatic name extraction from first H1 if not provided
    """
    if isinstance(source, Path) or (isinstance(source, str) and ("\n" not in source and Path(source).is_file())):
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        source_path = str(path.resolve())
        default_name = path.stem.replace(".", "_")
    else:
        text = str(source)
        source_path = None
        default_name = "agent"

    frontmatter, body = _extract_yaml_frontmatter(text)

    # Determine agent name
    agent_name = name or frontmatter.get("name")
    if not agent_name:
        h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if h1_match:
            agent_name = re.sub(r"[^\w\s-]", "", h1_match.group(1)).strip().lower().replace(" ", "-")
        else:
            agent_name = default_name

    desc = frontmatter.get("description", "")
    if not desc:
        # First non-empty paragraph after H1
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and not p.strip().startswith("#")]
        if paragraphs:
            desc = paragraphs[0][:150]

    # Extract high-priority bullet rules if present
    rules: List[str] = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")) and len(line) > 5:
            rules.append(line[2:].strip())

    return AgentSpec(
        name=agent_name,
        instruction_text=text.strip(),
        description=desc or f"Compiled agent guidelines from {agent_name}",
        rules=rules,
        source_path=source_path,
        metadata={"frontmatter": frontmatter},
    )


def parse_skill_markdown(source: Union[str, Path], name: Optional[str] = None) -> SkillSpec:
    """Parses a SKILL.md file into a SkillSpec.
    
    Follows industry-standard SKILL.md format:
    ---
    name: git-commit-reviewer
    description: Inspects git diffs and ensures commit messages follow conventional commits
    ---
    # Instructions / Procedure Body
    """
    if isinstance(source, Path) or (isinstance(source, str) and ("\n" not in source and Path(source).is_file())):
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        source_path = str(path.resolve())
        default_name = path.parent.name if path.name.lower() == "skill.md" else path.stem
    else:
        text = str(source)
        source_path = None
        default_name = "custom-skill"

    frontmatter, body = _extract_yaml_frontmatter(text)

    skill_name = name or frontmatter.get("name")
    if not skill_name:
        h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if h1_match:
            skill_name = re.sub(r"[^\w\s-]", "", h1_match.group(1)).strip().lower().replace(" ", "-")
        else:
            skill_name = default_name

    desc = frontmatter.get("description", "")
    if not desc:
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and not p.strip().startswith("#")]
        if paragraphs:
            desc = paragraphs[0][:150]
        else:
            desc = f"Compiled procedure for {skill_name}"

    # Extract required tools if specified
    tools: List[str] = []
    tools_val = frontmatter.get("tools")
    if tools_val:
        tools = [t.strip() for t in tools_val.split(",") if t.strip()]

    return SkillSpec(
        name=skill_name,
        description=desc,
        procedure_text=text.strip(),
        tools_required=tools,
        source_path=source_path,
        metadata={"frontmatter": frontmatter},
    )

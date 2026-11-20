"""Unit tests for AGENTS.md and SKILL.md markdown parsing."""

import pytest
from pathlib import Path
import tempfile

from kv_cartridge.markdown import (
    parse_agent_markdown,
    parse_skill_markdown,
    _extract_yaml_frontmatter,
)


def test_extract_yaml_frontmatter():
    text = """---
name: my-cool-agent
description: A helpful assistant
tools: git, bash
---
# Main Content
Here is the body.
"""
    fm, body = _extract_yaml_frontmatter(text)
    assert fm["name"] == "my-cool-agent"
    assert fm["description"] == "A helpful assistant"
    assert fm["tools"] == "git, bash"
    assert body.startswith("# Main Content")


def test_parse_agent_markdown_from_string():
    md = """# Autonomous Coding Agent

You are a senior staff engineer.

## Rules
- Always run tests before committing.
- Never leak secrets.
- Use strict typing.
"""
    spec = parse_agent_markdown(md)
    assert spec.name == "autonomous-coding-agent"
    assert "senior staff engineer" in spec.description
    assert len(spec.rules) == 3
    assert "Always run tests before committing." in spec.rules


def test_parse_skill_markdown_with_frontmatter():
    md = """---
name: pr-reviewer
description: Reviews pull requests for style and correctness
tools: git_diff, post_comment
---
# Instructions
Follow these steps:
1. Check changes.
2. Report bugs.
"""
    spec = parse_skill_markdown(md)
    assert spec.name == "pr-reviewer"
    assert spec.description == "Reviews pull requests for style and correctness"
    assert spec.tools_required == ["git_diff", "post_comment"]
    assert "Follow these steps:" in spec.procedure_text


def test_parse_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_file = Path(tmpdir) / "AGENTS.md"
        agent_file.write_text("# Test Bot\n\n- Rule 1\n- Rule 2\n", encoding="utf-8")
        spec = parse_agent_markdown(agent_file)
        assert spec.name == "test-bot"
        assert len(spec.rules) == 2
        assert spec.source_path == str(agent_file.resolve())


if __name__ == "__main__":
    test_extract_yaml_frontmatter()
    test_parse_agent_markdown_from_string()
    test_parse_skill_markdown_with_frontmatter()
    test_parse_from_file()
    print("PASS: test_markdown_parser")

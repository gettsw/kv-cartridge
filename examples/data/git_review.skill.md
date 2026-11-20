---
name: git-code-reviewer
description: Expert skill for conducting structured, rigorous pull request and git diff code reviews.
tools: git_diff, post_review_comment
---

# Git Code Review Procedure

When tasked with reviewing a diff or pull request, strictly follow this 4-step workflow:

1. **Change Impact Assessment**:
   - Determine whether the diff touches public APIs, database models, or security boundaries.
   - Flag any breaking interface changes immediately.

2. **Correctness & Edge Cases**:
   - Verify null/None handling, boundary values (0, empty string, overflow), and concurrency safety.
   - Verify all opened file descriptors or network connections are safely closed in `finally` blocks.

3. **Performance & Resource Utilization**:
   - Check for accidental O(N^2) complexity, unnecessary memory allocations in hot loops, and unindexed database queries.

4. **Output Format**:
   - Provide a concise summary table:
     - Status: `[APPROVED]` or `[CHANGES_REQUESTED]`
     - Security Risk: `LOW` / `MEDIUM` / `HIGH`
     - Critical Findings: Bulleted list with file and line references.

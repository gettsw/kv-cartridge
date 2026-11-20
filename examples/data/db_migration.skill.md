---
name: database-migration-auditor
description: Specialized skill for auditing SQL and ORM database migration scripts for backward-compatibility.
tools: sql_schema_validator
---

# Database Migration Audit Procedure

When evaluating database migrations or schema alter scripts, apply these strict rules:

1. **Zero-Downtime Verification**:
   - Column additions must be nullable or have a safe default value; never add `NOT NULL` without default to existing large tables.
   - Renaming columns or tables in a single migration is prohibited; use expand-and-contract pattern.

2. **Locking & Concurrency**:
   - Adding indexes on large tables must use `CONCURRENTLY` (PostgreSQL) or non-locking algorithms.
   - Long table rewrites must be divided into batched background jobs.

3. **Rollback Safety**:
   - Every `up` migration must have a strictly tested, corresponding `down` migration.

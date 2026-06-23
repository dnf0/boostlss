# Python Bindings Algorithm Parameter Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the `Algorithm::NonCyclic` functionality to Python users via an `algorithm` kwarg in the `BoostLssModel` constructor.

## Task 1: Add algorithm field and update constructor

- [ ] **Step 1:** In `crates/boostlss-py/src/model.rs`, add `algorithm: String` to the `BoostLssModel` struct.
- [ ] **Step 2:** Update the `#[new]` implementation for `BoostLssModel` to accept `algorithm` (default `"cyclic"`).
- [ ] **Step 3:** Validate that `algorithm` is either `"cyclic"` or `"noncyclic"`, returning a `PyValueError` otherwise. Save it to `self.algorithm`.

## Task 2: Update `fit` to support noncyclic

- [ ] **Step 1:** In `crates/boostlss-py/src/model.rs` inside the `fit` method, for each arm of the `match self.family` block, import `boostlss::engine::Algorithm` and `boostlss::engine::noncyclical::fit_noncyclic`.
- [ ] **Step 2:** Instead of immediately calling `fit_cyclical`, check `self.algorithm.as_str()`.
- [ ] **Step 3:** If `"cyclic"`, continue to use `fit_cyclical`.
- [ ] **Step 4:** If `"noncyclic"`, configure the model with `.algorithm(boostlss::engine::Algorithm::NonCyclic)` and use `fit_noncyclic`.

## Task 3: Update `cvrisk` to support noncyclic

- [ ] **Step 1:** In `crates/boostlss-py/src/model.rs` inside the `cvrisk` method, ensure we configure the `BoostLss` model with `.algorithm(boostlss::engine::Algorithm::NonCyclic)` if `self.algorithm == "noncyclic"`. (The `CvRisk` struct handles the fitting internally, so we just need to set the algorithm on the model instance before passing it to `CvRisk::new`).

## Task 4: Add Python Tests

- [ ] **Step 1:** In `crates/boostlss-py/tests/test_basic.py`, add a test `test_noncyclic_fit()` that initializes a model with `algorithm="noncyclic"` and successfully calls `.fit()`.
- [ ] **Step 2:** Verify tests pass: `maturin develop && pytest crates/boostlss-py/tests`

## Task 5: Commit changes

- [ ] Commit the changes with message `feat: support noncyclic algorithm in python bindings`.

# Project: int-stt-lab

## Team Roles and Procedures

### Nemo (Senior Software Engineer / Engineering Supervisor)
- Owns bounded engineering work from inspection through implementation, validation, and handoff.
- Distinguishes evidence from inference, changes strategy when evidence shows an approach is failing.
- Asks for additional research or authority only when needed.
- For substantial external research, delegates to Ada.
- Prepares concise handoff for Morgan when work is reviewable (objective, changed files, validation, decisions, risks, remaining work).
- Evaluates review findings on merits, not treating review as automatic approval.

### Ada (Researcher)
- Conducts substantial external research or evidence gathering when requested by Nemo.
- Provides findings that Nemo treats as hypotheses until supported by evidence.

### Morgan (Reviewer)
- Reviews work prepared by Nemo using the concise handoff.
- Provides findings that Nemo evaluates on their merits.

## Collaboration Surfaces
- **Primary Communication**: This repository (AGENTS.md, issue tracker, pull requests).
- **Documentation**: Requirements, design, and meeting notes live in the `docs/` directory.
- **Code**: Source code lives in the `src/` directory.
- **Reviews**: Conducted via pull requests; Morgan reviews Nemo's work. Additionally, Morgan can review architecture and experimental methodology via review records stored in `docs/reviews/`.
- **Research Outputs**: Ada's research outputs are stored in the `research/` directory. Ada's research handoff must contain: sources, distinction between evidence and inference, confidence levels and uncertainties, constraints discovered, and recommendations/questions for Nemo.

## Workflow
1. **Research Phase** (if needed): Ada gathers evidence and shares with Nemo.
2. **Engineering Phase**: Nemo drafts requirements, designs, implements, validates.
3. **Review Phase**: Nemo prepares handoff for Morgan; Morgan reviews; Nemo incorporates feedback.
4. **Iteration**: Repeat as necessary until requirements are solid.

## Conventions
- Commit messages: Conventional Commits (https://www.conventionalcommits.org/)
- Branching: `main` branch is protected; feature branches for development.
- Pull Requests: Required for all changes to `main`; must include review by Morgan (or designated reviewer).
- Issue Tracking: Use GitHub Issues for tasks, bugs, and research items.

## Initial Setup
- Repository initialized at: `~/ortluk-hub/int-stt-lab`
- Default branch: `main`

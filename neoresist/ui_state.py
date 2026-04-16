from __future__ import annotations


def section_style(visible: bool) -> dict[str, str]:
    return {"display": "block"} if visible else {"display": "none"}


def resolve_view_state(nav: str | None, ui_mode: str | None) -> dict[str, object]:
    current = nav or "Overview"
    is_expert = str(ui_mode or "Simple") == "Expert"
    if not is_expert:
        simple_overview = current in {"Overview", "Advanced Strategies"}
        return {
            "show_filters": False,
            "note": "Simple mode: upload, ranked answers, tier badges, and evidence cards.",
            "advanced_simple_note": False,
            "advanced_section": False,
            "overview_section": simple_overview,
            "patients_section": False,
            "upload_section": current == "Upload",
            "cases_section": False,
            "about_section": False,
            "pipeline_status_section": False,
            "scatter_card": simple_overview,
            "simple_hero_shell": True,
            "expert_hero_shell": False,
            "expert_methodology_shell": False,
            "expert_strategy_shell": False,
            "simple_answer_shell": simple_overview,
            "expert_diagnostics_shell": False,
            "expert_two_panel_shell": False,
            "expert_extra_diagnostics_shell": False,
        }
    # In Simple mode, treat Advanced Strategies as Overview so core plots never
    # disappear behind Expert-only navigation state.
    effective = "Overview" if (current == "Advanced Strategies" and not is_expert) else current
    show_filters = effective in {"Overview", "Patients"}
    if current in {"Upload", "Cases"}:
        note = "Case workflow mode: keep the left rail simple, use the case panel for module toggles and reruns, and open Advanced Strategies only when you want to tune scoring."
    elif current == "Advanced Strategies":
        note = "Modularity workspace: compare strategies, inspect consensus, and edit safe weights in Expert mode."
    else:
        note = "Cohort mode: use the left filters to explore the cohort, then open Cases or Advanced Strategies when you need deeper workflow controls."
    return {
        "show_filters": show_filters,
        "note": note,
        "advanced_simple_note": False,
        "advanced_section": current == "Advanced Strategies" and is_expert,
        "overview_section": effective == "Overview",
        "patients_section": effective == "Patients",
        "upload_section": effective == "Upload",
        "cases_section": effective == "Cases",
        "about_section": effective == "About",
        "pipeline_status_section": effective == "Pipeline Status",
        "scatter_card": effective == "Overview",
        "simple_hero_shell": not is_expert,
        "expert_hero_shell": is_expert,
        "expert_methodology_shell": is_expert,
        "expert_strategy_shell": is_expert,
        "simple_answer_shell": effective == "Overview",
        "expert_diagnostics_shell": effective == "Overview" and is_expert,
        "expert_two_panel_shell": effective == "Overview" and is_expert,
        "expert_extra_diagnostics_shell": effective == "Overview" and is_expert,
    }

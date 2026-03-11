#!/usr/bin/env python3
"""
Fix: Add missing feedback methods to 01_market_analysis.py
The patch added the __init__ call but the method bodies didn't get inserted.
"""
import os
import ast

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_market_analysis.py")

with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# Check what's actually missing
has_load = "def _load_feedback_context" in content
has_get = "def _get_feedback_prompt_section" in content
has_init_call = "self.feedback_context = self._load_feedback_context()" in content

print(f"  _load_feedback_context method: {'EXISTS' if has_load else 'MISSING'}")
print(f"  _get_feedback_prompt_section method: {'EXISTS' if has_get else 'MISSING'}")
print(f"  __init__ call: {'EXISTS' if has_init_call else 'MISSING'}")

methods_to_add = ""

if not has_load:
    methods_to_add += '''
    def _load_feedback_context(self) -> Dict:
        """Load AI feedback context from Module 09 (Signal Intelligence)."""
        feedback_path = os.path.join(self.script_dir, "data", "ai_feedback_context.json")
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path, 'r') as f:
                    ctx = json.load(f)
                # Check freshness (use if less than 48 hours old)
                ts = ctx.get("generated", "")
                if ts:
                    try:
                        parsed = datetime.fromisoformat(ts)
                        age_hours = (datetime.now() - parsed).total_seconds() / 3600
                        if age_hours < 48:
                            return ctx
                    except Exception:
                        return ctx
            except Exception:
                pass
        return {}

'''

if not has_get:
    methods_to_add += '''    def _get_feedback_prompt_section(self) -> str:
        """Build prompt section from Signal Intelligence feedback (Module 09)."""
        if not hasattr(self, 'feedback_context') or not self.feedback_context:
            return ""
        
        sections = []
        
        # Performance report card
        perf = self.feedback_context.get("performance_report", "")
        if perf and "Insufficient" not in perf:
            sections.append(perf)
        
        # Regime-specific constraints
        regime_prompts = self.feedback_context.get("regime_prompt_injection", {})
        if regime_prompts and self.regime_context:
            current_regime = self.regime_context.get("regime", "sideways").lower()
            matched = False
            for key in regime_prompts:
                if key in current_regime or current_regime in key:
                    sections.append(regime_prompts[key])
                    matched = True
                    break
            if not matched and "sideways" in regime_prompts:
                sections.append(regime_prompts["sideways"])
        
        # Pattern library
        pattern_text = self.feedback_context.get("pattern_prompt_injection", "")
        if pattern_text and "insufficient" not in pattern_text.lower():
            sections.append(pattern_text)
        
        # Post-mortem insights
        pm_insights = self.feedback_context.get("postmortem_insights", "")
        if pm_insights:
            sections.append(pm_insights)
        
        if not sections:
            return ""
        
        return "\\n\\n".join(sections) + "\\n"

'''

if not methods_to_add:
    print("\n  Both methods already exist. Nothing to do.")
else:
    # Insert before _get_regime_prompt_section
    marker = "    def _get_regime_prompt_section(self)"
    if marker in content:
        content = content.replace(marker, methods_to_add + marker)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n  Inserted missing method(s) before _get_regime_prompt_section")
    else:
        # Fallback: insert before _get_portfolio_prompt_section
        marker2 = "    def _get_portfolio_prompt_section(self)"
        if marker2 in content:
            content = content.replace(marker2, methods_to_add + marker2)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\n  Inserted missing method(s) before _get_portfolio_prompt_section")
        else:
            print("\n  ERROR: Could not find insertion point. Add methods manually.")

# Verify syntax
try:
    with open(target, "r", encoding="utf-8") as f:
        ast.parse(f.read())
    print("  Syntax OK!")
except SyntaxError as e:
    print(f"  Syntax error at line {e.lineno}: {e.msg}")
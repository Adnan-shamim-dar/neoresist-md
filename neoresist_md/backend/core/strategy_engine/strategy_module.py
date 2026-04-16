from __future__ import annotations

from ..base_module import BaseModule, ModuleResult


class StrategyModule(BaseModule):
    module_id = "strategy_engine"
    display_name = "Strategy Engine"

    def run(self, table, context):
        return ModuleResult(self.module_id, "complete", "Strategy consensus placeholder.", table=table)


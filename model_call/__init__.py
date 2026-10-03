"""model-call: structured output from model APIs that still works on a bad day.

from model_call import Client, Price, Budget
client = Client("anthropic", "claude-haiku-4-5", price=Price(1.0, 5.0, 0.1), budget=Budget(5.0), log="calls.jsonl")
res = client.structured(system, user, schema)
if res.ok: use(res.value)
"""

from .budget import Budget, BudgetExceeded, Price
from .client import Client, Result, examine
from .ledger import InDoubt, NotApplied, ToolLedger
from .log import CallLog
from .pacing import Pacer
from .repair import Repaired, repair
from .schema import SchemaError, validate
from .transport import AuthError, BalanceError, ModelCallError, NotFoundError, RetryPolicy

__version__ = "0.1.0"
__all__ = [
    "AuthError",
    "BalanceError",
    "Budget",
    "BudgetExceeded",
    "CallLog",
    "Client",
    "InDoubt",
    "ModelCallError",
    "NotApplied",
    "NotFoundError",
    "Pacer",
    "Price",
    "Repaired",
    "Result",
    "RetryPolicy",
    "SchemaError",
    "ToolLedger",
    "examine",
    "repair",
    "validate",
]

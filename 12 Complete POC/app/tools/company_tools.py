"""
app/tools/company_tools.py — Agent Tools

[Concept: Tool Calling System]

────────────────────────────────────────────────────────────────
HOW LLM TOOL CALLING WORKS
────────────────────────────────────────────────────────────────
Modern LLMs (Claude, GPT-4) can decide to use tools rather than
answering from memory. Here's the flow:

  1. You give the LLM a list of available tools with descriptions
  2. The LLM reads the user's query
  3. If it determines a tool is needed, it outputs a structured
     "tool call" (JSON) instead of a regular text answer:
       {"tool": "calculate_bonus", "input": {"salary": 50000, "rate": 0.10}}
  4. Your code executes the actual function
  5. The result is fed back to the LLM as an "observation"
  6. The LLM generates a final natural-language answer

This is different from RAG in a key way:
  - RAG: search documents, use text as context
  - Tools: execute code/APIs, get structured data back

Example decision:
  Query: "What is the vacation policy?" → Use RAG (document lookup)
  Query: "Calculate bonus for $50,000 salary at 10%" → Use calculator tool
  Query: "Summarize the IT security document" → Use summarize tool

────────────────────────────────────────────────────────────────
THE REASONING LOOP (ReAct Pattern)
────────────────────────────────────────────────────────────────
The agent follows the Thought → Action → Observation loop:

  Thought:     "The user wants to calculate a bonus. I should use
               the calculate_bonus tool, not guess."
  Action:      Call calculate_bonus(salary=50000, rate=0.10)
  Observation: "Bonus amount: $5,000.00"
  Thought:     "I have the result. I can now answer the user."
  Final:       "Based on a salary of $50,000 with a 10% bonus rate,
               your bonus would be $5,000.00."

This explicit reasoning loop makes agent behavior auditable.
────────────────────────────────────────────────────────────────
"""

from langchain_core.tools import tool
from typing import Optional, List
from app.observability.logger import log_tool_use, logger


# ──────────────────────────────────────────────
# Tool 1: Bonus Calculator
# ──────────────────────────────────────────────

@tool
def calculate_bonus(salary: float, bonus_rate: float) -> str:
    """
    Calculate an employee's bonus amount based on their salary and bonus rate.

    Use this tool when the user asks to calculate a bonus amount.
    Do NOT use this for general questions about bonus policy.

    Args:
        salary:     The employee's annual salary in dollars (e.g., 50000)
        bonus_rate: The bonus rate as a decimal (e.g., 0.10 for 10%)

    Returns:
        A formatted string with the calculated bonus amount.
    """
    if salary <= 0:
        return "Error: Salary must be a positive number."
    if bonus_rate < 0 or bonus_rate > 1:
        return "Error: Bonus rate must be between 0 and 1 (e.g., 0.10 for 10%)."

    bonus_amount = salary * bonus_rate
    tax_estimate = bonus_amount * 0.25  # Rough 25% tax estimate

    result = (
        f"Bonus Calculation:\n"
        f"  Annual Salary:    ${salary:,.2f}\n"
        f"  Bonus Rate:       {bonus_rate * 100:.1f}%\n"
        f"  Gross Bonus:      ${bonus_amount:,.2f}\n"
        f"  Est. Tax (25%):   ${tax_estimate:,.2f}\n"
        f"  Net Bonus (est.): ${bonus_amount - tax_estimate:,.2f}\n"
        f"\nNote: Tax estimate is approximate. Consult HR for exact figures."
    )

    log_tool_use("system", "calculate_bonus", f"salary={salary}, rate={bonus_rate}", result[:80])
    return result


# ──────────────────────────────────────────────
# Tool 2: Employee Policy Lookup
# ──────────────────────────────────────────────

# Hardcoded policy summaries — in production this would query a database
# Each entry includes an "access_level" field that controls who can see it:
#   "public"       → all employees
#   "manager"      → managers and admins only
#   "confidential" → admins only
POLICY_DATABASE = {
    "vacation": {
        "access_level": "public",
        "content": (
            "Vacation Policy Summary:\n"
            "- Full-time employees: 15 days/year\n"
            "- After 5 years: 20 days/year\n"
            "- After 10 years: 25 days/year\n"
            "- Days accrue monthly (1.25 days/month for 15-day policy)\n"
            "- Max carryover: 5 days to next year\n"
            "- Must be approved by manager 2 weeks in advance\n"
            "For full details see the HR Handbook."
        ),
    },
    "sick_leave": {
        "access_level": "public",
        "content": (
            "Sick Leave Policy Summary:\n"
            "- 10 sick days per year (all employees)\n"
            "- Doctor's note required for absences > 3 consecutive days\n"
            "- Sick days do not carry over year-to-year\n"
            "- Can be used for family member illness\n"
            "For full details see the HR Handbook."
        ),
    },
    "remote_work": {
        "access_level": "public",
        "content": (
            "Remote Work Policy Summary:\n"
            "- Employees may work remotely up to 3 days per week\n"
            "- Must be in office on Tuesdays and Thursdays (core days)\n"
            "- Home office equipment reimbursement: up to $500/year\n"
            "- Internet stipend: $50/month\n"
            "For full details see the Remote Work Policy document."
        ),
    },
    "maternity_paternity": {
        "access_level": "public",
        "content": (
            "Parental Leave Policy Summary:\n"
            "- Primary caregiver: 16 weeks paid leave\n"
            "- Secondary caregiver: 6 weeks paid leave\n"
            "- Applies to birth, adoption, and foster care\n"
            "- Must give 30 days notice when possible\n"
            "For full details see the Parental Leave Policy document."
        ),
    },
    "performance_review": {
        "access_level": "public",
        "content": (
            "Performance Review Policy Summary:\n"
            "- Annual reviews in December\n"
            "- Mid-year check-in in June\n"
            "- 360-degree feedback from peers, managers, direct reports\n"
            "- Rating scale: 1 (Needs Improvement) to 5 (Exceptional)\n"
            "For full details see the Performance Management Handbook."
        ),
    },
    "compensation_bands": {
        "access_level": "manager",
        "content": (
            "Compensation Bands Summary (Manager Only):\n"
            "- L1 (Junior):   $60,000 – $80,000\n"
            "- L2 (Mid):      $80,000 – $110,000\n"
            "- L3 (Senior):   $110,000 – $150,000\n"
            "- L4 (Staff):    $150,000 – $200,000\n"
            "For full details see the Compensation Policy document."
        ),
    },
}


@tool
def lookup_employee_policy(policy_name: str) -> str:
    """
    Look up a specific HR policy by name.

    Use this tool when the user asks for a QUICK SUMMARY of a specific
    named policy. For detailed policy questions, use RAG search instead.

    If you are unsure which policies exist, call list_available_documents
    first to discover valid policy names before calling this tool.

    Args:
        policy_name: The name or partial name of the policy (e.g., "vacation", "sick leave")

    Returns:
        A formatted policy summary string.
    """
    # NOTE: This @tool function is called by execute_policy_tool() in knowledge_agent.py,
    # which performs the permission check BEFORE invoking this function.
    # The allowed_keys argument is injected by the executor — it contains only
    # the policy keys the user's role is permitted to see.
    policy_key = policy_name.lower().strip().replace(" ", "_")

    # Try exact match first
    if policy_key in POLICY_DATABASE:
        result = POLICY_DATABASE[policy_key]["content"]
        log_tool_use("system", "lookup_employee_policy", policy_name, result[:80])
        return result

    # Try partial match
    for key in POLICY_DATABASE:
        if policy_key in key or key in policy_key:
            result = POLICY_DATABASE[key]["content"]
            log_tool_use("system", "lookup_employee_policy", policy_name, result[:80])
            return result

    available = ", ".join(POLICY_DATABASE.keys())
    return (
        f"Policy '{policy_name}' not found in the quick-lookup database.\n"
        f"Available policies: {available}\n"
        f"For other policies, please ask a specific question and I'll search the documents."
    )


# ──────────────────────────────────────────────
# Tool 3: Document Summary
# ──────────────────────────────────────────────

DOCUMENT_SUMMARIES = {
    "hr_handbook": {
        "access_level": "public",
        "content": (
            "HR Handbook Summary:\n"
            "The HR Handbook covers all people policies at the company including:\n"
            "- Employee onboarding and orientation\n"
            "- Compensation and benefits overview\n"
            "- Leave policies (vacation, sick, parental, bereavement)\n"
            "- Code of conduct and anti-harassment policy\n"
            "- Performance management process\n"
            "- Disciplinary procedures\n"
            "- Termination and offboarding\n"
            "Total: 12 sections, last updated January 2024."
        ),
    },
    "it_security": {
        "access_level": "public",
        "content": (
            "IT Security Policy Summary:\n"
            "The IT Security Policy governs how employees use company systems:\n"
            "- Password requirements (12+ chars, changed every 90 days)\n"
            "- Acceptable use of company computers and networks\n"
            "- Data classification (Public, Internal, Confidential, Restricted)\n"
            "- Incident reporting procedures\n"
            "- Remote access and VPN requirements\n"
            "- Software installation policy\n"
            "Total: 8 sections, last updated March 2024."
        ),
    },
    "finance_policy": {
        "access_level": "manager",
        "content": (
            "Finance Policy Summary:\n"
            "The Finance Policy covers financial procedures for employees:\n"
            "- Expense reimbursement (submit within 30 days, receipts required)\n"
            "- Travel policy (economy class, approved hotels, per diem rates)\n"
            "- Purchase authorization limits by role\n"
            "- Budget approval process\n"
            "- Vendor payment terms\n"
            "Total: 6 sections, last updated February 2024."
        ),
    },
    "exec_compensation": {
        "access_level": "confidential",
        "content": (
            "Executive Compensation Summary (Confidential):\n"
            "- Executive salary bands and equity grants\n"
            "- Annual bonus structure for VP and above\n"
            "- Long-term incentive plan details\n"
            "Total: 4 sections, last updated January 2024."
        ),
    },
}


@tool
def summarize_document(document_name: str) -> str:
    """
    Get a high-level summary of a specific company document.

    Use this tool when the user explicitly asks to 'summarize' or get an
    'overview' of a document. Pass the document name as mentioned by the user.

    If you are unsure which documents exist, call list_available_documents
    first to discover valid names before calling this tool.

    Args:
        document_name: The name or partial name of the document to summarize

    Returns:
        A structured summary of the document's main topics.
    """
    # NOTE: Permission check is performed by execute_summarize_tool() in
    # knowledge_agent.py before this function is invoked. The document_name
    # passed here has already been validated against the user's access level.
    doc_key = document_name.lower().strip().replace(" ", "_")

    if doc_key in DOCUMENT_SUMMARIES:
        result = DOCUMENT_SUMMARIES[doc_key]["content"]
        log_tool_use("system", "summarize_document", document_name, result[:80])
        return result

    # Try partial match
    for key in DOCUMENT_SUMMARIES:
        if doc_key in key or key in doc_key:
            result = DOCUMENT_SUMMARIES[key]["content"]
            log_tool_use("system", "summarize_document", document_name, result[:80])
            return result

    available = ", ".join(DOCUMENT_SUMMARIES.keys())
    return (
        f"Document '{document_name}' not found.\n"
        f"Available documents for summary: {available}"
    )


# ──────────────────────────────────────────────
# Tool 4: Document & Policy Discovery
# ──────────────────────────────────────────────

@tool
def list_available_documents(allowed_levels: List[str]) -> str:
    """
    List company documents and HR policies the current user is allowed to see.

    Use this tool when:
    - The user asks what documents or policies are available
    - The user wants to summarize 'all documents' or 'everything'
    - You are about to call summarize_document or lookup_employee_policy
      but are unsure whether the document/policy name the user mentioned exists
    - summarize_document or lookup_employee_policy returned 'not found'

    Args:
        allowed_levels: List of access levels the user is permitted to see.
                        Provided automatically by the system — do not ask the
                        user for this value.

    Returns:
        A filtered list of documents and policies visible to this user.
    """
    lines = ["Available company documents and policies:\n"]

    # Only include entries whose access_level is within the user's allowed levels.
    # This prevents existence-leaking — users never see documents they can't access.
    visible_docs = {k: v for k, v in DOCUMENT_SUMMARIES.items() if v["access_level"] in allowed_levels}
    visible_policies = {k: v for k, v in POLICY_DATABASE.items() if v["access_level"] in allowed_levels}

    lines.append("── Documents (use with: summarize_document) ──")
    if visible_docs:
        for key in visible_docs:
            lines.append(f"  • {key}")
    else:
        lines.append("  (none available)")

    lines.append("\n── HR Policies (use with: lookup_employee_policy) ──")
    if visible_policies:
        for key in visible_policies:
            lines.append(f"  • {key}")
    else:
        lines.append("  (none available)")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────

ALL_TOOLS = [
    calculate_bonus,
    lookup_employee_policy,
    summarize_document,
    list_available_documents,
]

TOOL_NAMES = {tool.name: tool for tool in ALL_TOOLS}

from typing import TypedDict, List, Dict, Any, Literal
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI # 예시 LLM
from dotenv import load_dotenv
import os

# --- 1. Agent State Definition (논문의 'Agent Test-time Scaling' 문제 공식화 및 자원 관리) ---
# Agent의 현재 상태를 정의합니다. 이는 논문에서 정의한 에이전트의 '예산(budget)',
# '계획(plan)', '실행 궤적(trajectory)', '소모된 자원(realized cost)' 등의 개념을 나타냅니다.
class AgentState(TypedDict):
    """
    LangGraph 워크플로우 전반에 걸쳐 예산 인식 에이전트의 현재 상태를 나타냅니다.
    이는 논문에서 정의한 'Agent Test-time Scaling' 문제의 공식화 및 자원 관리 개념에 해당합니다.
    """
    question: str
    chat_history: List[BaseMessage] # LLM과의 대화 기록 (컨텍스트 관리)
    plan: str  # 논문의 'Planning Module'에서 관리하는 트리 구조 계획을 간략화한 문자열
    budget_status: Dict[str, int]  # 논문의 'Budget Tracker'에서 제공하는 실시간 예산 정보 (used, total)
    trajectory: List[str]  # 에이전트의 '추론 궤적(reasoning trajectory)' 기록
    current_answer: str  # 에이전트가 제안한 현재 답변
    tool_results: List[str] # 툴 호출 결과 기록
    unified_cost: float # 논문의 'Unified Cost Metric' (토큰 + 툴 호출 비용)
    attempts: int # 답변 시도 횟수 (BATS의 'Attempt K' 개념, 여기서는 PIVOT 시 증가)
    max_attempts: int # 최대 시도 횟수
    final_verified_answers: List[str] # 검증을 통과한 최종 답변들
    # 다음 행동을 제안받기 위한 필드
    next_action_type: Literal["tool", "answer", "stop", "reason"] # agent_reasoning_node의 결정
    proposed_tool_call: Dict[str, Any] # tool_invocation_node에서 사용할 툴 호출 정보
    verification_decision: Literal["SUCCESS", "CONTINUE", "PIVOT", "NONE"] # verify_node의 결정

# --- 2. Tool Definitions (논문의 'Problem Instantiation with Search Agent'에 명시된 툴) ---
# 논문에서 'Search' 및 'Browse' 툴을 언급합니다. 여기서는 실제 API 호출 대신 더미 함수를 사용합니다.
class SearchToolArgs(BaseModel):
    query: str = Field(description="웹 검색을 위한 쿼리 문자열.")

class BrowseToolArgs(BaseModel):
    url: str = Field(description="방문할 웹페이지의 URL.")
    goal: str = Field(description="웹페이지 탐색을 위한 구체적인 정보 목표.")

# 더미 툴 함수
def search_tool_mock(query: str) -> str:
    """주어진 쿼리에 대해 웹 검색을 시뮬레이션합니다."""
    print(f"--- Calling Search Tool with query: {query} ---")
    if "LangGraph workflow" in query:
        return "Search result for Python LangGraph: LangGraph is a library for building stateful, multi-actor applications."
    elif "budget-aware agents" in query:
        return "Search result for budget-aware agents: Agents that strategically manage their computational resources like tool calls or tokens."
    return f"Search result for '{query}': No specific information found."

def browse_tool_mock(url: str, goal: str) -> str:
    """주어진 URL을 탐색하는 것을 시뮬레이션합니다."""
    print(f"--- Calling Browse Tool on URL: {url} for goal: {goal} ---")
    if "langgraph.readthedocs.io" in url:
        return "Browse result for LangGraph docs: Provides detailed documentation on building graphs."
    return f"Browse result for '{url}': Content not available or could not be browsed for '{goal}'."

# LLM 모델 초기화 (Gemini 계열). 실제 호출 시 .env의 GOOGLE_API_KEY 사용.
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.0-pro",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 3. Graph Nodes (논문의 'BATS Framework'의 모듈에 대응) ---

# 3.1. Think & Plan Node (논문의 'Budget-Aware Planning' 및 'Budget Tracker' 통합)
def think_and_plan_node(state: AgentState) -> Dict[str, Any]:
    """
    이 노드는 BATS 프레임워크의 'Thinking & Planning' 모듈을 나타냅니다.
    Budget Tracker 정보를 프롬프트에 포함하여 예산 인식을 통합합니다.

    논문의 다음 부분에 해당합니다:
    -   Figure 1 (bottom)의 'Think + Plan' 모듈.
    -   Section 3.1 'Budget Tracker'가 지속적인 예산 인식을 제공합니다.
    -   Section 4.1 'Budget-Aware Planning'의 제약 조건 분해 및 구조화된 동적 계획.
    -   LLM이 남은 예산(HIGH/MEDIUM/LOW/CRITICAL)에 따라 전략을 동적으로 조정합니다.
    """
    question = state['question']
    current_plan = state['plan']
    budget_info = state['budget_status']
    history = state['chat_history']
    trajectory = state['trajectory']

    search_rem = budget_info.get('search_total', 0) - budget_info.get('search_used', 0)
    browse_rem = budget_info.get('browse_total', 0) - budget_info.get('browse_used', 0)

    # 예산 상태에 따른 전략 가이드 (논문의 C.1, C.2 섹션의 프롬프트를 통합)
    budget_guidance = ""
    total_search = budget_info.get('search_total', 0)
    total_browse = budget_info.get('browse_total', 0)
    # 예산 비율 기준으로 HIGH / MEDIUM / LOW / CRITICAL 분류
    if (search_rem >= total_search * 0.7) and (browse_rem >= total_browse * 0.7):
        budget_guidance = "HIGH Budget: 폭넓게 탐색하여 빠르게 컨텍스트를 구축하세요. 여러 쿼리를 배치로 실행할 수 있습니다."
    elif ((search_rem >= total_search * 0.3 and search_rem < total_search * 0.7) or \
          (browse_rem >= total_browse * 0.3 and browse_rem < total_browse * 0.7)):
        budget_guidance = "MEDIUM Budget: 핵심 지식 격차를 해소하기 위해 정확하고 세련된 쿼리/탐색에 집중하세요."
    elif ((search_rem >= total_search * 0.1 and search_rem < total_search * 0.3) or \
          (browse_rem >= total_browse * 0.1 and browse_rem < total_browse * 0.3)):
        budget_guidance = "LOW Budget: 단 하나의 중요한 사실을 검증하거나 답변을 확정하는 데 집중하세요."
    else:  # CRITICAL
        budget_guidance = "CRITICAL Budget: 예산이 거의 소진되었습니다. 고갈된 도구 사용을 피하고, 필수적인 경우에만 최소 비용 쿼리/탐색을 수행하세요. 불확실성이 남았는데 도구 사용이 불가능하면 답변으로 None을 출력하세요."

    prompt_content = f"""
    You are an AI reasoner with Google Search and Browsing tools. Solve the question by iterating:
    think, tool_code, tool_response, answer.

    ## Tools
    You have access to the following tools: search (for web search), browse (for visiting URLs).

    ## Current Budget Status
    - Search Budget: Used: {budget_info.get('search_used', 0)}, Remaining: {search_rem} / {budget_info.get('search_total', 0)}
    - Browse Budget: Used: {budget_info.get('browse_used', 0)}, Remaining: {browse_rem} / {budget_info.get('browse_total', 0)}

    ## Budget-Aware Strategy:
    {budget_guidance}

    ## About Planning (C.2 Planning Module in BATS)
    Maintain a tree-structured checklist of actionable steps.
    - Mark each step with its status: [ ] pending, [x] done, [!] failed, [~] partial.
    - Use numbered branches (1.1, 1.2) to represent alternative paths or candidate leads.
    - Log resource usage after execution: (Query=#, URL=#).
    - Keep all executed steps, never delete them, retain history to avoid repeats.
    - Update dynamically as you reason and gather info, adding or revising steps as needed.
    - Always consider current and remaining budget when updating the plan.

    ## Previous Trajectory (Summarized for context management if needed):
    {state['trajectory'][-1] if state['trajectory'] else "None"}

    ## Current Question: {question}
    ## Current Plan: {current_plan if current_plan else "No plan yet. Formulate an initial plan."}
    ## Previous Tool Results: {state['tool_results'][-1] if state['tool_results'] else "None"}

    Based on the above, what is your next action?
    Think step-by-step. If you decide to call a tool, output a JSON object in <tool_code> tags.
    If you have a final answer, output it in <answer> tags.
    If you cannot proceed due to budget or lack of information, output <stop>.
    """

    # LLM을 호출하여 생각, 계획 업데이트 및 다음 툴 호출/답변/중단 제안을 받습니다.
    # 여기서는 LLM이 논문의 프롬프트에 따라 적절한 JSON 형식으로 응답한다고 가정합니다.
    # 실제 구현에서는 LLM 출력을 파싱하는 복잡한 로직이 필요합니다.
    print(f"\n--- LLM Thinking for: {question} ---")
    print(f"Current budget: Search Rem {search_rem}, Browse Rem {browse_rem}")

    # 가상의 LLM 추론 및 결정 로직
    # 실제 LLM 호출 대신 질문과 예산 상태에 따라 다음 행동을 시뮬레이션
    llm_thought = f"LLM is thinking based on budget and question: '{question}'."
    updated_plan = current_plan # 계획은 LLM이 생성하지만, 여기선 단순화를 위해 업데이트하지 않음
    next_action_type = "stop"
    proposed_tool_call = None
    final_answer = ""

    if search_rem > 0 and "LangGraph workflow" in question and not any("LangGraph is a library" in res for res in state['tool_results']):
        llm_thought = f"남은 검색 예산을 활용하여 '{question}'에 대한 초기 정보를 검색합니다."
        proposed_tool_call = {"name": "search", "arguments": SearchToolArgs(query="LangGraph workflow").dict()}
        next_action_type = "tool"
    elif browse_rem > 0 and any("LangGraph is a library" in res for res in state['tool_results']) and not any("detailed documentation" in res for res in state['tool_results']):
        llm_thought = f"검색 결과에서 LangGraph 정보를 확인했습니다. 남은 탐색 예산을 사용하여 'langgraph.readthedocs.io'를 탐색하여 자세한 문서를 확인합니다."
        proposed_tool_call = {"name": "browse", "arguments": BrowseToolArgs(url="https://langgraph.readthedocs.io", goal="detailed documentation").dict()}
        next_action_type = "tool"
    elif any("detailed documentation" in res for res in state['tool_results']):
        final_answer = "LangGraph는 상태 저장, 다중 액터 애플리케이션을 구축하기 위한 라이브러리이며, 자세한 문서는 langgraph.readthedocs.io에서 찾을 수 있습니다."
        llm_thought = f"충분한 정보를 수집했습니다. 최종 답변을 제공합니다: {final_answer}"
        next_action_type = "answer"
    else:
        llm_thought = "더 이상 도구를 사용할 예산이 없거나, 명확한 다음 단계가 없습니다. 작업을 중단합니다."
        next_action_type = "stop"

    return {
        "chat_history": history + [HumanMessage(content=llm_thought)],
        "plan": updated_plan,
        "next_action_type": next_action_type,
        "proposed_tool_call": proposed_tool_call,
        "current_answer": final_answer, # 답변이 결정되면 여기에 저장
        "trajectory": state['trajectory'] + [llm_thought] # 추론 궤적 업데이트
    }

# 3.2. Tool Call Node (논문의 'acting' via tool calls)
def call_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    이 노드는 제안된 도구 호출을 실행하고 에이전트의 상태를 업데이트합니다.

    논문의 다음 부분에 해당합니다:
    -   Figure 1의 'Tool Call' 모듈.
    -   에이전트 스케일링의 'acting' (도구 호출을 통한 행동) 측면.
    -   도구 소비량에 따라 'budget_status'를 업데이트합니다.
    """
    proposed_tool_call = state['proposed_tool_call']
    budget_info = state['budget_status']
    tool_results = state['tool_results']
    unified_cost = state['unified_cost']

    if not proposed_tool_call:
        return {"tool_results": tool_results, "unified_cost": unified_cost, "next_action_type": "reason"}

    tool_name = proposed_tool_call['name']
    tool_args = proposed_tool_call['arguments']

    current_search_used = budget_info.get('search_used', 0)
    current_browse_used = budget_info.get('browse_used', 0)
    search_total = budget_info.get('search_total', 0)
    browse_total = budget_info.get('browse_total', 0)

    result_message = ""
    cost_per_call = 0.001 # 논문의 A.1 섹션에 정의된 툴 호출 비용을 시뮬레이션

    if tool_name == "search":
        if current_search_used < search_total:
            response = search_tool_mock(tool_args['query'])
            result_message = f"Search Tool Output: {response}"
            current_search_used += 1
            unified_cost += cost_per_call
        else:
            result_message = f"Search Tool Output: 검색 예산이 부족하여 호출할 수 없습니다. (남은 예산: {search_total - current_search_used})"
            print(result_message)
    elif tool_name == "browse":
        if current_browse_used < browse_total:
            response = browse_tool_mock(tool_args['url'], tool_args['goal'])
            result_message = f"Browse Tool Output: {response}"
            current_browse_used += 1
            unified_cost += cost_per_call
        else:
            result_message = f"Browse Tool Output: 탐색 예산이 부족하여 호출할 수 없습니다. (남은 예산: {browse_total - current_browse_used})"
            print(result_message)
    else:
        result_message = f"Tool Call Failed: Unknown tool '{tool_name}'."
        print(result_message)

    tool_results.append(result_message)

    updated_budget_status = {
        **budget_info,
        'search_used': current_search_used,
        'search_remaining': search_total - current_search_used,
        'browse_used': current_browse_used,
        'browse_remaining': browse_total - current_browse_used
    }

    return {
        "tool_results": tool_results,
        "unified_cost": unified_cost,
        "chat_history": state['chat_history'] + [HumanMessage(content=f"Tool call: {tool_name} with args {tool_args}. Result: {result_message}")],
        "proposed_tool_call": None, # 툴 호출 후 초기화
        "budget_status": updated_budget_status,
        "next_action_type": "reason", # 툴 호출 후 다시 추론 노드로 돌아가도록 설정
        "trajectory": state['trajectory'] + [f"Tool Call ({tool_name}): {tool_args}, Result: {result_message}"]
    }

# 3.3. Verification Node (논문의 'Budget-Aware Self-verification' 및 컨텍스트 관리)
def verify_node(state: AgentState) -> Dict[str, Any]:
    """
    이 노드는 BATS 프레임워크의 'Verification' 모듈을 나타냅니다.
    현재 답변과 궤적을 평가하고, 전략적 결정(SUCCESS/CONTINUE/PIVOT)을 내리며,
    컨텍스트 관리를 위한 간결한 요약을 생성합니다.

    논문의 다음 부분에 해당합니다:
    -   Figure 1 (bottom)의 'Verification' 모듈.
    -   Section 4.2 'Budget-Aware Self-verification'의 후방 검증, 의사 결정, 요약 생성.
    -   Appendix C.3에 상세한 검증자 프롬프트가 제공됩니다.
    -   Appendix A.2의 '더 지능적인 컨텍스트 관리' (궤적 요약화를 통해).
    """
    question = state['question']
    current_answer = state['current_answer']
    trajectory = state['trajectory']
    budget_info = state['budget_status']
    history = state['chat_history']
    attempts = state['attempts']

    search_rem = budget_info.get('search_total', 0) - budget_info.get('search_used', 0)
    browse_rem = budget_info.get('browse_total', 0) - budget_info.get('browse_used', 0)

    # 논문 C.3 섹션의 프롬프트를 통합하여 LLM에 전달합니다.
    verifier_prompt = f"""
    You are an AI Strategic Verifier. Your primary goal is to evaluate a proposed answer, assess
    the viability of the current problem-solving plan, and decide the best course of action:
    declare success, continue with the current plan, or pivot to a new one.

    ### Given Inputs
    * Question: {question}
    * Trajectory: {trajectory}
    * Current Answer: {current_answer}
    * Budget Status: Search Used: {budget_info.get('search_used', 0)}, Remaining: {search_rem};
                     URL Used: {budget_info.get('browse_used', 0)}, Remaining: {browse_rem}
    * Current Attempts: {attempts} / {state['max_attempts']}

    ### Your Task: A 3-Step Process (as described in Appendix C.3)
    You must proceed in the following order:
    #### Step 1: Conduct Verification Analysis
    First, perform a strict verification of the `Current Answer`.
    * Go through each constraint from the original Question one by one.
    * For each constraint, compare it against the `Current Answer` and the `Trajectory`.
    * State your finding for each constraint: satisfied, contradicted, or unverifiable.
    #### Step 2: Make a Strategic Decision
    Based on your verification and the budget, make one of three decisions.
    1. SUCCESS: If the verification in Step 1 passed (all constraints are satisfied). The task is complete.
    2. CONTINUE: If the verification failed because few constraints are unverifiable, but the
       overall plan is still sound and salvageable. This is the choice if **both** of these
       conditions are true:
       * Promising Path: The `Trajectory` is generally sound, and the failure was due to a
         correctable error.
       * Sufficient Budget: There is enough `Remaining Budget` to attempt a correction on this path.
    3. PIVOT: If the verification failed, signal to abandon the current plan and switch to another one.
       You should pivot if any of these conditions are true:
       * Dead End: The `Trajectory` reveals a fundamental flaw in the current plan's logic that
         cannot be easily fixed.
       * Failed Tool Calls: The Trajectory shows repeated, unsuccessful attempts to find certain info.
       * Insufficient Budget: The `Remaining Budget` is too low to make another meaningful
         attempt or correction within the *current* plan.
    #### Step 3: Summarize for the Next Step
    You need to first provide a **trajectory summary**: Summarize the agent's reasoning trajectory
    into a concise narrative. Explain its initial goal, the logical steps taken, key findings
    and the final conclusion, emphasizing how key findings or contradictions caused the agent
    to change its strategy.
    Then, provide additional details tailored to your decision in Step 2.
    * If the decision is SUCCESS: No further detail needed.
    * If the decision is CONTINUE / PIVOT:
      * Failure Analysis: Diagnose the root cause of the failure.
      * Useful information: Any useful intermediate findings or results.
      * Strategic Recommendations: Provide actionable advice for the agent's next attempt.

    ### **Output Requirement**
    Your final output must be a single JSON object with the following structure.
    {{
    "verification": "Verification analysis text.",
    "decision": "SUCCESS | CONTINUE | PIVOT",
    "justification": "A concise explanation for your strategic decision.",
    "trajectory_summary": "The informative trajectory summary.",
    "details": {{}}
    }}
    """

    # 가상의 LLM 검증 및 결정 로직

    decision = "CONTINUE"
    trajectory_summary = "Current trajectory is summarized. Still need more info."
    new_attempts = state['attempts']
    if "detailed documentation" in "".join(state['tool_results']) and "LangGraph" in question and current_answer:
        decision = "SUCCESS" # 충분한 정보가 있고 답변이 있으면 성공
    elif search_rem <= 0 and browse_rem <= 0:
        decision = "PIVOT" # 예산이 모두 소진되면 피벗 (새로운 시도 또는 종료 고려)
        new_attempts += 1 # 시도 횟수 증가
    # 컨텍스트 관리: 논문 Appendix A.2 처럼 오래된 궤적을 요약으로 대체
    # 여기서는 단순화를 위해 현재 궤적을 요약으로 대체합니다.
    new_trajectory_for_context = [trajectory_summary] if decision != "SUCCESS" else trajectory

    return {

        "trajectory": new_trajectory_for_context, # 컨텍스트 관리를 위해 요약된 궤적으로 업데이트
        "verification_decision": decision, # 그래프 라우팅을 위한 결정
        "chat_history": history + [HumanMessage(content=f"Verification Decision: {decision}. Summary: {trajectory_summary}")],
        "attempts": new_attempts # 시도 횟수 업데이트
    }

def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    이 노드는 현재 상태와 궤적을 기반으로 최종 답변을 생성합니다.
    'Think & Plan' 또는 'Verification' 노드에서 답변이 형성될 수 있다고 판단될 때 트리거됩니다.
    논문의 Figure 1의 'Answer' 블록에 해당합니다.
    """
    question = state['question']
    trajectory = state['trajectory']
    tool_results = state['tool_results']
    current_answer = state['current_answer']

    print("\n--- Generating Final Answer ---")

    if state['verification_decision'] == "SUCCESS" and current_answer:
        final_ans = current_answer
    elif state['attempts'] >= state['max_attempts'] or (
        not current_answer and (
            state['budget_status'].get('search_remaining', 0) <= 0 and
            state['budget_status'].get('browse_remaining', 0) <= 0
        )
    ):
        final_ans = "None"  # 최대 시도 횟수 초과 또는 예산 고갈 시 답변 없음
    else:
        # LLM을 호출하여 최종 답변을 생성합니다. 응답에서 <answer> 태그를 파싱합니다.
        final_answer_prompt = f"""
        Based on the following question, your reasoning trajectory, and tool results, provide the final answer.
        If you are confident, output it within <answer> tags. If you cannot find a definitive answer, state 'None'.
        Question: {question}
        Reasoning Trajectory: {trajectory}
        Tool Results: {tool_results}
        Previous Answer Candidate: {current_answer if current_answer else 'None'}
        Your final answer:
        """
        try:
            response = llm.invoke([HumanMessage(content=final_answer_prompt)])
            content = getattr(response, "content", "")
            # 간단한 태그 파서: <answer>...</answer> 추출
            start = content.find("<answer>")
            end = content.find("</answer>")
            if start != -1 and end != -1 and end > start:
                final_ans = content[start + len("<answer>") : end].strip()
            else:
                # 태그가 없으면 보수적으로 None 처리
                final_ans = "None"
        except Exception:
            # LLM 호출 실패 시 안전한 기본값
            final_ans = "None"
        return {
            "current_answer": final_ans,
            "final_verified_answers": state['final_verified_answers'] + [final_ans],
            "chat_history": state['chat_history'] + [HumanMessage(content=f"Final Answer: {final_ans}")]
        }


def route_from_think_plan(state: AgentState) -> str:
    if state['next_action_type'] == "tool":
        return "call_tool_node"
    elif state['next_action_type'] == "answer":
        return "generate_answer_node"
    elif state['next_action_type'] == "reason":
        # 도구 호출 후 추가 추론/검증 단계로 이동
        return "verify_node"
    else:  # "stop"
        return END

def route_after_verification(state: AgentState) -> str:

    """
    'verify_node'의 검증 결정에 따라 그래프를 라우팅합니다.
    논문의 다음 부분에 해당합니다:
    -   Section 4.2 'Strategic Decision' (SUCCESS, CONTINUE, PIVOT).
    -   Figure 6는 검증 후의 흐름을 보여줍니다.
    """
    decision = state['verification_decision']
    attempts = state['attempts']
    max_attempts = state['max_attempts']
    if decision == "SUCCESS":
        return "generate_answer_node" # 검증 성공 시 최종 답변 생성 및 종료
    elif decision == "CONTINUE":
        return "think_plan_node" # 현재 계획을 계속 진행 (다시 생각/계획)
    elif decision == "PIVOT":
        if attempts < max_attempts:
            return "think_plan_node" # 논문의 'Attempt K' 개념에 따라 새로운 시도 (다시 생각/계획)
        else:
            return "generate_answer_node" # 최대 시도 횟수 초과 시 최종 답변 (또는 None)

    return "think_plan_node" # 기본값 (안전 장치)

def main():
    # 그래프 빌더 생성
    graph_builder = StateGraph(AgentState)

    # 노드 등록 (함수명과 노드명 일치시켜 가독성 향상)
    graph_builder.add_node("think_plan_node", think_and_plan_node)
    graph_builder.add_node("call_tool_node", call_tool_node)
    graph_builder.add_node("verify_node", verify_node)
    graph_builder.add_node("generate_answer_node", generate_answer_node)

    # 간선 구성
    graph_builder.add_conditional_edges(
        "think_plan_node",
        route_from_think_plan,
        {
            "call_tool_node": "call_tool_node",
            "generate_answer_node": "generate_answer_node",
            "verify_node": "verify_node",
            END: END,
        }
    )
    graph_builder.add_conditional_edges(
        "verify_node",
        route_after_verification,
        {
            "generate_answer_node": "generate_answer_node",
            "think_plan_node": "think_plan_node",
        }
    )
    graph_builder.add_edge("call_tool_node", "think_plan_node")  # 툴 호출 후 다시 생각/계획

    # 시작 노드 설정
    graph_builder.set_entry_point("think_plan_node")

    # 그래프 컴파일 (실행 가능한 앱 생성)
    app = graph_builder.compile()
    # 간단한 초기 상태로 테스트 실행 (직접 실행 시 동작)
    initial_state: AgentState = {
        "question": "Explain LangGraph workflow",
        "chat_history": [],
        "plan": "",
        "budget_status": {"search_total": 3, "search_used": 0, "browse_total": 2, "browse_used": 0},
        "trajectory": [],
        "current_answer": "",
        "tool_results": [],
        "unified_cost": 0.0,
        "attempts": 0,
        "max_attempts": 2,
        "final_verified_answers": [],
        "next_action_type": "reason",
        "proposed_tool_call": {},
        "verification_decision": "NONE",
    }
    result = app.invoke(initial_state, config={"recursion_limit": 50})
    print(result)

if __name__ == "__main__":
    main()

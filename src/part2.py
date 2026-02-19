"""
Author: Group 8
Filename: part2.py
Description: Generates natural language explanations for an ai agent's
             decision-making using a hybrid approach. Rule-based templates
             translate formal explanation factors into structured english
             sentences, which are then synthesised into fluent, non-technical
             prose using the Gemini API. (Falls back to rule-based output if
             the API is unavailable.)
"""

import json
import os
import re
import sys
import textwrap
import time
import google.generativeai as genai

try:
    from assignment4 import main as run_assignment4
except ImportError:
    sys.exit(
        "ERROR: assignment4.py not found. "
        "Place it in the same directory as this script."
    )


# Gemini API - go to aigooglestudio.com to make your own and paste it in ""
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") #->add the api here 

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_available = True
    else:
        print(
            "WARNING: No Gemini API key found. "
            "Using rule-based output only.\n"
        )
        gemini_available = False
except Exception:
    print(
        "WARNING: google-generativeai not installed. "
        "Using rule-based output only.\n"
    )
    gemini_available = False


# RB logic -> English sentences
def camel_to_words(camel_case_name):
    """Convert a camelCase or PascalCase name to lowercase readable words.

    Example:
        'getCoffeeKitchen' -> 'get coffee kitchen'
        'AnnInOffice'      -> 'ann in office'
    """
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', camel_case_name)
    spaced = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', spaced)
    return spaced.lower()


def format_belief_list(belief_names):
    """Format a list of belief names into a readable English enumeration.

    Example:
        ['ownCard', 'atKitchen'] -> "'own card' and 'at kitchen'"
    """
    if not belief_names:
        return "no specific conditions"
    readable = [camel_to_words(belief) for belief in belief_names]
    if len(readable) == 1:
        return f"'{readable[0]}'"
    joined = ", ".join(f"'{b}'" for b in readable[:-1])
    return f"{joined} and '{readable[-1]}'"


def format_cost_vector(cost_values, value_dimension_names):
    """Format a cost vector alongside its dimension names.

    Example:
        [5.0, 0.0, 1.0], ['quality', 'price', 'time']
        -> '(quality: 5.0, price: 0.0, time: 1.0)'
    """
    if not cost_values or not value_dimension_names:
        return str(cost_values)
    num_dims = min(len(cost_values), len(value_dimension_names))
    parts = [
        f"{value_dimension_names[i]}: {cost_values[i]}"
        for i in range(num_dims)
    ]
    return "(" + ", ".join(parts) + ")"


def format_norm_as_sentence(norm_expression):
    """Convert a formal norm string into a plain English sentence fragment.

    Example:
        'P(payShop)'-> 'it is prohibited to pay shop'
        'O(gotoShop,gotoKitchen)' ->
            'at least one of the following actions is required: ...'
    """
    match = re.match(r'(P|O)\((.+)\)', norm_expression)
    if not match:
        return norm_expression
    norm_type = match.group(1)
    raw_actions = match.group(2)
    action_words = [
        camel_to_words(action.strip())
        for action in raw_actions.split(",")
    ]
    action_string = " or ".join(f"'{a}'" for a in action_words)
    if norm_type == "P":
        return f"it is prohibited to {action_string}"
    return (
        f"at least one of the following actions is required: {action_string}"
    )


def translate_factor_to_sentence(factor, value_dimension_names):
    """Translate a single formal explanation factor into an english sentence.

    Each factor is a list whose first element is a letter code identifying
    its type. Returns an empty string for unrecognised factor types.
    """
    if not factor:
        return ""

    factor_type = factor[0]

    if factor_type == "P":
        action_name = factor[1]
        satisfied_conditions = factor[2]
        action_words = camel_to_words(action_name)
        if not satisfied_conditions:
            return (
                f"The action '{action_words}' was performed because it "
                f"had no preconditions blocking it."
            )
        condition_string = format_belief_list(satisfied_conditions)
        return (
            f"The action '{action_words}' was performed because the "
            f"required conditions were met: {condition_string}."
        )

    if factor_type == "C":
        chosen_option = factor[1]
        satisfied_conditions = factor[2]
        chosen_words = camel_to_words(chosen_option)
        if not satisfied_conditions:
            return (
                f"The option '{chosen_words}' was selected as no "
                f"specific preconditions were needed."
            )
        condition_string = format_belief_list(satisfied_conditions)
        return (
            f"The option '{chosen_words}' was chosen because the "
            f"necessary conditions were satisfied: {condition_string}."
        )

    if factor_type == "V":
        chosen_option = factor[1]
        chosen_costs = factor[2]
        rejected_option = factor[4]
        rejected_costs = factor[5]
        chosen_words = camel_to_words(chosen_option)
        rejected_words = camel_to_words(rejected_option)
        chosen_cost_string = format_cost_vector(
            chosen_costs, value_dimension_names
        )
        rejected_cost_string = format_cost_vector(
            rejected_costs, value_dimension_names
        )
        return (
            f"'{chosen_words}' {chosen_cost_string} was preferred over "
            f"'{rejected_words}' {rejected_cost_string} because it "
            f"scored better on your priorities."
        )

    if factor_type == "N":
        rejected_option = factor[1]
        norm_expression = factor[2]
        rejected_words = camel_to_words(rejected_option)
        norm_sentence = format_norm_as_sentence(norm_expression)
        return (
            f"The option '{rejected_words}' was ruled out because "
            f"{norm_sentence}."
        )

    if factor_type == "F":
        rejected_option = factor[1]
        missing_conditions = factor[2]
        rejected_words = camel_to_words(rejected_option)
        missing_string = format_belief_list(missing_conditions)
        return (
            f"The option '{rejected_words}' could not be chosen because "
            f"the required conditions were not met: {missing_string}."
        )

    if factor_type == "L":
        source_node = factor[1]
        target_node = factor[3]
        source_words = camel_to_words(source_node)
        target_words = camel_to_words(target_node)
        return (
            f"Performing '{source_words}' directly enables "
            f"'{target_words}' to happen afterwards."
        )

    if factor_type == "D":
        goal_node_name = factor[1]
        goal_words = camel_to_words(goal_node_name)
        return (
            f"This action contributes to achieving the goal: "
            f"'{goal_words}'."
        )

    if factor_type == "U":
        preference_pair = factor[1]
        dimension_names = preference_pair[0]
        priority_order = preference_pair[1]
        priority_names = [dimension_names[i] for i in priority_order]
        priority_string = " > ".join(priority_names)
        return (
            f"Throughout, the agent respected your stated priorities: "
            f"{priority_string} (most important first)."
        )

    return ""


def build_rule_based_explanation(
    action_to_explain,
    formal_explanation_factors,
    value_dimension_names,
):
    """Convert all formal factors into a structured paragraph.

    This makes the input to the Gemini processing step.
    """
    action_words = camel_to_words(action_to_explain)
    sentences = [
        f"You asked why the agent performed '{action_words}'. "
        f"Here is the explanation:\n"
    ]
    for factor in formal_explanation_factors:
        sentence = translate_factor_to_sentence(
            factor, value_dimension_names
        )
        if sentence:
            sentences.append(sentence)
    return "\n".join(sentences)


# Gemini synthesis
GEMINI_SYSTEM_PROMPT = """\
You are an assistant explaining an AI agent's decisions to everyday
users with no technical background. Rewrite the provided sentences as
a single, clear, coherent explanation in plain English.

Rules:
- Always refer to the agent in third person ("the agent"), never "I".
- Write in second person for the user ("for you", "your preferences").
- Never use technical jargon: avoid "preconditions", "OR node",
  "trace", "costs", "score", or "index".
- When explaining why an alternative was rejected due to a rule or
  obligation, clearly state WHAT the rule requires and WHY the
  alternative fails to meet it — do not just say it "wasn't considered".
- When explaining cost comparisons, describe them in plain terms
  (e.g. "free", "takes less time", "higher quality") rather than
  showing raw numbers.
- Preserve ALL information — do not drop any reason.
- Group sentences logically: first why the chosen option was selected,
  then why alternatives were rejected, then what goal this serves.
- Aim for 5-8 sentences. Be concise but complete.
- Do NOT start with "Sure", "Of course", or "You're wondering".
"""


def synthesize_with_gemini(
    action_to_explain,
    rule_based_text,
    scenario_context,
):
    """Send the rule-based sentences to Gemini.
    """
    if not gemini_available:
        return rule_based_text

    action_words = camel_to_words(action_to_explain)
    readable_trace = [
        camel_to_words(node)
        for node in scenario_context.get("trace", [])
    ]
    context_block = (
        f"Goal: {scenario_context.get('goal', [])}\n"
        f"User preferences: {scenario_context.get('preferences', [])}\n"
        f"Norm: {scenario_context.get('norm', {})}\n"
        f"Action being explained: '{action_words}'\n"
        f"Full execution trace: {readable_trace}"
    )
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Structured factual sentences to synthesize:\n"
        f"{rule_based_text}\n\n"
        f"Please write a single, fluent, friendly explanation of why "
        f"the agent performed '{action_words}'."
    )

    try:
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=GEMINI_SYSTEM_PROMPT,
        )
        for attempt in range(3):
            try:
                response = gemini_model.generate_content(user_prompt)
                return response.text.strip()
            except Exception as error:
                rate_limited = "429" in str(error)
                if rate_limited and attempt < 2:
                    wait_seconds = 60
                    print(
                        f"Rate limited. Waiting {wait_seconds}s "
                        f"before retry {attempt + 2}/3..."
                    )
                    time.sleep(wait_seconds)
                else:
                    raise
    except Exception as error:
        print(
            f"WARNING: Gemini call failed ({error}). "
            f"Falling back to rule-based output."
        )
        return rule_based_text


# public API
def generate_explanation(
    goal_tree,
    norm,
    goal,
    initial_beliefs,
    preferences,
    action_to_explain,
):
    """Run the full explanation pipeline for a single scenario. 
        Calls ass. 4 to get the selected trace and formal factors.
        Then translate factors to English via the RB layer.
        Finally synthesizes into fluent explanations via Gemini.

    Returns a tuple of (natural_language_explanation, trace, formal_factors).
    """
    selected_trace, formal_factors = run_assignment4(
        goal_tree,
        norm,
        goal,
        initial_beliefs,
        preferences,
        action_to_explain,
    )

    if not selected_trace:
        action_words = camel_to_words(action_to_explain)
        message = (
            f"The action '{action_words}' was not part of the agent's "
            f"chosen plan, so there is nothing to explain."
        )
        return message, selected_trace, formal_factors

    value_dimension_names = preferences[0] if preferences else []

    rule_based_text = build_rule_based_explanation(
        action_to_explain,
        formal_factors,
        value_dimension_names,
    )

    scenario_context = {
        "goal": goal,
        "preferences": preferences,
        "norm": norm,
        "trace": selected_trace,
    }

    natural_language_explanation = synthesize_with_gemini(
        action_to_explain,
        rule_based_text,
        scenario_context,
    )

    return natural_language_explanation, selected_trace, formal_factors


# formatting - used Gemini to help format it a bit better
LINE_WIDTH = 72


def divider_line(character="-"):
    """Return a full-width divider line using the given character."""
    return character * LINE_WIDTH


def wrap_text(raw_text, width=LINE_WIDTH):
    """Wrap multi-paragraph text to the given line width."""
    wrapped_paragraphs = []
    for paragraph in raw_text.split("\n"):
        if paragraph.strip():
            wrapped_paragraphs.append(
                textwrap.fill(paragraph.strip(), width=width)
            )
        else:
            wrapped_paragraphs.append("")
    return "\n".join(wrapped_paragraphs)


def format_preference_string(preferences):
    """Format a preference pair as a readable priority chain.

    Example:
        [['quality', 'price', 'time'], [1, 2, 0]]
        -> 'price > time > quality (most important first)'
    """
    dimension_names = preferences[0]
    priority_order = preferences[1]
    ordered_names = [dimension_names[i] for i in priority_order]
    return " > ".join(ordered_names) + " (most important first)"


def format_norm_for_display(norm):
    """Format a norm dictionary as a readable display string."""
    norm_type = norm["type"]
    action_list = ", ".join(norm["actions"])
    if norm_type == "P":
        return f"It is prohibited to perform: {action_list}"
    if norm_type == "O":
        return (
            f"At least one of the following must be performed: "
            f"{action_list}"
        )
    return str(norm)


def format_scenario_block(
    scenario_number,
    scenario,
    selected_trace,
    formal_factors,
    natural_language_explanation,
):
    output_lines = []

    output_lines.append(divider_line("="))
    label_upper = scenario["label"].upper()
    output_lines.append(f"SCENARIO {scenario_number}: {label_upper}")
    output_lines.append(divider_line("="))
    output_lines.append("")

    output_lines.append("INPUTS")
    output_lines.append(divider_line("-"))
    output_lines.append(
        f"Goal: {', '.join(scenario['goal'])}"
    )
    output_lines.append(
        f"Norm: {format_norm_for_display(scenario['norm'])}"
    )
    output_lines.append(
        f"Beliefs: {', '.join(scenario['beliefs'])}"
    )
    output_lines.append(
        f"Preferences: "
        f"{format_preference_string(scenario['preferences'])}"
    )
    output_lines.append(
        f"Action to Explain: {scenario['action_to_explain']}"
    )
    output_lines.append("")

    output_lines.append("SELECTED EXECUTION TRACE")
    output_lines.append(divider_line("-"))
    output_lines.append(" -> ".join(selected_trace))
    output_lines.append("")

    output_lines.append("FORMAL EXPLANATION FACTORS")
    output_lines.append(divider_line("-"))
    if formal_factors:
        for factor in formal_factors:
            output_lines.append(str(factor))
    else:
        output_lines.append(
            "(action not in selected trace — no factors generated)"
        )
    output_lines.append("")

    output_lines.append("NATURAL LANGUAGE EXPLANATION")
    output_lines.append(divider_line("-"))
    output_lines.append(wrap_text(natural_language_explanation))
    output_lines.append("")
    output_lines.append(divider_line("="))
    output_lines.append("")

    return "\n".join(output_lines)


# different secanrios to test
SCENARIOS = [
    {
        # Scenario 1: Standard case. payShop is prohibited so the agent
        # must use the kitchen. Preferences: price > time > quality.
        # Kitchen (price 0.0) beats shop (price 3.0).
        # Explains: getCoffeeKitchen.
        "label": (
            "Prohibition on paying at the shop "
            "— explaining kitchen coffee"
        ),
        "norm": {"type": "P", "actions": ["payShop"]},
        "goal": ["haveCoffee"],
        "beliefs": [
            "staffCardAvailable", "ownCard", "colleagueAvailable",
            "haveMoney", "AnnInOffice",
        ],
        "preferences": [["quality", "price", "time"], [1, 2, 0]],
        "action_to_explain": "getCoffeeKitchen",
    },
    {
        # scenario 2: no meaningful norm. Staff card not available so
        # kitchen is blocked. Preferences: price > quality > time.
        # ann's office (price 0.0) beats shop (price 3.0).
        # explains: getCoffeeAnnOffice.
        "label": (
            "No norm, staff card unavailable "
            "— explaining Ann's office coffee"
        ),
        "norm": {"type": "P", "actions": ["nonExistentAction"]},
        "goal": ["haveCoffee"],
        "beliefs": ["AnnInOffice", "haveMoney", "colleagueAvailable"],
        "preferences": [["quality", "price", "time"], [1, 0, 2]],
        "action_to_explain": "getCoffeeAnnOffice",
    },
    {
        # scenario 3: Going to the kitchen is prohibited. Agent chooses
        # between Ann's office and the shop. Preferences: quality first.
        # shop (quality 0.0) beats Ann's office (quality 2.0).
        # explains: getCoffeeShop.
        "label": (
            "Prohibition on going to the kitchen "
            "— explaining shop coffee"
        ),
        "norm": {"type": "P", "actions": ["gotoKitchen"]},
        "goal": ["haveCoffee"],
        "beliefs": [
            "staffCardAvailable", "ownCard", "colleagueAvailable",
            "haveMoney", "AnnInOffice",
        ],
        "preferences": [["quality", "price", "time"], [0, 1, 2]],
        "action_to_explain": "getCoffeeShop",
    },
    {
        # scenario 4: obligation to visit the shop or kitchen.
        # preferences: quality > price > time.
        # shop (quality 0.0) beats all alternatives.
        # obligation is satisfied by visiting the shop.
        # explains: gotoShop.
        "label": (
            "Obligation to visit shop or kitchen, quality-first "
            "— explaining going to the shop"
        ),
        "norm": {"type": "O", "actions": ["gotoShop", "gotoKitchen"]},
        "goal": ["haveCoffee"],
        "beliefs": [
            "staffCardAvailable", "ownCard", "colleagueAvailable",
            "haveMoney", "AnnInOffice",
        ],
        "preferences": [["quality", "price", "time"], [0, 1, 2]],
        "action_to_explain": "gotoShop",
    },
    {
        # scenario 5: Only colleague available (no own card, Ann absent).
        # agent must use colleague's card to access the kitchen.
        # preferences: price > time > quality.
        # explains: getOthersCard.
        "label": (
            "Only colleague available, price-first "
            "— explaining getting a colleague's card"
        ),
        "norm": {"type": "P", "actions": ["nonExistentAction"]},
        "goal": ["haveCoffee"],
        "beliefs": [
            "staffCardAvailable", "colleagueAvailable", "haveMoney",
        ],
        "preferences": [["quality", "price", "time"], [1, 2, 0]],
        "action_to_explain": "getOthersCard",
    },
]

# probably have to change before Pairielearn to standard output later
if __name__ == "__main__":

    json_file_path = "coffee.json"
    try:
        with open(json_file_path, "r") as json_file:
            goal_tree_data = json.load(json_file)
    except FileNotFoundError:
        sys.exit(
            f"ERROR: '{json_file_path}' not found. "
            f"Place coffee.json in the same directory."
        )

    output_file_path = "explanations_output.txt"
    all_output_blocks = []

    document_header = "\n".join([
        divider_line("="),
        "EXPLAINABLE AI (INFOMXAI) — PROJECT 1, PART 2",
        "Natural Language Explanations for Agent Decision-Making",
        divider_line("="),
        "",
    ])
    print(document_header)
    all_output_blocks.append(document_header)

    for scenario_index, current_scenario in enumerate(SCENARIOS, start=1):
        print(
            f"Processing Scenario {scenario_index}: "
            f"{current_scenario['label']} ..."
        )

        explanation, trace, factors = generate_explanation(
            goal_tree=goal_tree_data,
            norm=current_scenario["norm"],
            goal=current_scenario["goal"],
            initial_beliefs=current_scenario["beliefs"],
            preferences=current_scenario["preferences"],
            action_to_explain=current_scenario["action_to_explain"],
        )

        scenario_block = format_scenario_block(
            scenario_number=scenario_index,
            scenario=current_scenario,
            selected_trace=trace,
            formal_factors=factors,
            natural_language_explanation=explanation,
        )

        print(scenario_block)
        all_output_blocks.append(scenario_block)

    full_document = "\n".join(all_output_blocks)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(full_document)

    print(f"Output saved to: {output_file_path}")

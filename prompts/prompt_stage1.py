"""Prompt builders for Stage 1: idea to structured screenplay."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def format_character_roster(characters: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for character in characters:
        section = (
            f"- **{character.get('name', 'Unknown')}**: "
            f"{character.get('age', '?')} years old, "
            f"{character.get('gender', '?')}, "
            f"{character.get('occupation', '?')}. "
            f"Personality: {character.get('personality traits', '?')}. "
            f"Speaking style: {character.get('speaking style', '?')}."
        )
        sections.append(section)
    return "\n".join(sections)


def build_character_prompt(topic: str, *, max_characters: int = 6) -> str:
    return (
        "You are tasked with directing a film based on a provided topic. You "
        "need to brainstorm the main characters in the film and provide a "
        "profile for each character.\n\n"
        f"### Film topic:\n{topic.strip()}\n\n"
        "### Requirements:\n"
        "1. **Profile content**:\n"
        "   - The profile should include the name, gender(male or female), age, "
        "occupation, personality traits and speaking style.\n"
        "   - The name should only have one word.\n\n"
        "2. **Number of characters**:\n"
        f"   - Include no more than {max_characters} main characters.\n\n"
        "Your response should only contain the following JSON content:\n"
        "[{\"name\": \"...\",\n"
        "\"age\": \"...\",\n"
        "\"gender\": \"...\",\n"
        "\"occupation\": \"...\",\n"
        "\"personality traits\": \"...\",\n"
        "\"speaking style\": \"...\"\n"
        "},\n"
        "...\n"
        "]"
    )


def build_scene_prompt(
    topic: str,
    characters: Sequence[Mapping[str, Any]],
    *,
    max_scenes: int = 3,
) -> str:
    roster = format_character_roster(characters)
    return (
        "You are tasked with directing a film based on a provided topic. You "
        "need to plan several coherent scenes according to the topic.\n\n"
        f"### Film Topic:\n{topic.strip()}\n\n"
        "### Main Characters:\n"
        f"{roster}\n\n"
        "### Planning Steps:\n"
        "1. Decide the number of scenes and assign a concise sub-topic to each scene.\n"
        f"   - The number of scenes should be no more than {max_scenes}.\n"
        "2. For each scene, provide ONLY a short location prompt (short_prompt):\n"
        "   - short_prompt: a concise phrase naming the scene type and vibe "
        "(e.g., 'a study desk setup', 'a living room table arrangement', "
        "'a workshop bench with tools', 'a tea ceremony setting', 'a kitchen island display').\n"
        "   - Do NOT provide any long, detailed layout description here.\n"
        "3. Based on the sub-topic, select several characters from the Main Characters for each scene.\n"
        "   - The number of characters selected for each scene should be at least two.\n"
        "   - All the main characters must appear at least once.\n"
        "4. Based on the sub-topic, write a story plot for each scene.\n"
        "   - The story plot must involve only the characters selected in this scene.\n"
        "   - All story plots should be coherent, dramatic, and specific.\n"
        "   - Include enough contextual detail to motivate the interaction and emotional arc.\n"
        "5. Based on the story plot, write a final dialogue goal so that the dialogue between characters "
        "in this scene can end naturally.\n\n"
        "### Output format:\n"
        "[{\n"
        "  \"sub-topic\": \"...\",\n"
        "  \"selected-characters\": [\"...\",\"...\"],\n"
        "  \"selected-location\": {\n"
        "      \"short_prompt\": \"...\"\n"
        "  },\n"
        "  \"story-plot\": \"...\",\n"
        "  \"dialogue-goal\": \"...\"\n"
        "}, ...]\n"
    )


def build_dialogue_prompt(script_outline: str) -> str:
    return (
        "You are tasked as a screenwriter to create specific dialogues based on the provided script outline. "
        "Please use your creativity and understanding of the plot to write vivid dialogues that drive the story forward, "
        "making the script rich and engaging.\n\n"
        "### Script Outline:\n"
        f"{script_outline.strip()}\n\n"
        "### Requirements:\n"
        "   - Keep the dialogue natural, concise, and vivid, avoiding repetition, clichés, and the use of numbers.\n"
        "   - In each scene, the characters participating in the dialogue can only include \"involved characters\" for this scene specified in the Script Outline.\n"
        "   - Each sentence of dialogue should not be too long.\n"
        "   - The number of dialogues in each scene should NOT be too many.\n"
        "   - The dialogue in each scene should ultimately achieve the given dialogue-goal, allowing the scene to end naturally.\n\n"
        "Your response should only contain the following JSON content:\n"
        "[{\"scene-topic\": \"...\",\n"
        "\"scene-plot\": \"...\",\n"
        "\"scene-dialogue\": [{\"speaker\": \"...\", \"content\": \"...\"}, ...]\n"
        "}, ...]\n"
    )


def build_long_prompt_detailer(scenes_json: str, final_dialogue_json: str) -> str:
    return (
        "You are a Scene Detailer collaborating on a film previsualization pipeline. "
        "Your task is to expand concise short_prompts into full, spatially detailed scene descriptions.\n\n"
        "### Input: Scenes (short prompts + story plots)\n"
        f"{scenes_json}\n\n"
        "### Input: Final Dialogue JSON\n"
        f"{final_dialogue_json}\n\n"
        "### Your Task:\n"
        "For EACH scene, write a single paragraph of vivid English description (`long_prompt`) "
        "that elaborates on the spatial layout and arrangement of the scene.\n\n"
        "### Requirements:\n"
        "- Focus ONLY on describing spatial relationships among objects or groups of objects.\n"
        "- Use clear relative position terms: in front of, behind, next to, on top of, under, to the left/right of, at the center of, along the wall, surrounding, etc.\n"
        "- DO NOT list materials, colors, or textures in detail; just describe placement and relation.\n"
        "- Mention all major elements implied by the short_prompt and dialogue context.\n"
        "- Avoid listing objects numerically; describe naturally as a single continuous paragraph.\n"
        "- Avoid generic filler phrases. Be spatially explicit and cinematic.\n\n"
        "### Output JSON (strict):\n"
        "[{\n"
        "  \"scene-index\": 1,\n"
        "  \"short_prompt\": \"...\",\n"
        "  \"long_prompt\": \"At the center of the room, a wooden table stands surrounded by three chairs; behind it, a tall shelf lines the wall, while a small lamp sits to the right on a low cabinet...\"\n"
        "}, ...]\n\n"
        "Return ONLY the JSON array. Do not include commentary or additional text."
    )


def build_director_feedback_prompt(dialogue_json: str, outline: str) -> str:
    return (
        "You are an experienced film director reviewing a screenplay dialogue draft. "
        "Your role is to evaluate the dramatic tension, pacing, and overall story flow.\n\n"
        "### Script Outline:\n"
        f"{outline}\n\n"
        "### Current Dialogue Draft:\n"
        f"{dialogue_json}\n\n"
        "### Your Task:\n"
        "Provide professional feedback focusing on:\n"
        "1. Dramatic tension: Are conflicts escalating properly? Is there emotional depth?\n"
        "2. Pacing: Does the dialogue flow naturally? Are there redundant exchanges?\n"
        "3. Scene coherence: Does each scene achieve its dialogue goal effectively?\n"
        "4. Story progression: Do the scenes connect logically and build toward a satisfying arc?\n\n"
        "### Output Format:\n"
        "Provide your feedback as plain text, organized by scene if needed. "
        "Be specific and constructive. Point out what works well and what needs improvement.\n"
    )


def build_actor_feedback_prompt(dialogue_json: str, characters: Sequence[Mapping[str, Any]]) -> str:
    roster = format_character_roster(characters)
    return (
        "You are a method actor reviewing dialogue from the perspective of character authenticity. "
        "Your role is to ensure each character's voice is distinct, believable, and consistent with their profile.\n\n"
        "### Character Profiles:\n"
        f"{roster}\n\n"
        "### Current Dialogue Draft:\n"
        f"{dialogue_json}\n\n"
        "### Your Task:\n"
        "Provide professional feedback focusing on:\n"
        "1. Character voice: Does each character sound unique and true to their personality?\n"
        "2. Speaking style consistency: Do the dialogues match the characters' defined speaking styles?\n"
        "3. Emotional authenticity: Are the characters' reactions and expressions believable?\n"
        "4. Character development: Do we see any growth or change in the characters through dialogue?\n\n"
        "### Output Format:\n"
        "Provide your feedback as plain text. Address specific character inconsistencies and suggest "
        "how to make dialogue more authentic for each character.\n"
    )


def build_actor_revision_prompt(original_dialogue: str, actor_feedback: str, outline: str) -> str:
    return (
        "You are a professional screenwriter revising your dialogue script based on feedback from the actors. "
        "Your goal is to incorporate their insights about character authenticity while maintaining the story's integrity.\n\n"
        "### Original Script Outline:\n"
        f"{outline}\n\n"
        "### Your Current Dialogue:\n"
        f"{original_dialogue}\n\n"
        "### Actor's Feedback:\n"
        f"{actor_feedback}\n\n"
        "### Your Task:\n"
        "Revise the dialogue by:\n"
        "1. Incorporating the actor's feedback on character authenticity and voice consistency\n"
        "2. Ensuring each character's speaking style matches their profile\n"
        "3. Making emotional expressions more believable and natural\n"
        "4. Maintaining the original story structure and dialogue goals\n\n"
        "### Requirements:\n"
        "   - Keep the same JSON structure as the original\n"
        "   - Maintain scene topics and plots unless absolutely necessary to change\n"
        "   - Focus on improving character voice and authenticity\n"
        "   - Do not add or remove scenes; only improve existing dialogue\n\n"
        "Your response should only contain the following JSON content:\n"
        "[{\"scene-topic\": \"...\",\n"
        "\"scene-plot\": \"...\",\n"
        "\"scene-dialogue\": [{\"speaker\": \"...\", \"content\": \"...\"}, ...]\n"
        "}, ...]\n"
    )


def build_director_review_prompt(revised_dialogue: str, actor_feedback: str, outline: str) -> str:
    return (
        "You are an experienced film director reviewing a revised screenplay dialogue. "
        "The screenwriter has revised the dialogue based on actor feedback. Your role is to evaluate whether "
        "these revisions are acceptable or if further changes are needed.\n\n"
        "### Script Outline:\n"
        f"{outline}\n\n"
        "### Actor's Original Feedback:\n"
        f"{actor_feedback}\n\n"
        "### Revised Dialogue (after actor feedback):\n"
        f"{revised_dialogue}\n\n"
        "### Your Task:\n"
        "Review the revised dialogue and determine:\n"
        "1. Overall quality: Has the revision improved the dialogue?\n"
        "2. Dramatic tension: Is the dramatic impact maintained or enhanced?\n"
        "3. Pacing: Does the dialogue still flow naturally?\n"
        "4. Story coherence: Do the scenes still achieve their goals effectively?\n\n"
        "### Decision:\n"
        "At the end of your review, you MUST explicitly state one of the following:\n"
        "- DECISION: APPROVED\n"
        "- DECISION: NEEDS REVISION\n\n"
        "If you request revision, provide specific, constructive feedback on what needs to be improved. "
        "Be clear about which scenes or dialogue exchanges need work and why.\n\n"
        "### Output Format:\n"
        "Provide your review as plain text, ending with your clear decision.\n"
    )


def build_director_revision_prompt(
    actor_revised_dialogue: str,
    director_feedback: str,
    actor_feedback: str,
    outline: str,
) -> str:
    return (
        "You are a professional screenwriter revising your dialogue script based on the director's feedback. "
        "You had previously revised the script based on actor feedback, but the director has requested further changes.\n\n"
        "### Original Script Outline:\n"
        f"{outline}\n\n"
        "### Your Actor-Revised Dialogue:\n"
        f"{actor_revised_dialogue}\n\n"
        "### Actor's Original Feedback:\n"
        f"{actor_feedback}\n\n"
        "### Director's Feedback (requesting changes):\n"
        f"{director_feedback}\n\n"
        "### Your Task:\n"
        "Revise the dialogue by:\n"
        "1. Addressing the director's concerns about dramatic tension, pacing, and story flow\n"
        "2. While still maintaining the character authenticity improvements from actor feedback\n"
        "3. Finding a balance between character voice and dramatic effectiveness\n"
        "4. Ensuring the story arc is compelling and satisfying\n\n"
        "### Requirements:\n"
        "   - Keep the same JSON structure as the original\n"
        "   - Maintain scene topics and plots\n"
        "   - Focus on addressing director's specific concerns\n"
        "   - Do not sacrifice character authenticity while improving dramatic impact\n"
        "   - Do not add or remove scenes; only improve existing dialogue\n\n"
        "Your response should only contain the following JSON content:\n"
        "[{\"scene-topic\": \"...\",\n"
        "\"scene-plot\": \"...\",\n"
        "\"scene-dialogue\": [{\"speaker\": \"...\", \"content\": \"...\"}, ...]\n"
        "}, ...]\n"
    )

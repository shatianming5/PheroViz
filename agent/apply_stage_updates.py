from pathlib import Path

path = Path('app/services/single_chain_runner.py')
text = path.read_text(encoding='utf-8')

# Ensure _llm_generate_slots returns response
old = "    try:\n        response = _get_llm_client().chat_json(messages)\n    except Exception as exc:  # noqa: BLE001\n        return {\"slots\": {}, \"notes\": f\"llm_error: {exc}\", \"prompt\": prompt}\n\n    slots = response.get(\"slots\") if isinstance(response, dict) else {}\n    if not isinstance(slots, dict):\n        slots = {}\n    clean_slots: Dict[str, str] = {}\n    for key, value in slots.items():\n        if isinstance(key, str) and isinstance(value, str) and value.strip():\n            clean_slots[key.strip()] = value.strip()\n    notes = response.get(\"notes\") if isinstance(response, dict) else \"\"\n    if not isinstance(notes, str):\n        notes = \"\"\n    return {\"slots\": clean_slots, \"notes\": notes, \"prompt\": prompt}\n"
new = "    try:\n        response = _get_llm_client().chat_json(messages)\n    except Exception as exc:  # noqa: BLE001\n        return {\"slots\": {}, \"notes\": f\"llm_error: {exc}\", \"prompt\": prompt, \"response\": None}\n\n    slots = response.get(\"slots\") if isinstance(response, dict) else {}\n    if not isinstance(slots, dict):\n        slots = {}\n    clean_slots: Dict[str, str] = {}\n    for key, value in slots.items():\n        if isinstance(key, str) and isinstance(value, str) and value.strip():\n            clean_slots[key.strip()] = value.strip()\n    notes = response.get(\"notes\") if isinstance(response, dict) else \"\"\n    if not isinstance(notes, str):\n        notes = \"\"\n    return {\"slots\": clean_slots, \"notes\": notes, \"prompt\": prompt, \"response\": response}\n"
if old not in text:
    raise SystemExit('target llm block not found')
text = text.replace(old, new, 1)

# Update _build_stage_prompt task descriptions
old_l1 = "        task = (\n            \"- 基于数据画像与业务意图完善 canvas/overlays/scales/layout/theme/flags。\\n\"\n            \"- `spec.compose` 必须返回完整 dict；`spec.theme_defaults` 可返回 `{}`。\\n\"\n            \"- 避免破坏已有字段，尽量增量调整。\\n\"\n        )"
new_l1 = "        task = (\n            \"- 基于数据画像与业务意图完善 canvas/overlays/scales/layout/theme/flags，并可添加多 overlay（如 bar+scatter/line）。\\n\"\n            \"- 支持右轴（设置 overlays[i].yaxis='right'）与断轴/范围控制（scales.y_left.breaks / range）。\\n\"\n            \"- `spec.compose` 必须返回完整 dict；`spec.theme_defaults` 返回主题默认值（可 `{}`），应包含字体、字号、色板、线宽、刻度策略。\\n\"\n            \"- 避免破坏已有字段，优先增量调整，必要时写明理由。\\n\"\n        )"
text = text.replace(old_l1, new_l1, 1)

old_l2 = "        task = (\n            \"- `data.prepare` 负责清洗与派生列，必须返回 DataFrame。\\n\"\n            \"- `data.aggregate` 仅在需要聚合时使用，否则返回原 DataFrame。\\n\"\n            \"- `data.encode` 产出绘图所需列，可直接 `return df`。\\n\"\n            \"- 禁止出现轴/图例/ matplotlib 绘图指令。\\n\"\n        )"
new_l2 = "        task = (\n            \"- `data.prepare` 负责清洗、缺失处理、类型转换（如 to_datetime）、派生列及过滤，必须返回 DataFrame。\\n\"\n            \"- `data.aggregate` 仅在需要聚合/topK 时使用；否则直接 `return df`。\\n\"\n            \"- `data.encode` 输出绘图所需列（如颜色/尺寸编码），若无需额外处理可 `return df`。\\n\"\n            \"- 禁止出现轴/图例/注释/主题或 Matplotlib 绘图指令。\\n\"\n        )"
text = text.replace(old_l2, new_l2, 1)

old_l3 = "        task = (\n            \"- 为 overlays 生成 marks.*，负责几何绘制与颜色/marker。\\n\"\n            \"- 按需设置 scales.* 与 colorbar.apply，确保对数轴/断轴安全。\\n\"\n            \"- 不得调用轴/图例/注释/主题相关函数。\\n\"\n        )"
new_l3 = "        task = (\n            \"- 为 overlays 生成 marks.*，负责几何绘制、调色与图例标签，必要时支持误差棒/CI/回归等增强。\\n\"\n            \"- 按需设置 scales.* 与 colorbar.apply，正确处理对数轴、安全断轴、双轴范围。\\n\"\n            \"- 不得调用轴/图例/注释/主题相关函数。\\n\"\n        )"
text = text.replace(old_l3, new_l3, 1)

old_l4 = "        task = (\n            \"- 设置标题、轴标签、刻度密度/旋转，整理 legend/grid。\\n\"\n            \"- 可添加 annot.* 与 theme.* 细节。\\n\"\n            \"- 禁止操作 data.* 或 marks.*。\\n\"\n        )"
new_l4 = "        task = (\n            \"- 设置标题四向、轴标签/单位、刻度密度与旋转，整理 legend/grid、spines。\\n\"\n            \"- 可添加 annot.reference_lines/bands/text 等注释，及 theme.* 微调（字体、配色、背景）。\\n\"\n            \"- 禁止操作 data.* 或 marks.*。\\n\"\n        )"
text = text.replace(old_l4, new_l4, 1)

# Include slot_keys in payloads
text = text.replace("payload_l1 = {\n            \"data_profile\": profile,\n            \"intent\": base_intent,\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n        }", "payload_l1 = {\n            \"data_profile\": profile,\n            \"intent\": base_intent,\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n            \"slot_keys\": [\"spec.compose\", \"spec.theme_defaults\"],\n        }")

text = text.replace("payload_l2 = {\n            \"df_head\": df.head(8).to_dict(orient=\"list\"),\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n        }", "payload_l2 = {\n            \"df_head\": df.head(8).to_dict(orient=\"list\"),\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n            \"slot_keys\": [\"data.prepare\", \"data.aggregate\", \"data.encode\"],\n        }")

text = text.replace("payload_l3 = {\n            \"dff_head\": df.head(8).to_dict(orient=\"list\"),\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n        }", "payload_l3 = {\n            \"dff_head\": df.head(8).to_dict(orient=\"list\"),\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n            \"slot_keys\": \"marks.*,scales.*,colorbar.apply\",\n        }")

text = text.replace("payload_l4 = {\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n        }", "payload_l4 = {\n            \"spec\": spec,\n            \"feedback\": feedback_text,\n            \"slot_keys\": \"axes.*,legend.apply,grid.apply,annot.*,theme.*\",\n        }")

# Ensure stage_logs store response snapshot from return
old_stage_log = "        stage_logs[layer] = {\n            \"prompt\": out.get(\"prompt\"),\n            \"response\": _snapshot(out.get(\"response\")),\n            \"payload\": _snapshot(payload),\n            \"notes\": out.get(\"notes\", \"\"),\n            \"raw_slots\": out.get(\"slots\", {}),\n            \"accepted_slots\": ok_layer,\n            \"rejected_slots\": rej_layer,\n        }"
if old_stage_log not in text:
    raise SystemExit('stage log block not found')
# Already capturing snapshot, no change needed

path.write_text(text, encoding='utf-8')

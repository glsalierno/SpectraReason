"""Interactive custom wavenumber range editor for product REPORT.html."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from reports.discussion_regions import default_ranges_config
from reports.paper_ftir_figures import format_download_links


RANGE_EDITOR_CSS = """
.range-editor-section { margin: 20px 0; padding: 14px 16px; border: 1px solid #dbeafe; border-radius: 10px; background: #f8fafc; }
.range-editor-section h3 { margin: 0 0 8px; }
.range-editor-hint { font-size: 0.85rem; color: #475569; margin: 0 0 10px; }
.range-editor-controls { overflow-x: auto; margin: 10px 0; }
.range-editor-controls table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
.range-editor-controls th, .range-editor-controls td { border: 1px solid #cbd5e1; padding: 4px 6px; }
.range-editor-controls th { background: #e2e8f0; }
.range-editor-controls input[type=text], .range-editor-controls input[type=number] { width: 72px; font-size: 0.82rem; }
.range-editor-controls input[type=color] { width: 36px; height: 24px; padding: 0; border: none; }
.range-editor-toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; align-items: center; }
.range-editor-btn { font-size: 0.78rem; padding: 3px 8px; cursor: pointer; }
.chunks-links-section h4 { margin: 14px 0 6px; font-size: 0.95rem; }
"""


def build_range_editor_section_html(
    *,
    editor_id: str,
    ranges_payload: dict[str, Any] | None = None,
    chunks_manifest: dict[str, Any] | None = None,
    report_dir: Path,
) -> str:
    payload = ranges_payload or default_ranges_config()
    report_dir = Path(report_dir)
    parts = [
        f"<section class='range-editor-section card' id='{html.escape(editor_id)}' "
        f"data-range-editor='1'>",
        "<h3>Custom Wavenumber Ranges / Chunks</h3>",
        "<p class='range-editor-hint'>Edit discussion windows for chunk plots, offset stacks, and collages. "
        "Download <code>ranges_config.json</code> and re-run with <code>--regions-file</code>.</p>",
        _range_table_html(editor_id, payload),
        "<div class='range-editor-toolbar'>",
        f"<button type='button' class='range-editor-btn' data-range-editor='{html.escape(editor_id)}' "
        f"data-action='add-range'>Add range</button>",
        f"<button type='button' class='range-editor-btn' data-range-editor='{html.escape(editor_id)}' "
        f"data-action='dup-range'>Duplicate selected</button>",
        f"<button type='button' class='range-editor-btn' data-range-editor='{html.escape(editor_id)}' "
        f"data-action='del-range'>Delete selected</button>",
        f"<button type='button' class='range-editor-btn' data-range-editor='{html.escape(editor_id)}' "
        f"data-action='download-ranges'>Download ranges_config.json</button>",
        f"<label><input type='file' accept='application/json,.json' "
        f"id='{html.escape(editor_id)}-file' style='max-width:220px'/> Load JSON</label>",
        "</div>",
        f"<script type='application/json' id='{html.escape(editor_id)}-data'>{json.dumps(payload)}</script>",
    ]
    if chunks_manifest:
        parts.append(_chunks_links_html(chunks_manifest, report_dir))
    parts.append("</section>")
    return "".join(parts)


def _range_table_html(editor_id: str, payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for i, row in enumerate(payload.get("ranges") or []):
        if not isinstance(row, dict):
            continue
        name = html.escape(str(row.get("name", f"range_{i}")))
        wn_min = float(row.get("wn_min", 400))
        wn_max = float(row.get("wn_max", 4000))
        color = html.escape(str(row.get("color", "#0072bd")))
        show = "checked" if row.get("show_in_stacks", True) else ""
        rows.append(
            f"<tr data-range-idx='{i}'>"
            f"<td><input type='checkbox' class='range-select'/></td>"
            f"<td><input type='text' class='range-name' value='{name}'/></td>"
            f"<td><input type='number' class='range-wn-min' value='{wn_min:.0f}' step='1'/></td>"
            f"<td><input type='number' class='range-wn-max' value='{wn_max:.0f}' step='1'/></td>"
            f"<td><input type='color' class='range-color' value='{color}'/></td>"
            f"<td><input type='checkbox' class='range-show-stacks' {show}/></td>"
            f"</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='6' class='muted'>No ranges defined.</td></tr>")
    return (
        f"<div class='range-editor-controls' data-range-editor-table='{html.escape(editor_id)}'>"
        f"<table><thead><tr><th></th><th>Name</th><th>Min cm⁻¹</th><th>Max cm⁻¹</th>"
        f"<th>Color</th><th>In stacks</th></tr></thead>"
        f"<tbody id='{html.escape(editor_id)}-tbody'>{''.join(rows)}</tbody></table></div>"
    )


def _chunks_links_html(manifest: dict[str, Any], report_dir: Path) -> str:
    parts = ["<div class='chunks-links-section'>", "<h4>Generated chunk exports</h4>"]
    extra: list[tuple[str, str]] = []
    rc = manifest.get("ranges_config_path")
    if rc:
        extra.append(("ranges_config.json", str(rc)))
    idx = manifest.get("index")
    if idx:
        extra.append(("Chunks index", str(idx)))
    cm = manifest.get("chunk_data") or {}
    for key, paths in cm.items():
        if isinstance(paths, dict):
            if paths.get("json"):
                extra.append((f"{key} JSON", paths["json"]))
            if paths.get("csv"):
                extra.append((f"{key} CSV", paths["csv"]))
    if extra:
        parts.append(f"<p class='paper-dl'>{format_download_links([], report_dir, extra=extra)}</p>")
    for key, paths in (manifest.get("outputs") or {}).items():
        if not paths:
            continue
        parts.append(f"<h4>{html.escape(key.replace('_', ' '))}</h4>")
        parts.append(f"<p class='paper-dl'>{format_download_links(list(paths), report_dir)}</p>")
    for mode, paths in (manifest.get("collage") or {}).items():
        parts.append(f"<h4>Collage — {html.escape(str(mode))}</h4>")
        parts.append(f"<p class='paper-dl'>{format_download_links(list(paths), report_dir)}</p>")
    parts.append("</div>")
    return "".join(parts)


RANGE_EDITOR_JS = r"""
<script>
(function () {
  function readPayload(editorId) {
    var el = document.getElementById(editorId + "-data");
    if (!el) return { range_set_name: "Custom FTIR discussion ranges", ranges: [] };
    try { return JSON.parse(el.textContent || "{}"); } catch (e) { return { ranges: [] }; }
  }
  function writePayload(editorId, payload) {
    var el = document.getElementById(editorId + "-data");
    if (el) el.textContent = JSON.stringify(payload);
  }
  function collectRanges(editorId) {
    var tbody = document.getElementById(editorId + "-tbody");
    if (!tbody) return [];
    var rows = tbody.querySelectorAll("tr[data-range-idx]");
    var out = [];
    rows.forEach(function (row) {
      var wmin = parseFloat(row.querySelector(".range-wn-min").value);
      var wmax = parseFloat(row.querySelector(".range-wn-max").value);
      out.push({
        name: row.querySelector(".range-name").value.trim() || "range",
        wn_min: isFinite(wmin) ? wmin : 400,
        wn_max: isFinite(wmax) ? wmax : 4000,
        color: row.querySelector(".range-color").value || "#0072bd",
        show_in_stacks: row.querySelector(".range-show-stacks").checked,
        label_policy: "selected_only"
      });
    });
    return out;
  }
  function renderTable(editorId, ranges) {
    var tbody = document.getElementById(editorId + "-tbody");
    if (!tbody) return;
    var html = "";
    ranges.forEach(function (r, i) {
      html += "<tr data-range-idx='" + i + "'>" +
        "<td><input type='checkbox' class='range-select'/></td>" +
        "<td><input type='text' class='range-name' value='" + (r.name || "") + "'/></td>" +
        "<td><input type='number' class='range-wn-min' value='" + r.wn_min + "' step='1'/></td>" +
        "<td><input type='number' class='range-wn-max' value='" + r.wn_max + "' step='1'/></td>" +
        "<td><input type='color' class='range-color' value='" + (r.color || "#0072bd") + "'/></td>" +
        "<td><input type='checkbox' class='range-show-stacks' " + (r.show_in_stacks !== false ? "checked" : "") + "/></td>" +
        "</tr>";
    });
    tbody.innerHTML = html || "<tr><td colspan='6'>No ranges</td></tr>";
  }
  function downloadRanges(editorId) {
    var payload = readPayload(editorId);
    payload.ranges = collectRanges(editorId);
    writePayload(editorId, payload);
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "ranges_config.json";
    a.click();
  }
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest("button[data-range-editor]");
    if (!btn) return;
    var editorId = btn.getAttribute("data-range-editor");
    var action = btn.getAttribute("data-action");
    var payload = readPayload(editorId);
    if (action === "download-ranges") return downloadRanges(editorId);
    payload.ranges = collectRanges(editorId);
    if (action === "add-range") {
      payload.ranges.push({ name: "new_range", wn_min: 1500, wn_max: 1800, color: "#0072bd", show_in_stacks: true, label_policy: "selected_only" });
    } else if (action === "dup-range") {
      var sel = document.querySelectorAll("#" + editorId + "-tbody tr .range-select:checked");
      if (sel.length) {
        var idx = parseInt(sel[0].closest("tr").getAttribute("data-range-idx"), 10);
        var copy = JSON.parse(JSON.stringify(payload.ranges[idx] || {}));
        copy.name = (copy.name || "range") + "_copy";
        payload.ranges.splice(idx + 1, 0, copy);
      }
    } else if (action === "del-range") {
      var sel2 = document.querySelectorAll("#" + editorId + "-tbody tr .range-select:checked");
      var remove = [];
      sel2.forEach(function (cb) { remove.push(parseInt(cb.closest("tr").getAttribute("data-range-idx"), 10)); });
      remove.sort(function (a,b){return b-a;});
      remove.forEach(function (i) { payload.ranges.splice(i, 1); });
    }
    writePayload(editorId, payload);
    renderTable(editorId, payload.ranges);
  });
  document.addEventListener("change", function (ev) {
    var input = ev.target;
    if (input.id && input.id.endsWith("-file") && input.files && input.files[0]) {
      var editorId = input.id.replace(/-file$/, "");
      var reader = new FileReader();
      reader.onload = function () {
        try {
          var payload = JSON.parse(reader.result);
          if (!payload.ranges) payload = { range_set_name: "Custom FTIR discussion ranges", ranges: payload };
          writePayload(editorId, payload);
          renderTable(editorId, payload.ranges || []);
        } catch (e) { alert("Invalid ranges JSON"); }
      };
      reader.readAsText(input.files[0]);
    }
  });
})();
</script>
"""


def range_editor_css() -> str:
    return RANGE_EDITOR_CSS


def range_editor_js() -> str:
    return RANGE_EDITOR_JS
